"""R67 pure source/mocked-API contracts; no scientific imports or model calls."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/tabpfn_mps_allocator_flush_gate.py'


def load_gate():
    spec = importlib.util.spec_from_file_location('r67_pure_contract', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMPS:
    def __init__(self, events, *, driver_after=100, tensor_after=28, error=None):
        self.events = events
        self.driver = 300
        self.tensor = 28
        self.driver_after = driver_after
        self.tensor_after = tensor_after
        self.error = error

    def synchronize(self):
        self.events.append('synchronize')

    def empty_cache(self):
        self.events.append('empty_cache')
        if self.error:
            raise self.error
        self.driver = self.driver_after
        self.tensor = self.tensor_after

    def driver_allocated_memory(self):
        self.events.append('driver')
        return self.driver

    def current_allocated_memory(self):
        self.events.append('tensor')
        return self.tensor


class AllocatorFlushContracts(unittest.TestCase):
    def setUp(self):
        self.gate = load_gate()

    def test_import_and_source_contract_use_no_scientific_packages(self):
        before = set(sys.modules)
        contract = self.gate.source_contract()
        self.assertTrue(contract['installed_python_api_exact'])
        self.assertTrue(contract['live_buffer_release_forbidden'])
        self.assertTrue(contract['public_api_also_clears_framework_MPSGraph_compilation_cache'])
        self.assertTrue(contract['allocator_already_retries_release_on_oom'])
        self.assertTrue(contract['no_guarantee_that_flush_prevents_oom'])
        self.assertFalse({name.split('.')[0] for name in set(sys.modules) - before}
                         & {'torch', 'numpy', 'scipy', 'sklearn', 'joblib', 'tabpfn'})

    def test_no_top_level_scientific_import(self):
        tree = ast.parse(SOURCE.read_bytes())
        imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(name.name.split('.')[0] for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module.split('.')[0])
        self.assertFalse(imports & {'torch', 'numpy', 'scipy', 'sklearn', 'joblib', 'tabpfn'})

    def test_exact_one_flush_order_before_prediction(self):
        events, records = [], []
        mps = FakeMPS(events)
        def prediction():
            events.append('prediction')
            return 'result'
        wrapped = self.gate.flushed_callback(prediction, mps,
            lambda value: (events.append('record'), records.append(value)))
        self.assertEqual(wrapped(), 'result')
        self.assertEqual(events, ['synchronize', 'driver', 'tensor', 'empty_cache',
                                  'driver', 'tensor', 'record', 'prediction'])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['empty_cache_calls'], 1)
        self.assertEqual(records[0]['synchronization_calls'], 1)
        self.assertEqual(records[0]['driver_bytes_released'], 200)
        self.assertTrue(records[0]['live_tensor_bytes_unchanged'])

    def test_every_canonical_and_diagnostic_name_is_prediction(self):
        for name in ('original_0', 'original_512', 'swapped_0', 'repeat_first_0',
                     'diagnostic_prefix8', 'diagnostic_reverse512', 'future_prediction_name'):
            self.assertTrue(self.gate.is_prediction(name), name)
        for name in ('fit_context', 'fingerprint_initial_weights', 'fingerprint_final_weights',
                     'fingerprint_after_diagnostics'):
            self.assertFalse(self.gate.is_prediction(name), name)

    def test_fit_and_fingerprint_delegate_exact_callback_without_flush_or_import(self):
        callback = lambda: 'value'
        calls = []
        def delegate(output, name, passed):
            calls.append((output, name, passed))
            return passed(), 12.5
        with patch.object(self.gate, '_ORIGINAL_OPERATION', delegate):
            for name in ('fit_context', 'fingerprint_initial_weights', 'fingerprint_final_weights',
                         'fingerprint_after_diagnostics'):
                self.assertEqual(self.gate.allocator_flush_operation('output', name, callback), ('value', 12.5))
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(record[2] is callback for record in calls))

    def test_cleanup_is_inside_original_timed_callback_and_uses_output_argument(self):
        events = []
        fake_torch = types.SimpleNamespace(mps=FakeMPS(events))
        output = Path('/unrelated/native/reuse/output')
        records = []
        def delegate(actual_output, name, callback):
            self.assertEqual(actual_output, output)
            self.assertEqual(name, 'swapped_0')
            events.append('original_timer_monitor_start')
            result = callback()
            events.append('original_timer_monitor_end')
            return result, 7.0
        def record(actual_output, name, event):
            self.assertEqual(actual_output, output)
            records.append((name, event))
        with patch.dict(sys.modules, {'torch': fake_torch}), \
                patch.object(self.gate, '_ORIGINAL_OPERATION', delegate), \
                patch.object(self.gate, 'record_flush', record):
            value = self.gate.allocator_flush_operation(output, 'swapped_0',
                lambda: events.append('prediction') or 'result')
        self.assertEqual(value, ('result', 7.0))
        self.assertLess(events.index('original_timer_monitor_start'), events.index('synchronize'))
        self.assertLess(events.index('empty_cache'), events.index('prediction'))
        self.assertLess(events.index('prediction'), events.index('original_timer_monitor_end'))
        self.assertEqual(len(records), 1)

    def test_driver_reduction_is_diagnostic_and_never_selects_behavior(self):
        for after in (100, 300, 350):
            records, events = [], []
            callback = self.gate.flushed_callback(lambda: 'same', FakeMPS(events, driver_after=after), records.append)
            self.assertEqual(callback(), 'same')
            self.assertEqual(events.count('empty_cache'), 1)
            self.assertFalse(records[0]['driver_reduction_required'])

    def test_live_tensor_change_is_rejected_before_prediction(self):
        events, records = [], []
        callback = self.gate.flushed_callback(lambda: events.append('prediction'),
            FakeMPS(events, tensor_after=29), records.append)
        with self.assertRaisesRegex(ValueError, 'live MPS tensor'):
            callback()
        self.assertNotIn('prediction', events)
        self.assertEqual(events.count('empty_cache'), 1)
        self.assertFalse(records[0]['live_tensor_bytes_unchanged'])

    def test_cleanup_failure_never_calls_or_retries_prediction(self):
        events, records = [], []
        failure = RuntimeError('allocator failure')
        callback = self.gate.flushed_callback(lambda: events.append('prediction'),
            FakeMPS(events, error=failure), records.append)
        with self.assertRaises(RuntimeError) as raised:
            callback()
        self.assertIs(raised.exception, failure)
        self.assertNotIn('prediction', events)
        self.assertEqual(events.count('empty_cache'), 1)
        self.assertEqual(records[0]['status'], 'failed')
        self.assertFalse(records[0]['callback_called'])

    def test_prediction_failure_has_no_flush_rescue_or_retry(self):
        events, records = [], []
        failure = RuntimeError('prediction OOM')
        def prediction():
            events.append('prediction')
            raise failure
        callback = self.gate.flushed_callback(prediction, FakeMPS(events), records.append)
        with self.assertRaises(RuntimeError) as raised:
            callback()
        self.assertIs(raised.exception, failure)
        self.assertEqual(events.count('prediction'), 1)
        self.assertEqual(events.count('empty_cache'), 1)

    def test_counter_validation_rejects_invalid_bytes(self):
        mps = FakeMPS([])
        mps.driver = -1
        with self.assertRaisesRegex(ValueError, 'memory counter'):
            self.gate.memory_counters(mps)

    def test_output_log_is_relative_and_records_each_event_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.gate.record_flush(output, 'diagnostic_prefix8', {'status': 'complete'})
            lines = (output / 'allocator_flush.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), {'operation': 'diagnostic_prefix8', 'status': 'complete'})

    def test_core_preserves_original_model_math_timer_monitor_dispatch_and_limits(self):
        before = set(sys.modules)
        execution = self.gate.core()
        original = ROOT / 'experiments/tabpfn_mps_runtime_gate.py'
        self.assertEqual((execution.LIMIT_BYTES, execution.OPERATION_SECONDS, execution.WORKER_SECONDS),
                         (6 * 1024**3, 300, 900))
        self.assertIs(execution.operation, self.gate.allocator_flush_operation)
        self.assertEqual(Path(self.gate._ORIGINAL_OPERATION.__code__.co_filename).resolve(), original)
        for name in ('worker', 'main', 'supervise', 'canonical_predict', 'checkpoint_fingerprint',
                     'synthetic_data', 'fixed_environment', 'runtime_paths'):
            self.assertEqual(Path(getattr(execution, name).__code__.co_filename).resolve(), original)
        self.assertFalse({name.split('.')[0] for name in set(sys.modules) - before}
                         & {'torch', 'numpy', 'scipy', 'sklearn', 'joblib', 'tabpfn'})

    def test_exact_hardware_model_and_dependency_metadata_copies(self):
        hardware = self.gate.immutable_contract()
        self.assertEqual(hardware['mps_allocator_limit_bytes'], 4 * 1024**3)
        self.assertEqual(hardware['reserved_owned_rss_budget_bytes'], 2 * 1024**3)

    def test_cleanup_has_no_rng_or_model_context_mutation_api(self):
        tree = ast.parse(SOURCE.read_bytes())
        cleanup = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'flushed_callback')
        attrs = {n.attr for n in ast.walk(cleanup) if isinstance(n, ast.Attribute)}
        self.assertFalse(attrs & {'set_rng_state', 'manual_seed', 'seed', 'fit', 'transform',
                                 'load_state_dict', 'zero_', 'copy_', 'resize_', 'clear'})
        self.assertEqual(attrs & {'synchronize', 'empty_cache'}, {'synchronize', 'empty_cache'})


if __name__ == '__main__':
    unittest.main()
