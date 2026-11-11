"""Pure fake-frame/clock/role/source tests; no scientific imports or model calls."""
import ast
import copy
import importlib.util
import json
import hashlib
import math
import marshal
from pathlib import Path
import sys
import tempfile
import types
import unittest
import warnings
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/gaussian_process_runtime_gate_v2.py'

def load_gate():
    spec = importlib.util.spec_from_file_location('gp_v2_pure_gate_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class GaussianProcessSourceContracts(unittest.TestCase):
    def captured_codes(self):
        path = ROOT / 'data/control_gaussian_process_code_identity_diagnostic_v1/diagnostic.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            'a17d4613cadd52cd46b424761fb4d0cc5afa3094e92cf399e0a7e59be73ec825')
        value = json.loads(path.read_text())
        actual = bytes.fromhex(value['actual_default_marshal_hex'])
        pristine = bytes.fromhex(value['pristine_bound_default_marshal_hex'])
        self.assertNotEqual(actual, pristine)
        # Trusted captured author code is loaded as metadata, never executed.
        return marshal.loads(actual), marshal.loads(pristine), actual

    def test_captured_v1_false_alarm_retains_every_full_code_value(self):
        actual, pristine, raw = self.captured_codes()
        self.assertEqual(actual, pristine)
        self.assertEqual(self.gate.canonical_code_record(actual), self.gate.canonical_code_record(pristine))
        expected = self.gate.canonical_code_digest(pristine)
        self.gate.require_authored_code(actual, pristine, expected)
        record = self.gate.canonical_code_record(actual)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            names = {name for name in dir(actual) if name.startswith('co_') and not callable(getattr(actual, name))}
        self.assertEqual(set(record['all_noncallable_co_attributes']), names)
        for name in ('co_filename', 'co_qualname', 'co_flags', 'co_code', 'co_consts',
                     'co_firstlineno', 'co_linetable', 'co_exceptiontable', 'co_lnotab'):
            self.assertIn(name, record['all_noncallable_co_attributes'])
        self.assertIn('localsplusnames', record['complete_TYPE_CODE_payload'])
        self.assertIn('localspluskinds', record['complete_TYPE_CODE_payload'])
        self.assertEqual(record['complete_TYPE_CODE_payload'], self.gate.decoded_code_payload(raw))
        audit = self.gate.code_identity_audit(actual, pristine)
        self.assertIs(audit['semantic_comparison_allowed'], False)
        self.assertEqual(audit['actual_canonical_sha256'], audit['pristine_canonical_sha256'])

    def test_mutated_code_provenance_constants_and_executable_fields_reject(self):
        actual, pristine, _ = self.captured_codes()
        expected = self.gate.canonical_code_digest(pristine)
        mutations = dict(
            bytecode=actual.replace(co_code=actual.co_code+b'\x09\x00'),
            constants=actual.replace(co_consts=actual.co_consts+(920001,)),
            filename=actual.replace(co_filename=actual.co_filename+'.different'),
            firstline=actual.replace(co_firstlineno=actual.co_firstlineno+1),
            flags=actual.replace(co_flags=actual.co_flags^0x40),
            name=actual.replace(co_name=actual.co_name+'Different'),
            qualification=actual.replace(co_qualname=actual.co_qualname+'Different'),
            stacksize=actual.replace(co_stacksize=actual.co_stacksize+1),
            argcount=actual.replace(co_argcount=actual.co_argcount+1),
            positions=actual.replace(co_linetable=b''))
        for name, changed in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.gate.require_authored_code(changed, pristine, expected)
        with patch.object(self.gate.sys, 'gettrace', return_value=object()), self.assertRaises(ValueError):
            self.gate.require_authored_code(actual, pristine, expected)
        with self.assertRaises(TypeError):
            self.gate.canonical_code_record(actual.replace(co_consts=actual.co_consts+(object(),)))

    def test_hidden_locals_kind_and_name_payload_mutations_are_visible_without_execution(self):
        _, _, raw = self.captured_codes()
        parser = self.gate.CodePayloadParser(raw)
        baseline = parser.object()
        for path in ('code.localspluskinds', 'code.localsplusnames[0]'):
            event = next(item for item in parser.events if item['path']==path and 'payload_start' in item)
            modified = bytearray(raw)
            position = event['payload_start']
            modified[position] ^= 0x10 if path.endswith('kinds') else 0x20
            changed = self.gate.decoded_code_payload(bytes(modified))
            field = 'localspluskinds' if path.endswith('kinds') else 'localsplusnames'
            self.assertNotEqual(changed[field], baseline[field])
            for key in baseline:
                if key!=field:
                    self.assertEqual(changed[key], baseline[key])
        with self.assertRaises(ValueError):
            self.gate.decoded_code_payload(raw[:-1])

    def test_recursive_code_constants_are_canonical_and_changes_reject(self):
        compiled = compile('def outer():\n    def inner():\n        return 1\n    return inner\n',
                           '/pure/code_metadata_fixture.py', 'exec')
        outer = self.gate.find_code(compiled, ('outer',))
        inner = self.gate.find_code(compiled, ('outer', 'inner'))
        altered = inner.replace(co_consts=inner.co_consts+(1234,))
        changed = outer.replace(co_consts=tuple(altered if item is inner else item for item in outer.co_consts))
        with self.assertRaises(ValueError):
            self.gate.require_authored_code(changed, outer, self.gate.canonical_code_digest(outer))
        self.assertIsInstance(self.gate.canonical_code_record(outer), dict)

    def test_only_execution_admission_changed_and_raw_audit_is_nonsemantic(self):
        identity = self.gate.numerical_ast_identity()
        self.assertIs(identity['all_unchanged'], True)
        self.assertEqual(identity['only_removed_old_admission_statements'], 1)
        self.assertEqual(identity['only_removed_new_admission_statements'], 3)
        tree = ast.parse(SOURCE.read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name=='worker')
        assignment = next(node for node in ast.walk(worker)
            if isinstance(node, ast.Assign) and any(isinstance(target,ast.Name) and target.id=='exact_files'
                for target in node.targets))
        self.assertNotIn('code_identity_audit.json', ast.unparse(assignment))

    def trace_record(self, last_delta=0.0):
        previous = 2.0
        current = previous + last_delta
        return dict(zero_initialization=True, returned=True, return_temporaries=True,
            exception=False, break_iteration=2, history=[
                dict(iteration=0, lml=1., previous='-inf', delta='+inf'),
                dict(iteration=1, lml=previous, previous=1., delta=1.),
                dict(iteration=2, lml=current, previous=previous, delta=current-previous)])

    def test_fixed_convergence_tolerance_actual_break_and_rejection_cases(self):
        self.assertTrue(self.gate.validate_posterior_trace(self.trace_record()))
        self.assertTrue(self.gate.validate_posterior_trace(self.trace_record(-5e-11)))
        for mutation in ('cap', 'decrease', 'nonfinite', 'no_break', 'false_initial', 'exception', 'bad_previous'):
            value = self.trace_record()
            if mutation == 'cap':
                value['history'] *= 34
            elif mutation == 'decrease':
                value = self.trace_record(-2e-10)
            elif mutation == 'nonfinite':
                value['history'][-1]['lml'] = math.nan
            elif mutation == 'no_break':
                value['break_iteration'] = None
            elif mutation == 'false_initial':
                value['zero_initialization'] = False
            elif mutation == 'exception':
                value['exception'] = True
            else:
                value['history'][1]['previous'] = '-inf'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.gate.validate_posterior_trace(value)

    def test_exact_author_text_ast_and_pristine_compile_are_source_only(self):
        before = set(sys.modules)
        value = self.gate.posterior_source_contract()
        self.assertEqual((value['iteration_line'], value['condition_line'], value['break_line']), (467, 491, 492))
        self.assertEqual(value['expected_actual_calls'], 2)
        self.assertFalse({'numpy', 'scipy', 'sklearn'} & {name.split('.')[0] for name in set(sys.modules)-before})
        self.assertEqual(value['code_sha256'], self.gate.posterior_source_contract()['code_sha256'])

    def test_fake_author_frames_observe_only_exact_identity_and_actual_break(self):
        def pure_identity():
            return None
        code, estimator = pure_identity.__code__, object()
        lines = dict(iteration_line=10, condition_line=20, break_line=21)
        inspected = []
        def inspect(values, initial):
            inspected.append(initial)
            return dict(invented_only=True)
        observer = self.gate.PosteriorTrace(code, lines, estimator, inspect)
        unrelated = types.SimpleNamespace(f_code=(lambda: 3).__code__)
        self.assertIsNone(observer(unrelated, 'call', None))
        values = dict(self=estimator, return_temporaries=True, _=0)
        frame = types.SimpleNamespace(f_code=code, f_locals=values, f_lineno=0)
        self.assertIs(observer(frame, 'call', None), observer)
        frame.f_lineno = 10
        observer(frame, 'line', None)
        for index, current, previous in ((0, 1., -math.inf), (1, 2., 1.), (2, 2., 2.)):
            values.update(_=index, lml=current, log_marginal_likelihood=previous)
            frame.f_lineno = 20
            before = dict(values)
            observer(frame, 'line', None)
            self.assertEqual(values, before)
        frame.f_lineno = 21
        observer(frame, 'line', None)
        observer(frame, 'return', (2., 'fake_temporaries'))
        self.assertTrue(self.gate.validate_posterior_trace(observer.record))
        self.assertEqual(inspected, [True, False, False, False])
        self.assertIsNone(observer.frame)

    def test_trace_wrapper_delegates_once_and_never_rewrites_author_arithmetic(self):
        tree = ast.parse(SOURCE.read_text())
        fit = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'fit_with_author_recording')
        observer = next(node for node in fit.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'observed_mode')
        calls = [node for node in ast.walk(observer) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'original']
        self.assertEqual(len(calls), 1)
        self.assertEqual([ast.unparse(node) for node in observer.decorator_list], ['functools.wraps(original)'])
        source = ast.unparse(observer)
        self.assertIn('sys.settrace(previous_trace)', source)
        self.assertNotIn('frame.f_locals[', source)
        gp = next(node for node in ast.walk(fit) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == 'GaussianProcessClassifier')
        config = {key.arg: ast.unparse(key.value) for key in gp.keywords}
        self.assertEqual(config['optimizer'], 'None')
        self.assertEqual(config['max_iter_predict'], 'MAX_ITERATIONS')
        self.assertEqual(config['warm_start'], 'False')
        self.assertEqual(config['copy_X_train'], 'True')
        self.assertNotIn('CalibratedClassifierCV', SOURCE.read_text())
        self.assertNotIn('scipy.linalg', source)

    def test_population_variance_scale_and_balanced_paired_roles(self):
        self.assertEqual(self.gate.resolved_length_scale(1.), math.sqrt(83.))
        self.assertEqual(self.gate.resolved_length_scale(0.), math.sqrt(.5))
        self.assertEqual(self.gate.resolved_length_scale(100./166), math.sqrt(50.))
        for invalid in (-1., math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                self.gate.resolved_length_scale(invalid)
        for count in (600, 3000):
            roles = self.gate.training_roles(count)
            self.assertEqual(roles['training'], list(range(count)))
            self.assertEqual(roles['pairs'][0], [0, count//2])
            self.assertEqual(len(roles['pairs']), count//2)
            self.assertIsNone(roles['calibration'])
            self.assertFalse(set(roles['original']) & set(roles['swapped']))
        for count in (0, 1, 601, True):
            with self.assertRaises(ValueError):
                self.gate.training_roles(count)

    def test_rng_state_serialization_is_explicit_readonly_and_no_field_exclusion(self):
        tree = ast.parse(SOURCE.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'semantic_state')
        text = ast.unparse(function)
        self.assertIn('isinstance(value, np.random.RandomState)', text)
        self.assertIn('semantic_state(value.get_state(), trail)', text)
        self.assertIn('semantic_state(vars(value), trail)', text)
        self.assertNotIn('set_state(', text)
        self.assertNotIn('exclude', text)

    def test_dual_clock_operation_marks_active_before_callback_and_retains_both(self):
        gate = self.gate
        start = dict(monotonic_ns=100, utc_ns=1000)
        end = dict(monotonic_ns=200, utc_ns=2000)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            def callback():
                marker = json.loads((output/'operation.json').read_text())
                self.assertIs(marker['active'], True)
                self.assertEqual(marker['started'], start)
                return 'pure_result'
            with patch.object(gate, 'clock_pair', side_effect=[start, end]), patch.object(gate, 'peak_bytes', return_value=0):
                value, elapsed = gate.operation(output, gate.FIT_OPERATION, 300, callback)
            self.assertEqual(value, 'pure_result')
            self.assertEqual(elapsed, dict(monotonic_ns=100, utc_ns=1000))
            marker = json.loads((output/'operation.json').read_text())
            self.assertIs(marker['active'], False)
            self.assertEqual(marker['status'], 'complete')

    def test_utc_suspension_overrun_and_clock_rollback_abort_independently(self):
        gate = self.gate
        start = dict(monotonic_ns=0, utc_ns=0)
        marker = dict(name=gate.FIT_OPERATION, active=True, started=start, limit_seconds=300)
        self.assertIsNone(gate.clock_abort_reason(start, dict(monotonic_ns=1, utc_ns=1), marker))
        self.assertIn('300seconds', gate.clock_abort_reason(start,
            dict(monotonic_ns=1, utc_ns=300_000_000_000), marker))
        self.assertIn('300seconds', gate.clock_abort_reason(start,
            dict(monotonic_ns=300_000_000_000, utc_ns=1), marker))
        self.assertIn('900seconds', gate.clock_abort_reason(start,
            dict(monotonic_ns=1, utc_ns=900_000_000_000), {}))
        self.assertIn('negative', gate.clock_abort_reason(start, dict(monotonic_ns=1, utc_ns=-1), {}))
        self.assertFalse(gate.duration_ns_ok(300_000_000_000, 300))
        self.assertFalse(gate.duration_ns_ok(True, 300))

    def test_operation_completion_cannot_hide_utc_overrun(self):
        gate = self.gate
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.object(gate, 'clock_pair', side_effect=[dict(monotonic_ns=0, utc_ns=0),
                    dict(monotonic_ns=1, utc_ns=300_000_000_000)]), \
                    patch.object(gate, 'peak_bytes', return_value=0):
                with self.assertRaisesRegex(RuntimeError, 'UTC'):
                    gate.operation(output, gate.FIT_OPERATION, 300, lambda: 'pure')
            self.assertIs(json.loads((output/'operation.json').read_text())['active'], False)

    def test_exact_two_complete_operation_events_and_summary_binding(self):
        gate = self.gate
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.object(gate, 'clock_pair', side_effect=[dict(monotonic_ns=index, utc_ns=index*10)
                    for index in range(4)]), patch.object(gate, 'peak_bytes', return_value=0):
                _, fit = gate.operation(output, gate.FIT_OPERATION, 300, lambda: None)
                _, query = gate.operation(output, gate.QUERY_OPERATION, 300, lambda: None)
            summary = dict(fit_elapsed=fit, query_elapsed=query)
            gate.validate_operation_evidence(output, summary)
            with self.assertRaises(ValueError):
                gate.validate_operation_evidence(output, {**summary, 'fit_elapsed': {'utc_ns': 999, 'monotonic_ns': 999}})
            (output/'operation.json').write_text('{}')
            with self.assertRaises(ValueError):
                gate.validate_operation_evidence(output, summary)

    def test_fixed_plan_and_no_scientific_modules_after_suite(self):
        plan = json.loads(self.gate.PLAN.read_text())
        self.assertEqual(plan['synthetic_shapes'], {'moderate': [600, 166], 'full': [3000, 166]})
        self.assertEqual(plan['resources']['rss_bytes'], 6*1024**3)
        self.gate.assert_source_only()


    @classmethod
    def setUpClass(cls):
        cls.gate = load_gate()

    def test_private_base_worker_and_effective_fixed_hash_flags(self):
        base = self.gate.PYTHON_ROOT / 'python'
        flags = dict(safe_path=True, no_site=True, dont_write_bytecode=True,
            ignore_environment=False, hash_randomization=False)
        self.gate.require_worker_runtime(base, base, base / 'bin/python3.12', flags)
        for key, value in (('safe_path', False), ('no_site', False),
                ('dont_write_bytecode', False), ('ignore_environment', True), ('hash_randomization', True)):
            with self.subTest(flag=key), self.assertRaises(ValueError):
                self.gate.require_worker_runtime(base, base, base / 'bin/python3.12', {**flags, key: value})

    def test_wrong_worker_prefix_base_and_executable_are_rejected_separately(self):
        base = self.gate.PYTHON_ROOT / 'python'
        flags = dict(safe_path=True, no_site=True, dont_write_bytecode=True,
            ignore_environment=False, hash_randomization=False)
        valid = (base, base, base / 'bin/python3.12')
        for index in range(3):
            values = list(valid); values[index] = Path('/invented/shared/runtime')
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.gate.require_worker_runtime(*values, flags)

    def test_worker_environment_removes_inherited_python_path_redirects(self):
        result = self.gate.worker_environment(dict(PYTHONHOME='/invented/shared',
            PYTHONPATH='/invented/unbound', PYTHONUSERBASE='/invented/user',
            PYTHONHASHSEED='123', PATH='/retained/search/path'))
        self.assertNotIn('PYTHONHOME', result)
        self.assertNotIn('PYTHONUSERBASE', result)
        self.assertEqual(result['PYTHONPATH'], '')
        self.assertEqual(result['PYTHONHASHSEED'], '0')
        self.assertEqual(result['PATH'], '/retained/search/path')

    def test_source_contract_has_no_scientific_import_or_array_read(self):
        before = set(sys.modules)
        contract = self.gate.source_contract()
        imported = {name.split('.')[0] for name in set(sys.modules) - before}
        self.assertFalse(imported & {'numpy', 'scipy', 'sklearn', 'pandas', 'torch'})
        self.assertEqual(len(contract['schema']['columns']), 166)
        self.assertFalse(contract['schema']['actual_arrays_read'])

    def test_no_top_level_scientific_import_or_implicit_run(self):
        tree = ast.parse(SOURCE.read_text())
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split('.')[0])
        self.assertFalse(imported & {'numpy', 'scipy', 'sklearn', 'pandas', 'torch'})
        self.assertIn('add_mutually_exclusive_group(required=True)', SOURCE.read_text())
        self.assertNotIn('evaluation/', SOURCE.read_text())
        self.assertNotIn('read_label_csv', SOURCE.read_text())

    def test_exact_named_four_stages_are_required_not_just_four_true_flags(self):
        gate = self.gate
        records = [dict(size=size, stage=stage, passed=True, exit_code=0, aborted=None) for size, stage in gate.STAGES]
        self.assertTrue(gate.all_stages_pass(records))
        self.assertFalse(gate.all_stages_pass(records[:3]))
        self.assertFalse(gate.all_stages_pass([records[0]] * 4))
        self.assertFalse(gate.all_stages_pass(records[::-1]))
        records[1]['aborted'] = 'rss_over_6GiB'
        self.assertFalse(gate.all_stages_pass(records))

    def test_resource_abort_overrides_an_existing_pass_summary(self):
        gate = self.gate
        process = Mock(pid=720072)
        process.poll.return_value = None
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            destination = output / 'moderate/first'
            destination.mkdir(parents=True)
            (destination / 'summary.json').write_text('{"passed":true}')
            sample = types.SimpleNamespace(returncode=0, stdout=str(gate.LIMIT_BYTES // 1024 + 1))
            with patch.object(gate.subprocess, 'Popen', return_value=process), \
                    patch.object(gate.subprocess, 'run', return_value=sample):
                record = gate.supervise('moderate', 'first', output)
            process.terminate.assert_called_once()
            self.assertFalse(record['passed'])
            self.assertEqual(record['aborted'], 'rss_over_6GiB')

    def test_gate_stops_after_first_failure_and_cannot_claim_partial_pass(self):
        gate = self.gate
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'assessment'
            failed = dict(size='moderate', stage='first', passed=False, exit_code=1, aborted=None)
            with patch.object(gate, 'verify'), patch.object(gate, 'require_resource_admission'), \
                    patch.object(gate, 'supervise', return_value=failed) as supervisor:
                with self.assertRaises(SystemExit):
                    gate.run_gate(output)
            supervisor.assert_called_once_with('moderate', 'first', output)
            self.assertFalse(json.loads((output / 'summary.json').read_text())['passed'])

    def test_resource_admission_is_separate_from_freeze_and_precedes_imports(self):
        gate = self.gate
        with tempfile.TemporaryDirectory() as folder:
            location = Path(folder)
            (location / 'resource_admission.json').write_text('{"approved_for_model_calls":true}')
            with patch.object(gate, 'ROUND', location):
                with self.assertRaisesRegex(ValueError, 'resource admission'):
                    gate.require_resource_admission()
        tree = ast.parse(SOURCE.read_text())
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        calls = [n.value.func.id for n in worker.body if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)]
        self.assertEqual(calls[:2], ['verify', 'require_resource_admission'])


if __name__ == '__main__':
    unittest.main()
