"""R72 source-only seam tests: no scientific imports, fits or native arrays."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/kernel_svm_runtime_gate.py'


def load_gate():
    spec = importlib.util.spec_from_file_location('r72_pure_gate_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelSVMSourceContracts(unittest.TestCase):
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

    def test_moderate_three_groups_have_real_contiguous_frame_purges(self):
        frames = [72_000_000 + 5 * i for i in range(300)]
        folds = self.gate.chronological_splits(frames)
        self.assertEqual([len(f['base_validation']) for f in folds], [100, 100, 100])
        self.assertEqual([len(f['base_train']) for f in folds], [150, 100, 150])
        self.assertEqual(folds[1]['base_train'], list(range(50)) + list(range(250, 300)))
        for fold in folds:
            self.assertTrue(all(abs(frames[t] - frames[v]) > 250
                for t in fold['base_train'] for v in fold['base_validation']))

    def test_mirrored_pair_roles_and_partition_survive_uneven_chunks(self):
        count = 302
        frames = [5 * i for i in range(count)]
        folds = self.gate.chronological_splits(frames)
        self.assertEqual([len(f['base_validation']) for f in folds], [101, 101, 100])
        self.assertEqual(sorted(i for f in folds for i in f['validation']), list(range(604)))
        for fold in folds:
            train, validation = set(fold['train']), set(fold['validation'])
            self.assertFalse(train & validation)
            for i in range(count):
                self.assertEqual(i in train, i + count in train)
                self.assertEqual(i in validation, i + count in validation)

    def test_purge_uses_actual_frames_instead_of_row_distance(self):
        dense = self.gate.chronological_splits([5 * i for i in range(300)])
        sparse = self.gate.chronological_splits([50 * i for i in range(300)])
        self.assertNotEqual(dense[1]['base_train'], sparse[1]['base_train'])
        self.assertEqual(len(sparse[1]['base_train']), 190)
        translated = self.gate.chronological_splits([10_000 + 5 * i for i in range(300)])
        for a, b in zip(dense, translated):
            self.assertEqual(a['train'], b['train'])
            self.assertEqual(a['validation'], b['validation'])

    def test_low_count_contiguous_groups_reject_instead_of_random_cv_fallback(self):
        with self.assertRaisesRegex(ValueError, 'empty after'):
            self.gate.chronological_splits([5 * i for i in range(32)])
        separated = self.gate.chronological_splits([1000 * i for i in range(32)])
        self.assertEqual([len(f['base_validation']) for f in separated], [11, 11, 10])
        self.assertTrue(all(f['base_train'] for f in separated))
        for invalid in ([0, 5, 5], [10, 5, 0], [0, 5], [0, 5, True]):
            with self.assertRaises(ValueError):
                self.gate.chronological_splits(invalid)

    def test_pipeline_preprocessing_is_within_author_calibration_not_precomputed(self):
        tree = ast.parse(SOURCE.read_text())
        fit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_with_author_recording')
        call = next(n for n in ast.walk(fit) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == 'CalibratedClassifierCV')
        kwargs = {k.arg: k.value for k in call.keywords}
        self.assertEqual(ast.unparse(kwargs['estimator']), 'make_estimator()')
        self.assertIs(ast.literal_eval(kwargs['ensemble']), False)
        self.assertEqual(ast.literal_eval(kwargs['n_jobs']), 1)
        self.assertEqual(ast.literal_eval(kwargs['method']), 'sigmoid')
        self.assertIsInstance(kwargs['cv'], ast.ListComp)
        pipeline = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_estimator')
        calls = [n for n in ast.walk(pipeline) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertEqual({n.func.id for n in calls},
            {'Pipeline', 'FunctionTransformer', 'SimpleImputer', 'StandardScaler', 'SVC', 'dict'})
        svc = next(n for n in calls if n.func.id == 'SVC')
        config = {k.arg: ast.literal_eval(k.value) for k in svc.keywords}
        self.assertEqual((config['C'], config['gamma'], config['kernel']), (1., 'scale', 'rbf'))
        self.assertNotIn('probability', config)

    def test_observers_delegate_author_methods_and_preserve_decision_method_name(self):
        tree = ast.parse(SOURCE.read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_with_author_recording')
        definitions = {n.name: n for n in owner.body if isinstance(n, ast.FunctionDef)}
        for name, original in (('observed_fit', 'original_fit'), ('observed_decision', 'original_decision')):
            function = definitions[name]
            self.assertEqual([ast.unparse(d) for d in function.decorator_list], ['functools.wraps(' + original + ')'])
            delegated = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == original]
            self.assertEqual(len(delegated), 1)
        text = ast.unparse(owner)
        self.assertNotIn('setattr(estimator', text)
        self.assertNotIn('estimator._audit', text)
        self.assertIn('len(fit_records) != 4', text)
        self.assertIn('sorted(margins) != [0, 1, 2]', text)

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

    def test_fixed_resource_shapes_and_environment_match_plan(self):
        gate = self.gate
        plan = json.loads(gate.PLAN.read_text())
        self.assertEqual(gate.CONTEXT_ROWS, {'moderate': 600, 'full': 3000})
        self.assertEqual((gate.QUERY_ROWS, gate.QUERY_SIZE, gate.PURGE_FRAMES), (7500, 512, 250))
        self.assertEqual((gate.FIT_SECONDS, gate.PREDICT_SECONDS, gate.WORKER_SECONDS), (300, 300, 900))
        self.assertEqual(plan['resources']['rss_bytes'], 6 * 1024**3)
        for name in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(gate.ENVIRONMENT[name], '1')
        self.assertEqual(gate.ENVIRONMENT['PYTHONDONTWRITEBYTECODE'], '1')


if __name__ == '__main__':
    unittest.main()
