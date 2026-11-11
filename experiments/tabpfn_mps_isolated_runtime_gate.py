"""R70: unchanged R67 numerical gate on project-local, checksum-bound Python.

The historical shared executable is explicitly superseded in this NEW protocol.
Old sources, manifests and their correctly failing runtime verifier remain intact.
No native inputs or scientific imports occur during preparation or verification.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'data/control_tabpfn_mps_isolated_runtime_v1'
PLAN = ROOT / 'experiments/TABPFN_MPS_ISOLATED_RUNTIME_PLAN.json'
SOURCE = ROOT / 'experiments/tabpfn_mps_allocator_flush_gate.py'
SOURCE_SHA = 'bf69d5c22fae4694a571d95d6668abff80f6c186d6eaa0b2af6ef96cba3ae073'
ISOLATED = ROOT / 'data/control_isolated_python_v1'
PYTHON = ISOLATED / 'python/bin/python3.12'
LEGACY = '/Users/derekko/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12'
LEGACY_SHA = 'd59f5251bce588951dc870bde45f0cd93de39327b1ed2431a8d26bfb567aac10'
CURRENT_SHA = 'ab9c532a75ce81eb96562bda04138e7a0b7528bd242c82a436ebeb1405aa361f'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def previous():
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen R67 source changed')
    spec = importlib.util.spec_from_file_location('r70_unchanged_r67', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_preserved():
    plan = json.loads(PLAN.read_text())
    if (plan['legacy_interpreter'] != LEGACY or plan['legacy_interpreter_sha256'] != LEGACY_SHA
            or plan['new_interpreter_sha256'] != CURRENT_SHA):
        raise ValueError('Declared executable migration differs')
    mapping = ROUND / 'preserved_bindings.json'
    if digest(mapping) != plan['preserved_binding_sha256']:
        raise ValueError('Preserved historical mapping changed')
    records = json.loads(mapping.read_text())
    if records.get(LEGACY) != LEGACY_SHA:
        raise ValueError('The one historical executable identity differs')
    for name, expected in records.items():
        if name == LEGACY:
            continue  # Obsolete shared executable; this protocol binds PYTHON below.
        if digest(ROOT / name) != expected:
            raise ValueError('Historical source/dependency/evidence changed: ' + name)
    identity = ISOLATED / 'acquisition.json'
    if digest(identity) != plan['isolated_acquisition_sha256']:
        raise ValueError('Isolated acquisition changed')
    isolation = json.loads(identity.read_text())
    if digest(PYTHON) != CURRENT_SHA or isolation['isolated_interpreter_sha256'] != CURRENT_SHA:
        raise ValueError('Isolated executable changed')
    for relative, expected in isolation['files_sha256'].items():
        if digest(ISOLATED / relative) != expected:
            raise ValueError('Isolated Python or stdlib changed: ' + relative)
    for relative, target in isolation['symlinks'].items():
        path = ISOLATED / relative
        if not path.is_symlink() or os.readlink(path) != target:
            raise ValueError('Isolated library/interpreter link changed')
    for name, expected in plan['isolated_venv_configuration_sha256'].items():
        if digest(ISOLATED / name / 'pyvenv.cfg') != expected:
            raise ValueError('Isolated environment configuration changed')
    return len(records)


def core():
    execution = previous().core()
    execution.ROUND = ROUND
    execution.PLAN = PLAN
    execution.PYTHON = PYTHON
    execution.__file__ = str(Path(__file__).resolve())
    execution.verify = verify
    if (execution.LIMIT_BYTES, execution.OPERATION_SECONDS, execution.WORKER_SECONDS) != (6 * 1024**3, 300, 900):
        raise ValueError('Original resource caps changed')
    return execution


def preflight():
    started = time.monotonic()
    compile(Path(__file__).read_bytes(), str(Path(__file__)), 'exec')
    count = verify_preserved()
    prior = previous()
    contract = prior.source_contract()
    execution = core()
    for name in ('worker', 'main', 'supervise', 'canonical_predict', 'checkpoint_fingerprint',
                 'synthetic_data', 'fixed_environment', 'runtime_paths'):
        if Path(getattr(execution, name).__code__.co_filename).resolve() != ROOT / 'experiments/tabpfn_mps_runtime_gate.py':
            raise ValueError('Original numerical/runtime function changed: ' + name)
    if Path(execution.operation.__code__.co_filename).resolve() != SOURCE:
        raise ValueError('Original source-supported cache release changed')
    hardware = json.loads((ROUND / 'hardware_probe.json').read_text())
    old_hardware = json.loads((ROOT / 'data/control_tabpfn_mps_allocator_flush_v1/hardware_probe.json').read_text())
    for key in ('recommended_max_memory_bytes', 'hash_probe', 'mps_allocator_limit_bytes',
                'reserved_owned_rss_budget_bytes', 'tabpfn_mps_memory_fraction'):
        if hardware[key] != old_hardware[key]:
            raise ValueError('Hardware, allocator or reproducibility setting changed')
    tests = ROOT / 'experiments/test_tabpfn_mps_allocator_flush.py'
    spec = importlib.util.spec_from_file_location('r70_exact_r67_pure_tests', tests)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if not result.wasSuccessful() or result.testsRun != 15:
        raise ValueError('Unchanged fifteen pure callback/source tests failed')
    if any(name in sys.modules for name in ('torch', 'numpy', 'scipy', 'sklearn', 'tabpfn', 'joblib')):
        raise ValueError('Scientific import occurred during source preparation')
    save(ROUND / 'source_preflight.json', dict(passed=True, seconds=time.monotonic()-started,
         implementation_sha256=digest(__file__), plan_sha256=digest(PLAN), pure_tests=15,
         historical_bindings=count, only_obsolete_executable_superseded=True,
         original_math_cache_callback_and_all_exact_checks=True, source_contract=contract,
         isolated_executable_sha256=digest(PYTHON), scientific_imports=False, model_calls=False))
    print('R70 source-only isolated-runtime preflight passed', flush=True)


def freeze():
    pre = json.loads((ROUND / 'source_preflight.json').read_text())
    review = json.loads((ROUND / 'root_review.json').read_text())
    for record in (pre, review):
        if record['implementation_sha256'] != digest(__file__) or record['plan_sha256'] != digest(PLAN):
            raise ValueError('ROOT review/preflight must bind final source and plan')
    if not pre['passed'] or not review['approved_for_freeze']:
        raise ValueError('ROOT review and pure preflight required')
    verify_preserved()
    files = {str(Path(__file__).relative_to(ROOT)): digest(__file__), str(PLAN.relative_to(ROOT)): digest(PLAN)}
    for path in sorted(ROUND.rglob('*')):
        if path.is_file() and 'protocol_freeze' not in path.parts and 'runtime_assessment' not in path.parts:
            files[str(path.relative_to(ROOT))] = digest(path)
    folder = ROUND / 'protocol_freeze'
    folder.mkdir(exist_ok=False)
    (folder / 'plan.json').write_bytes(PLAN.read_bytes())
    save(folder / 'implementation.json', files)
    save(folder / 'freeze.json', dict(implementation_sha256=digest(__file__), plan_sha256=digest(PLAN),
         old_R67_historical_pass_retained=True, new_isolated_runtime_four_stage_pass_required=True,
         numerical_parameters_changed=False, model_calls_started=False))
    print('Frozen R70 execution-only isolated runtime admission', flush=True)


def verify():
    folder = ROUND / 'protocol_freeze'
    if PLAN.read_bytes() != (folder / 'plan.json').read_bytes():
        raise ValueError('New isolated-runtime plan changed')
    for name, expected in json.loads((folder / 'implementation.json').read_text()).items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen R70 source/provenance changed: ' + name)
    verify_preserved()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    for name in ('preflight', 'freeze', 'verify', 'worker'):
        action.add_argument('--' + name, action='store_true')
    parser.add_argument('--size', choices=('moderate', 'full'))
    parser.add_argument('--stage', choices=('first', 'rebuild'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROUND / 'runtime_assessment').resolve()
    if not output.is_relative_to(ROUND.resolve()):
        parser.error('New outputs must remain inside R70')
    if args.preflight:
        preflight()
    elif args.freeze:
        freeze()
    elif args.verify:
        verify()
        print('R70 isolated runtime and preserved source bindings verified', flush=True)
    else:
        execution = core()
        if args.worker:
            if not args.size or not args.stage:
                parser.error('Worker requires size and stage')
            if Path(sys.base_prefix).resolve() != (ISOLATED / 'python').resolve() or Path(sys.executable).resolve() != PYTHON.resolve():
                raise ValueError('Model workers must execute the isolated interpreter')
            try:
                execution.worker(args.size, args.stage, output)
            except BaseException as error:
                if output.exists():
                    save(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                         synthetic_only=True, competition_inputs_read=False, historical_results_retained=True))
                raise
        else:
            execution.main(output)
