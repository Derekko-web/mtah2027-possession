"""R66: reallocate the same 6-GiB total budget; retain frozen R63 mathematics."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'data/control_tabpfn_mps_balanced_budget_v1'
PLAN = ROOT / 'experiments/TABPFN_MPS_BALANCED_BUDGET_PLAN.json'
PREVIOUS = ROOT / 'data/control_tabpfn_mps_validation_init_v1'
SOURCE = ROOT / 'experiments/tabpfn_mps_validation_init_gate.py'
SOURCE_SHA = '59823fca874ead58dbd1b8684b17647d2dea61cff53523466b70dde05657494f'
GIB = 1024**3


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
        raise ValueError('Frozen R63 source changed')
    spec = importlib.util.spec_from_file_location('r66_frozen_r63', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def budget_contract():
    original = json.loads((PREVIOUS / 'hardware_probe.json').read_text())
    expected = dict(original)
    del expected['mps_driver_allocation_budget_bytes']
    expected.update(mps_allocator_limit_bytes=4 * GIB,
                    reserved_owned_rss_budget_bytes=2 * GIB,
                    tabpfn_mps_memory_fraction=repr(4 * GIB / original['recommended_max_memory_bytes']),
                    derived_from_original_probe=True,
                    original_probe_sha256=digest(PREVIOUS / 'hardware_probe.json'),
                    new_hardware_probe_or_scientific_import=False)
    actual = json.loads((ROUND / 'hardware_probe.json').read_text())
    if actual != expected:
        raise ValueError('Prospective 4-GiB allocator / 2-GiB RSS allocation differs')
    if actual['recommended_max_memory_bytes'] != 12713115648 or 4 * GIB / actual['recommended_max_memory_bytes'] >= 1:
        raise ValueError('Unexpected hardware/fraction')
    if (ROUND / 'acquisition.json').read_bytes() != (PREVIOUS / 'acquisition.json').read_bytes():
        raise ValueError('Acquisition/model/dependency identities changed')
    return actual


def core():
    execution = previous().core()
    execution.ROUND = ROUND
    execution.PLAN = PLAN
    execution.__file__ = str(Path(__file__).resolve())
    execution.verify = verify
    if (execution.LIMIT_BYTES, execution.OPERATION_SECONDS, execution.WORKER_SECONDS) != (6 * GIB, 300, 900):
        raise ValueError('Original overall resource limits changed')
    return execution


def preflight():
    started = time.monotonic()
    compile(Path(__file__).read_bytes(), str(Path(__file__)), 'exec')
    prior = previous()
    prior.verify()
    budget = budget_contract()
    contract, validator = prior.source_contract()
    tests = prior.pure_tests(validator)
    execution = core()
    names = ('worker', 'main', 'supervise', 'operation', 'canonical_predict',
             'checkpoint_fingerprint', 'synthetic_data', 'fixed_environment', 'runtime_paths')
    for name in names:
        if Path(getattr(execution, name).__code__.co_filename).resolve() != ROOT / 'experiments/tabpfn_mps_runtime_gate.py':
            raise ValueError('Original numerical/runtime function changed: ' + name)
    if Path(execution.preprocessing_fingerprint.__code__.co_filename).resolve() != SOURCE:
        raise ValueError('Original R63 one-time full fingerprint boundary changed')
    if execution.fixed_environment()['TABPFN_MPS_MEMORY_FRACTION'] != budget['tabpfn_mps_memory_fraction']:
        raise ValueError('Pre-import allocator setting differs')
    if any(name in sys.modules for name in ('torch', 'numpy', 'scipy', 'sklearn', 'tabpfn', 'joblib')):
        raise ValueError('Scientific import occurred during pure preflight')
    save(ROUND / 'source_preflight.json', dict(passed=True, seconds=time.monotonic()-started,
         implementation_sha256=digest(__file__), plan_sha256=digest(PLAN), pure_tests=tests,
         source_contract=contract, budget=budget, original_function_paths_verified=list(names),
         original_full_fingerprint_and_numerical_functions_retained=True,
         prior_failed_rounds_verified=True, scientific_imports=False, model_calls=False))
    print('R66 pure source/budget preflight passed; no scientific imports', flush=True)


def freeze():
    review = json.loads((ROUND / 'independent_review.json').read_text())
    pre = json.loads((ROUND / 'source_preflight.json').read_text())
    for record in (review, pre):
        if record['implementation_sha256'] != digest(__file__) or record['plan_sha256'] != digest(PLAN):
            raise ValueError('Review/preflight must bind final source and plan')
    if not review['approved_for_freeze'] or not pre['passed']:
        raise ValueError('ROOT review and pure preflight required')
    previous().verify()
    budget_contract()
    checks = {str(Path(__file__).relative_to(ROOT)): digest(__file__), str(PLAN.relative_to(ROOT)): digest(PLAN)}
    for path in sorted(ROUND.rglob('*')):
        if 'protocol_freeze' not in path.parts and 'runtime_assessment' not in path.parts and path.is_file():
            if path.suffix == '.pyc' or '__pycache__' in path.parts:
                raise ValueError('Unexpected generated bytecode')
            checks[str(path.relative_to(ROOT))] = digest(path)
    # Retain all terminal R63 stage evidence, including the failed full context.
    for path in sorted((PREVIOUS / 'runtime_assessment').rglob('*')):
        if path.is_file():
            checks[str(path.relative_to(ROOT))] = digest(path)
    for name in ('protocol_freeze/implementation.json', 'protocol_freeze/freeze.json', 'runtime_gate.log'):
        path = PREVIOUS / name
        checks[str(path.relative_to(ROOT))] = digest(path)
    folder = ROUND / 'protocol_freeze'
    folder.mkdir(exist_ok=False)
    (folder / 'plan.json').write_bytes(PLAN.read_bytes())
    save(folder / 'implementation.json', checks)
    save(folder / 'freeze.json', dict(implementation_sha256=digest(__file__), plan_sha256=digest(PLAN),
         bound_files=len(checks), resource_allocation_only=True, allocator_limit_GiB=4,
         combined_rss_plus_full_driver_limit_GiB=6, original_math_and_full_raw_checks=True,
         prior_R63_remains_failed=True, model_calls_started=False))
    print('Frozen R66 allocation-only experiment; no model call', flush=True)


def verify():
    folder = ROUND / 'protocol_freeze'
    if PLAN.read_bytes() != (folder / 'plan.json').read_bytes():
        raise ValueError('Frozen R66 plan changed')
    for path, expected in json.loads((folder / 'implementation.json').read_text()).items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen R66 input changed: ' + path)
    previous().verify()
    budget_contract()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group()
    for name in ('preflight', 'freeze', 'verify', 'worker'):
        action.add_argument('--' + name, action='store_true')
    p.add_argument('--size', choices=('moderate', 'full'))
    p.add_argument('--stage', choices=('first', 'rebuild'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    output = (args.output or ROUND / 'runtime_assessment').resolve()
    if not output.is_relative_to(ROUND.resolve()):
        p.error('Artifacts must remain in R66')
    if args.preflight:
        preflight()
    elif args.freeze:
        freeze()
    elif args.verify:
        verify()
        print('R66 and prior frozen evidence verified', flush=True)
    else:
        execution = core()
        if args.worker:
            if not args.size or not args.stage:
                p.error('Worker requires size and stage')
            try:
                execution.worker(args.size, args.stage, output)
            except BaseException as error:
                if output.exists():
                    save(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                         synthetic_only=True, competition_inputs_read=False, prior_failures_retained=True))
                raise
        else:
            execution.main(output)
