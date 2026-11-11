"""Source-only preparation of one separately admitted invented Laplace-GPC gate.

No preparation/preflight/test/freeze/verification action imports science or reads
native data. Only frozen, independently reviewed, separately admitted workers do.
"""
import argparse
import ast
from contextlib import ExitStack
import functools
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import math
import marshal
import os
from pathlib import Path
import pickle
import resource
import socket
import struct
import subprocess
import sys
import time
import traceback
import types
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'data/control_gaussian_process_runtime_v1'
PLAN = ROOT / 'experiments/GAUSSIAN_PROCESS_RUNTIME_PLAN.json'
PYTHON_ROOT = ROOT / 'data/control_isolated_python_v1'
PYTHON = PYTHON_ROOT / 'head_runtime/bin/python'
PACKAGES = PYTHON_ROOT / 'head_runtime/lib/python3.12/site-packages'
SCHEMA = ROOT / 'experiments/modernnca_schema_contract.py'
AUTHOR_REVISION = '866c0f51e7560ef0303cbcc5f159df5382ea9e3f'
AUTHOR_SOURCES = {
    'sklearn/gaussian_process/_gpc.py': (39387, '90628fca68b26aac2c4f57305b6302c549952d3873e6651209ffcafb2c95ff3d'),
    'sklearn/gaussian_process/kernels.py': (85311, 'e985326714f38ab7a6c7cbe10a9eb3409fdcec198c1708de2194fe979bf54f7b'),
    'sklearn/gaussian_process/tests/test_gpc.py': (11927, 'ade3641cae0eb5508cab192b77d2d0854cf1b108b0d36e0fe595610ce2d73485'),
    'COPYING': (1532, '50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0'),
}
CONTEXT_ROWS = {'moderate': 600, 'full': 3000}
QUERY_ROWS, QUERY_SIZE = 7500, 512
FIT_SECONDS, PREDICT_SECONDS, WORKER_SECONDS = 300, 300, 900
LIMIT_BYTES = 6 * 1024**3
MAX_ITERATIONS, CONVERGENCE_TOLERANCE = 100, 1e-10
FIT_OPERATION = 'complete_outer_training_preprocessing_and_author_gp_fit'
QUERY_OPERATION = 'all_queries_repeat_symmetry_reload'
STAGES = (('moderate', 'first'), ('moderate', 'rebuild'), ('full', 'first'), ('full', 'rebuild'))
ENVIRONMENT = {name: '1' for name in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
ENVIRONMENT.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='',
    PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
SOURCE_FILES = ('experiments/gaussian_process_runtime_gate.py',
    'experiments/test_gaussian_process_source_contract.py',
    'experiments/GAUSSIAN_PROCESS_RUNTIME_PLAN.json',
    'experiments/GAUSSIAN_PROCESS_LEARNER_RESEARCH_20261101.md')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def assert_source_only():
    forbidden = {'numpy', 'scipy', 'sklearn', 'pandas', 'torch', 'xgboost', 'catboost'}
    if forbidden.intersection(sys.modules):
        raise ValueError('Scientific imports are forbidden for this source-only action')


def load_schema():
    spec = importlib.util.spec_from_file_location('gp_literal_schema', SCHEMA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.schema_contract()


def significant_files(folder):
    """Bind all non-bytecode bytes, including optional installed dependencies."""
    for path in sorted(Path(folder).rglob('*')):
        if '__pycache__' in path.parts or path.suffix == '.pyc' or path.name == '.DS_Store':
            continue
        if path.is_file():
            yield path


def runtime_snapshot():
    acquisition_path = PYTHON_ROOT / 'acquisition.json'
    acquisition = json.loads(acquisition_path.read_text())
    files = {str(acquisition_path.relative_to(ROOT)): digest(acquisition_path)}
    links = {}
    # Verify the copied Python/stdlib against its existing acquisition, not just
    # whatever bytes happen to be present when GP preparation starts.
    for relative, expected in acquisition['files_sha256'].items():
        path = PYTHON_ROOT / relative
        if digest(path) != expected:
            raise ValueError('Private Python acquisition changed: ' + relative)
    for folder in (PYTHON_ROOT / 'python', PYTHON_ROOT / 'head_runtime'):
        for path in significant_files(folder):
            files[str(path.relative_to(ROOT))] = digest(path)
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():
                links[str(path.relative_to(ROOT))] = os.readlink(path)
    for relative, target in acquisition['symlinks'].items():
        if relative.startswith(('python/', 'head_runtime/')):
            path = PYTHON_ROOT / relative
            if not path.is_symlink() or os.readlink(path) != target:
                raise ValueError('Private Python acquisition link changed: ' + relative)
    # pathlib does not recurse through the head runtime's site-packages symlink.
    # Inventory its entire existing target explicitly, including binary libraries.
    package_root = PACKAGES.resolve()
    for path in significant_files(package_root):
        files[str(path.relative_to(ROOT))] = digest(path)
    for path in sorted(package_root.rglob('*')):
        if path.is_symlink() and '__pycache__' not in path.parts:
            links[str(path.relative_to(ROOT))] = os.readlink(path)
    if len(files) < 1000 or digest(PYTHON.resolve()) != acquisition['isolated_interpreter_sha256']:
        raise ValueError('Incomplete private interpreter/runtime binding')
    return files, links


def prepare_metadata():
    assert_source_only()
    if (ROUND / 'source_metadata.json').exists() or (ROUND / 'protocol_freeze').exists():
        raise ValueError('Source metadata already exists; preparation cannot overwrite it')
    ROUND.mkdir(parents=True, exist_ok=True)
    sources = {}
    for relative, (length, expected) in AUTHOR_SOURCES.items():
        local = PACKAGES / relative
        if local.is_file() and digest(local) == expected:
            body = local.read_bytes()
        else:
            url = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/' + AUTHOR_REVISION + '/' + relative
            body = urllib.request.urlopen(url, timeout=30).read()
        if len(body) != length or hashlib.sha256(body).hexdigest() != expected:
            raise ValueError('Author text differs: ' + relative)
        destination = ROUND / 'public_sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        sources[str(destination.relative_to(ROOT))] = expected
    # Preserve all existing notices for these distributions, including bundled
    # libraries. Reading metadata/licenses does not import any scientific code.
    notices = {}
    for pattern in ('scikit_learn-*.dist-info', 'numpy-*.dist-info', 'scipy-*.dist-info',
                    'joblib-*.dist-info', 'threadpoolctl-*.dist-info'):
        matches = list(PACKAGES.glob(pattern))
        if len(matches) != 1:
            raise ValueError('Ambiguous existing distribution: ' + pattern)
        distribution = matches[0]
        for local in significant_files(distribution):
            if local.name != 'METADATA' and not any(
                    part.lower() == 'licenses' or 'license' in part.lower() or part == 'COPYING'
                    for part in local.relative_to(distribution).parts):
                continue
            destination = ROUND / 'notices' / distribution.name / local.relative_to(distribution)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(local.read_bytes())
            notices[str(destination.relative_to(ROOT))] = digest(destination)
    files, links = runtime_snapshot()
    write_json(ROUND / 'runtime_manifest.json', files)
    write_json(ROUND / 'runtime_symlinks.json', links)
    versions = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in metadata.distributions(path=[str(PACKAGES)])}
    write_json(ROUND / 'source_metadata.json', dict(author_revision=AUTHOR_REVISION,
        author_source_sha256=sources, notices_sha256=notices,
        posterior_source=posterior_source_contract(),
        runtime_files=len(files), runtime_symlinks=len(links),
        runtime_manifest_sha256=digest(ROUND / 'runtime_manifest.json'),
        runtime_symlinks_sha256=digest(ROUND / 'runtime_symlinks.json'), versions=versions,
        interpreter=str(PYTHON.relative_to(ROOT)), interpreter_sha256=digest(PYTHON.resolve()),
        schema=load_schema(), scientific_imports=False, model_calls=False,
        native_arrays_read=False, package_installation=False, frozen=False))
    plan = json.loads(PLAN.read_text())
    plan['metadata_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in
        (ROUND / 'source_metadata.json', ROUND / 'runtime_manifest.json', ROUND / 'runtime_symlinks.json')}
    write_json(PLAN, plan)
    assert_source_only()
    print('GP source metadata prepared; no scientific imports, model calls or freeze', flush=True)


def source_contract():
    plan = json.loads(PLAN.read_text())
    for name, expected in plan['metadata_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Source-only metadata changed: ' + name)
    acquired = json.loads((ROUND / 'source_metadata.json').read_text())
    if acquired['author_revision'] != AUTHOR_REVISION or acquired['scientific_imports']:
        raise ValueError('Unexpected source acquisition identity')
    for relative, (length, expected) in AUTHOR_SOURCES.items():
        path = ROUND / 'public_sources' / relative
        if path.stat().st_size != length or digest(path) != expected:
            raise ValueError('Retained author source changed: ' + relative)
        if relative.endswith('.py') and digest(PACKAGES / relative) != expected:
            raise ValueError('Installed author source changed: ' + relative)
    for relative, expected in acquired['notices_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Retained notice/metadata changed: ' + relative)
    posterior = posterior_source_contract()
    if posterior != acquired['posterior_source']:
        raise ValueError('Exact author posterior source/AST/compiler identity changed')
    schema = load_schema()
    if schema != acquired['schema']:
        raise ValueError('The exact 166-column source/swap contract changed')
    required = dict(cpu_threads=1, workers=1, fit_seconds=FIT_SECONDS,
        prediction_seconds=PREDICT_SECONDS, worker_seconds=WORKER_SECONDS, rss_bytes=LIMIT_BYTES)
    if plan['resources'] != required or plan['synthetic_shapes'] != {
            k: [n, 166] for k, n in CONTEXT_ROWS.items()}:
        raise ValueError('Prospective resources or invented shapes differ')
    return dict(author_revision=AUTHOR_REVISION, author_sources=AUTHOR_SOURCES,
        schema=schema, versions=acquired['versions'], resources=required,
        posterior_source=posterior)


def verify_runtime():
    expected_files = json.loads((ROUND / 'runtime_manifest.json').read_text())
    expected_links = json.loads((ROUND / 'runtime_symlinks.json').read_text())
    files, links = runtime_snapshot()
    if files != expected_files or links != expected_links:
        raise ValueError('Read-only private Python or installed dependency inventory changed')
    return len(files)


def preflight():
    assert_source_only()
    started = time.monotonic()
    contract = source_contract()
    count = verify_runtime()
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if path.suffix == '.py':
            compile(path.read_bytes(), str(path), 'exec')
    assert_source_only()
    write_json(ROUND / 'source_preflight.json', dict(passed=True, scientific_imports=False,
        model_calls=False, native_arrays_read=False, runtime_bound_files=count,
        plan_sha256=digest(PLAN), implementation_sha256=digest(__file__),
        source_sha256={name: digest(ROOT / name) for name in SOURCE_FILES},
        source_contract=contract, seconds=time.monotonic() - started))
    print('GP source-only preflight passed; no model work started', flush=True)


def freeze():
    assert_source_only()
    review_path = ROUND / 'independent_review.json'
    review = json.loads(review_path.read_text())
    if not review.get('approved_for_freeze') or review.get('unresolved_findings') != 0:
        raise ValueError('Independent root review of final source is required')
    for relative in SOURCE_FILES:
        if review['source_sha256'].get(relative) != digest(ROOT / relative):
            raise ValueError('Review does not bind final source: ' + relative)
    proof = json.loads((ROUND / 'source_preflight.json').read_text())
    if not proof['passed'] or proof['plan_sha256'] != digest(PLAN) or proof['implementation_sha256'] != digest(__file__):
        raise ValueError('A final matching source-only preflight is required')
    source_contract()
    verify_runtime()
    files = {name: digest(ROOT / name) for name in SOURCE_FILES}
    files[str(SCHEMA.relative_to(ROOT))] = digest(SCHEMA)
    files.update(load_schema()['source_sha256'])
    for path in significant_files(ROUND):
        if 'protocol_freeze' in path.parts or 'runtime_assessment' in path.parts:
            continue
        files[str(path.relative_to(ROOT))] = digest(path)
    folder = ROUND / 'protocol_freeze'
    folder.mkdir(exist_ok=False)
    (folder / 'plan.json').write_bytes(PLAN.read_bytes())
    write_json(folder / 'implementation.json', files)
    write_json(folder / 'freeze.json', dict(plan_sha256=digest(PLAN),
        implementation_sha256=digest(__file__), implementation_manifest_sha256=digest(folder / 'implementation.json'),
        independent_review_sha256=digest(review_path), scientific_imports=False,
        model_calls=False, native_arrays_read=False, resource_admission_required=True))
    assert_source_only()
    print('GP reviewed source frozen; separate model resource admission remains required', flush=True)


def verify():
    folder = ROUND / 'protocol_freeze'
    proof = json.loads((folder / 'freeze.json').read_text())
    if PLAN.read_bytes() != (folder / 'plan.json').read_bytes():
        raise ValueError('GP frozen plan changed')
    if digest(folder / 'implementation.json') != proof['implementation_manifest_sha256']:
        raise ValueError('GP frozen manifest changed')
    for relative, expected in json.loads((folder / 'implementation.json').read_text()).items():
        if digest(ROOT / relative) != expected:
            raise ValueError('GP frozen input changed: ' + relative)
    source_contract()
    verify_runtime()


def require_resource_admission():
    approval = json.loads((ROUND / 'resource_admission.json').read_text())
    if (approval.get('approved_for_model_calls') is not True or
            approval.get('scope') != 'invented_only_gaussian_process_v1' or
            approval.get('predecessor_r73_closed_or_passed') is not True or
            approval.get('other_model_workers_stopped') is not True or
            approval.get('freeze_sha256') != digest(ROUND / 'protocol_freeze/freeze.json')):
        raise ValueError('Separate matching root resource admission is required before model imports')


def runtime_paths():
    base = PYTHON_ROOT / 'python/lib'
    sys.path[:] = [str(ROOT / 'experiments'), str(PACKAGES.resolve()),
        str(base / 'python312.zip'), str(base / 'python3.12'), str(base / 'python3.12/lib-dynload')]
    sys.dont_write_bytecode = True


def require_worker_runtime(prefix, base_prefix, executable, flags):
    expected = PYTHON_ROOT / 'python'
    if (Path(prefix).resolve() != expected.resolve() or Path(base_prefix).resolve() != expected.resolve()
            or Path(executable).resolve() != (expected / 'bin/python3.12').resolve()):
        raise ValueError('Model worker must execute the bound private Python base')
    if (not all(flags.get(name) for name in ('safe_path', 'no_site', 'dont_write_bytecode'))
            or flags.get('ignore_environment') is not False or flags.get('hash_randomization') is not False):
        raise ValueError('Model worker requires -P -S -B and effective PYTHONHASHSEED=0')


def worker_environment(environ):
    result = {name: value for name, value in environ.items() if not name.startswith('PYTHON')}
    result.update(ENVIRONMENT)
    return result


def initialize():
    require_worker_runtime(sys.prefix, sys.base_prefix, sys.executable,
        {name: bool(getattr(sys.flags, name)) for name in
            ('safe_path', 'no_site', 'dont_write_bytecode', 'ignore_environment', 'hash_randomization')})
    for name, value in ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise ValueError('Fixed worker environment is missing: ' + name)
    runtime_paths()
    import numpy as np
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info, threadpool_limits
    expected = source_contract()['versions']
    for name, module in (('numpy', np), ('scipy', scipy), ('scikit-learn', sklearn)):
        if module.__version__ != expected[name] or not Path(module.__file__).resolve().is_relative_to(PACKAGES.resolve()):
            raise ValueError('Scientific package escaped the bound runtime: ' + name)
    if 'torch' in sys.modules:
        raise ValueError('GP is a CPU scikit-learn gate, without Torch')
    return np, threadpool_limits, threadpool_info


def array_record(value):
    import numpy as np
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError('Object arrays cannot enter a canonical learned-state record')
    return dict(shape=list(array.shape), dtype=array.dtype.str,
        sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def semantic_state(value, trail=None):
    """Exact semantic bytes for all fitted __dict__ fields; never repr(address)."""
    import numpy as np
    trail = set() if trail is None else trail
    if isinstance(value, np.ndarray):
        return {'array': array_record(value)}
    if isinstance(value, np.generic):
        return {'numpy_scalar': array_record(value)}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {'float64_bits': struct.pack('>d', value).hex()}
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if isinstance(value, np.random.RandomState):
        return {'numpy_RandomState': semantic_state(value.get_state(), trail)}
    if isinstance(value, np.dtype):
        return {'numpy_dtype': value.str}
    if isinstance(value, type):
        return {'type': value.__module__ + '.' + value.__qualname__}
    if callable(value) and not hasattr(value, 'get_params'):
        module = getattr(value, '__module__', None)
        name = getattr(value, '__qualname__', getattr(value, '__name__', None))
        if module is None or name is None:
            raise TypeError('Unidentified callable in semantic state: ' + type(value).__name__)
        return {'callable': module + '.' + name}
    identity = id(value)
    if identity in trail:
        raise ValueError('Unexpected cyclic fitted state')
    trail = trail | {identity}
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise TypeError('Fitted dictionaries require explicit string keys')
        return {key: semantic_state(item, trail) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return {'tuple' if isinstance(value, tuple) else 'list': [semantic_state(x, trail) for x in value]}
    if hasattr(value, '__dict__'):
        return {'class': type(value).__module__ + '.' + type(value).__qualname__,
            'dict': semantic_state(vars(value), trail)}
    raise TypeError('Unhandled fitted-state type: ' + type(value).__module__ + '.' + type(value).__qualname__)


def state_digest(value):
    state = semantic_state(value)
    payload = json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def probability_checks(value, count):
    import numpy as np
    if (value.shape != (count, 2) or not np.isfinite(value).all() or (value < 0).any() or
            (value > 1).any() or not np.allclose(value.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Invalid H/A probability array')


def canonical_predict(model, query):
    import numpy as np
    result = []
    for start in range(0, len(query), QUERY_SIZE):
        batch = np.ascontiguousarray(query[start:start + QUERY_SIZE], dtype=np.float64)
        count = len(batch)
        if count < QUERY_SIZE:
            batch = np.concatenate((batch, np.repeat(batch[-1:], QUERY_SIZE - count, axis=0)))
        value = model.predict_proba(batch)
        probability_checks(value, QUERY_SIZE)
        result.append(value[:count])
    return np.ascontiguousarray(np.concatenate(result), dtype=np.float64)


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def all_stages_pass(records):
    return ([(r.get('size'), r.get('stage')) for r in records] == list(STAGES)
        and all(r.get('passed') is True and r.get('exit_code') == 0 and r.get('aborted') is None for r in records))


def run_gate(output):
    verify()
    require_resource_admission()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for size, stage in STAGES:
        records.append(supervise(size, stage, output))
        if not records[-1]['passed']:
            break
    passed = all_stages_pass(records)
    if passed:
        for size in CONTEXT_ROWS:
            summary = json.loads((output / size / 'rebuild/summary.json').read_text())
            if summary.get('checks', {}).get('independent_refit_exact') is not True:
                raise ValueError('Final gate lacks an independent complete refit')
    write_json(output / 'summary.json', dict(passed=passed, records=records,
        plan_sha256=digest(PLAN), implementation_sha256=digest(__file__),
        synthetic_only=True, native_processing_started=False, native_plan_authorized=False))
    if not passed:
        raise SystemExit(1)


def find_code(parent, names):
    value = parent
    for name in names:
        matches = [item for item in value.co_consts
                   if isinstance(item, types.CodeType) and item.co_name == name]
        if len(matches) != 1:
            raise ValueError('Author code object is not uniquely identified: ' + name)
        value = matches[0]
    return value


def pristine_posterior_code():
    path = PACKAGES.resolve() / 'sklearn/gaussian_process/_gpc.py'
    compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True,
                       optimize=sys.flags.optimize)
    return find_code(compiled, ('_BinaryGaussianProcessClassifierLaplace', '_posterior_mode'))


def posterior_source_contract():
    """Inspect/compile author text without executing/importing it."""
    path = PACKAGES.resolve() / 'sklearn/gaussian_process/_gpc.py'
    if digest(path) != AUTHOR_SOURCES['sklearn/gaussian_process/_gpc.py'][1]:
        raise ValueError('Author posterior source hash changed')
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == '_BinaryGaussianProcessClassifierLaplace')
    functions = {node.name: node for node in owner.body if isinstance(node, ast.FunctionDef)}
    posterior = functions['_posterior_mode']
    loops = [node for node in ast.walk(posterior) if isinstance(node, ast.For)]
    if len(loops) != 1 or ast.unparse(loops[0].iter) != 'range(self.max_iter_predict)':
        raise ValueError('Author posterior loop changed')
    loop = loops[0]
    stops = [node for node in loop.body if isinstance(node, ast.If)
             and ast.unparse(node.test) == 'lml - log_marginal_likelihood < 1e-10']
    if len(stops) != 1 or len(stops[0].body) != 1 or not isinstance(stops[0].body[0], ast.Break):
        raise ValueError('Author convergence branch changed')
    # This is author execution flow, not a reimplementation: one likelihood
    # call enters mode-finding and fit then enters it again for prediction state.
    fit = functions['fit']
    fit_posterior = [node for node in ast.walk(fit) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == '_posterior_mode']
    likelihood_posterior = [node for node in ast.walk(functions['log_marginal_likelihood'])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == '_posterior_mode']
    if len(fit_posterior) != 1 or len(likelihood_posterior) != 1:
        raise ValueError('Author two-call fit flow changed')
    code = pristine_posterior_code()
    return dict(source_sha256=digest(path), code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest(),
        first_line=posterior.lineno, iteration_line=loop.body[0].lineno,
        condition_line=stops[0].lineno, break_line=stops[0].body[0].lineno,
        expected_iterations=MAX_ITERATIONS, tolerance=CONVERGENCE_TOLERANCE,
        direct_fit_posterior_calls=1, likelihood_posterior_calls=1,
        expected_actual_calls=2, source_only=True)


def validate_posterior_trace(record):
    """Pure scalar admission rule. No learned/data-based tolerance or retry."""
    history = record.get('history', [])
    if (record.get('zero_initialization') is not True or record.get('returned') is not True
            or record.get('return_temporaries') is not True or record.get('exception') is not False
            or len(history) < 2 or len(history) >= MAX_ITERATIONS
            or record.get('break_iteration') != len(history) - 1):
        raise ValueError('Posterior cap/initialization/actual-break/return contract failed')
    for ordinal, entry in enumerate(history):
        current, previous, delta = entry['lml'], entry['previous'], entry['delta']
        if entry['iteration'] != ordinal or not math.isfinite(current):
            raise ValueError('Posterior history is nonfinite or not contiguous')
        if ordinal == 0:
            if previous != '-inf' or delta != '+inf':
                raise ValueError('First author likelihood previous value must be explicit -inf')
        elif (not all(type(value) in (int, float) and math.isfinite(value)
                      for value in (previous, delta)) or previous != history[ordinal - 1]['lml']
              or delta != current - previous or delta < -CONVERGENCE_TOLERANCE):
            raise ValueError('Posterior likelihood decreased substantially or has invalid history')
        should_break = ordinal > 0 and delta < CONVERGENCE_TOLERANCE
        if should_break != (ordinal == len(history) - 1):
            raise ValueError('Observed history differs from the exact author stopping predicate')
    if not -CONVERGENCE_TOLERANCE <= history[-1]['delta'] < CONVERGENCE_TOLERANCE:
        raise ValueError('Final posterior delta is outside the fixed roundoff/convergence interval')
    return True


class PosteriorTrace:
    """Read exact author frames; never assign locals or estimator state."""
    def __init__(self, code, contract, estimator, inspect_arrays):
        self.code, self.contract = code, contract
        self.estimator, self.inspect_arrays = estimator, inspect_arrays
        self.frame = None
        self.record = dict(history=[], returned=False, exception=False,
                           zero_initialization=False, return_temporaries=False,
                           break_iteration=None)

    def __call__(self, frame, event, argument):
        if frame.f_code is not self.code:
            return None
        if event == 'call':
            if self.frame is not None or frame.f_locals.get('self') is not self.estimator:
                raise ValueError('Unexpected or nested author posterior frame')
            self.frame = frame
            self.record['return_temporaries'] = frame.f_locals.get('return_temporaries') is True
        elif frame is not self.frame:
            raise ValueError('Author trace frame identity changed')
        if event == 'line' and frame.f_lineno == self.contract['iteration_line']:
            if frame.f_locals['_'] == 0:
                self.record['initial_array'] = self.inspect_arrays(frame.f_locals, initial=True)
                self.record['zero_initialization'] = True
        if event == 'line' and frame.f_lineno == self.contract['condition_line']:
            values = frame.f_locals
            iteration = values['_']
            current, previous = float(values['lml']), float(values['log_marginal_likelihood'])
            if not math.isfinite(current):
                raise ValueError('Nonfinite actual posterior likelihood')
            first = iteration == 0 and previous == -math.inf
            entry = dict(iteration=iteration, lml=current,
                previous='-inf' if first else previous,
                delta='+inf' if first else current - previous,
                arrays=self.inspect_arrays(values, initial=False))
            self.record['history'].append(entry)
        if event == 'line' and frame.f_lineno == self.contract['break_line']:
            self.record['break_iteration'] = frame.f_locals['_']
        if event == 'exception':
            self.record['exception'] = True
        if event == 'return':
            self.record['returned'] = argument is not None
            self.frame = None
        return self


def training_roles(rows):
    """Pure paired-role contract; no temporal calibration is added."""
    if type(rows) is not int or rows <= 0 or rows % 2:
        raise ValueError('A positive even augmented row count is required')
    half = rows // 2
    return dict(original=list(range(half)), swapped=list(range(half, rows)),
        pairs=[[index, index + half] for index in range(half)],
        training=list(range(rows)), queries='feature_only_all7500', calibration=None)


def resolved_length_scale(variance, columns=166):
    if type(variance) not in (float, int) or not math.isfinite(variance) or variance < 0 or columns != 166:
        raise ValueError('Invalid fixed-column training variance')
    ell = math.sqrt(columns * variance / 2) if variance else math.sqrt(.5)
    if not math.isfinite(ell) or ell <= 0:
        raise ValueError('Invalid resolved fixed RBF length scale')
    return ell


def synthetic_data(size):
    import numpy as np
    schema, rows = load_schema(), CONTEXT_ROWS[size]
    generator = np.random.default_rng(740042)
    base = generator.normal(size=(rows // 2, 166)).astype(np.float64)
    query = generator.normal(size=(QUERY_ROWS, 166)).astype(np.float64)
    permutation = np.asarray(schema['permutation'], dtype=np.int64)
    if schema['signs'] != [1] * 166:
        raise ValueError('Actual schema requires a pure team permutation')
    h, a = [schema['columns'].index(team + '__distance_1') for team in ('h', 'a')]
    label = (base[:, h] - base[:, a] > 0).astype(np.int64)
    base.flat[::37], query.flat[::41] = np.nan, np.nan
    empty = schema['columns'].index('h__set_latent_31')
    constant = schema['columns'].index('h__set_latent_30')
    for values in (base, query):
        values[:, [empty, permutation[empty]]] = np.nan
        values[:, [constant, permutation[constant]]] = 2.5
        values[0, h], values[0, a] = np.inf, -np.inf
    context = np.ascontiguousarray(np.concatenate((base, base[:, permutation])), dtype=np.float64)
    target = np.concatenate((label, 1 - label))
    data = dict(raw_context=context, target=target, raw_query=np.ascontiguousarray(query),
        permutation=permutation, base_frames=np.asarray(
            [74_000_000 + 5 * index for index in range(rows // 2)], dtype=np.int64))
    if np.bincount(target).tolist() != [rows // 2, rows // 2]:
        raise ValueError('Invented paired classes are not balanced')
    for value in data.values():
        value.flags.writeable = False
    return data


class FittedGP:
    """Own preprocessing owner; all GP mathematics delegates to the author."""
    def __init__(self, imputer, scaler, classifier):
        self.imputer, self.scaler, self.classifier = imputer, scaler, classifier

    def predict_proba(self, raw):
        import numpy as np
        transformed = np.nan_to_num(raw, copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan)
        transformed = self.scaler.transform(self.imputer.transform(transformed))
        return self.classifier.predict_proba(np.ascontiguousarray(transformed, dtype=np.float64))


def fit_with_author_recording(data, output):
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process._gpc import _BinaryGaussianProcessClassifierLaplace
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    context, target = data['raw_context'], data['target']
    imputer = SimpleImputer(strategy='median', add_indicator=False, keep_empty_features=True)
    finite_or_nan = np.nan_to_num(context, copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan)
    imputed = imputer.fit_transform(finite_or_nan)
    scaler = StandardScaler()
    transformed = np.ascontiguousarray(scaler.fit_transform(imputed), dtype=np.float64)
    if transformed.shape != context.shape or not np.isfinite(transformed).all():
        raise ValueError('Training-only preprocessing changed columns or produced nonfinite values')
    variance = float(transformed.var())
    ell = resolved_length_scale(variance)
    kernel = ConstantKernel(1.0, constant_value_bounds='fixed') * RBF(ell, length_scale_bounds='fixed')
    classifier = GaussianProcessClassifier(kernel=kernel, optimizer=None, n_restarts_optimizer=0,
        max_iter_predict=MAX_ITERATIONS, warm_start=False, copy_X_train=True,
        random_state=42, multi_class='one_vs_rest', n_jobs=1)
    contract = posterior_source_contract()
    original = _BinaryGaussianProcessClassifierLaplace._posterior_mode
    code = original.__code__
    if (marshal.dumps(code) != marshal.dumps(pristine_posterior_code())
            or hashlib.sha256(marshal.dumps(code)).hexdigest() != contract['code_sha256']
            or sys.gettrace() is not None):
        raise ValueError('Author executable code identity or exclusive trace scope differs')
    records = []

    def inspect_arrays(values, initial):
        names = ('f',) if initial else ('f', 'pi', 'W', 'W_sr', 'W_sr_K', 'B', 'L', 'b', 'a')
        result = {}
        for name in names:
            value = values[name]
            if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
                raise ValueError('Actual posterior array is nonfinite: ' + name)
            result[name] = array_record(value)
        if initial and np.count_nonzero(values['f']) != 0:
            raise ValueError('Author posterior did not start at zero')
        return result

    @functools.wraps(original)
    def observed_mode(estimator, K, return_temporaries=False):
        if len(records) >= 2 or estimator.warm_start is not False:
            raise ValueError('Unexpected posterior invocation or warm start')
        if K.shape != (len(context), len(context)) or not np.isfinite(K).all():
            raise ValueError('Invalid actual training kernel')
        before = array_record(K)
        observer = PosteriorTrace(code, contract, estimator, inspect_arrays)
        previous_trace = sys.gettrace()
        if previous_trace is not None:
            raise ValueError('An unrelated active trace prevents exact posterior observation')
        sys.settrace(observer)
        try:
            result = original(estimator, K, return_temporaries=return_temporaries)
        finally:
            sys.settrace(previous_trace)
        validate_posterior_trace(observer.record)
        if before != array_record(K) or not np.isfinite(estimator.f_cached).all():
            raise ValueError('Posterior mutated its kernel or retained a nonfinite cache')
        records.append(dict(ordinal=len(records), kernel=before,
                            cached_f=array_record(estimator.f_cached), **observer.record))
        return result

    before = dict(X=array_record(transformed), y=array_record(target))
    with patch.object(_BinaryGaussianProcessClassifierLaplace, '_posterior_mode', observed_mode):
        classifier.fit(transformed, target)
    if len(records) != 2 or before != dict(X=array_record(transformed), y=array_record(target)):
        raise ValueError('Exactly two genuine immutable-input posterior calculations are required')
    base = classifier.base_estimator_
    if (not np.array_equal(classifier.classes_, [0, 1]) or not np.array_equal(base.classes_, [0, 1])
            or base.kernel_.n_dims != 0 or base.optimizer is not None
            or base.kernel_.k1.constant_value != 1.0 or base.kernel_.k2.length_scale != ell
            or not math.isfinite(classifier.log_marginal_likelihood())
            or np.shares_memory(base.X_train_, transformed)):
        raise ValueError('Actual binary fixed-kernel GP recipe differs')
    for value in (imputer.statistics_, scaler.mean_, scaler.var_, scaler.scale_,
                  base.X_train_, base.y_train_, base.pi_, base.W_sr_, base.L_, base.f_cached):
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite complete learned state')
    model = FittedGP(imputer, scaler, classifier)
    write_json(output / 'posterior_traces.json', records)
    write_json(output / 'fit_record.json', dict(actual_author_fit=True, actual_posterior_calls=2,
        roles=training_roles(len(context)), raw_X=array_record(context), y=array_record(target),
        transformed_X=array_record(transformed), transformed_variance_bits=struct.pack('>d', variance).hex(),
        length_scale_bits=struct.pack('>d', ell).hex(), unit_amplitude=1.0, optimizer=None,
        class_order=[0, 1], posterior_contract=contract, semantic_sha256=state_digest(model)))
    write_json(output / 'full_state.json', semantic_state(model))
    return model


def query_and_reload_checks(model, data, output):
    import numpy as np
    query, permutation = data['raw_query'], data['permutation']
    before = state_digest(model)
    original = canonical_predict(model, query)
    swapped = canonical_predict(model, query[:, permutation])
    owner = (original + swapped[:, ::-1]) / 2
    probability_checks(owner, len(query))
    for name, value in (('original_probability', original), ('swapped_probability', swapped), ('owner', owner)):
        np.save(output / (name + '.npy'), value, allow_pickle=False)
    first_swap = model.predict_proba(np.ascontiguousarray(query[:QUERY_SIZE, permutation]))
    first_original = model.predict_proba(np.ascontiguousarray(query[:QUERY_SIZE]))
    if not np.array_equal(first_swap, swapped[:QUERY_SIZE]) or not np.array_equal(first_original, original[:QUERY_SIZE]):
        raise ValueError('Repeated first canonical raw probabilities differ')
    swapped_owner = (first_swap + first_original[:, ::-1]) / 2
    if not np.array_equal(swapped_owner, owner[:QUERY_SIZE, ::-1]):
        raise ValueError('Independent team symmetry differs')
    prefix = model.predict_proba(np.ascontiguousarray(query[:8]))
    reverse = model.predict_proba(np.ascontiguousarray(query[:QUERY_SIZE][::-1]))[::-1]
    probability_checks(prefix, 8)
    probability_checks(reverse, QUERY_SIZE)
    diagnostics = dict(prefix_max_abs=float(np.max(np.abs(prefix - original[:8]))),
        reverse_max_abs=float(np.max(np.abs(reverse - original[:QUERY_SIZE]))), selection_allowed=False)
    if state_digest(model) != before:
        raise ValueError('All queries mutated complete fitted GP/preprocessor/RNG state')
    with (output / 'gp_model.pkl').open('xb') as handle:
        pickle.dump(model, handle, protocol=5)
    with (output / 'gp_model.pkl').open('rb') as handle:
        reloaded = pickle.load(handle)
    if state_digest(reloaded) != before:
        raise ValueError('Complete learned semantic state differs on checkpoint reload')
    for name, expected, values in (('original', original, query), ('swap', swapped, query[:, permutation])):
        actual = canonical_predict(reloaded, values)
        if not np.array_equal(actual, expected):
            raise ValueError('Complete canonical raw query schedule differs after reload: ' + name)
    if state_digest(reloaded) != before:
        raise ValueError('Reloaded fitted state changed during all queries')
    return diagnostics


def duration_ns_ok(value, seconds):
    return type(value) is int and 0 <= value < seconds * 1_000_000_000


def clock_pair():
    return dict(monotonic_ns=time.monotonic_ns(), utc_ns=time.time_ns())


def clock_elapsed(start, end):
    return {name: end[name] - start[name] for name in ('monotonic_ns', 'utc_ns')}


def clock_abort_reason(start, now, marker):
    elapsed = clock_elapsed(start, now)
    if any(type(value) is not int or value < 0 for value in elapsed.values()):
        return 'worker_clock_negative_or_invalid'
    if any(value >= WORKER_SECONDS * 1_000_000_000 for value in elapsed.values()):
        return 'worker_over_900seconds_monotonic_or_UTC'
    if marker:
        if (marker.get('name') not in (FIT_OPERATION, QUERY_OPERATION)
                or marker.get('limit_seconds') != 300 or type(marker.get('active')) is not bool):
            return 'invalid_atomic_operation_marker'
        if marker['active']:
            try:
                elapsed = clock_elapsed(marker['started'], now)
            except (KeyError, TypeError):
                return 'invalid_operation_clock_marker'
            if any(type(value) is not int or value < 0 for value in elapsed.values()):
                return 'operation_clock_negative_or_invalid'
            if any(value >= 300 * 1_000_000_000 for value in elapsed.values()):
                return marker['name'] + '_over_300seconds_monotonic_or_UTC'
    return None


def operation(output, name, limit, callback):
    if name not in (FIT_OPERATION, QUERY_OPERATION) or limit != 300:
        raise ValueError('Only the two prospectively fixed complete operations are permitted')
    started = clock_pair()
    marker = dict(name=name, active=True, started=started, limit_seconds=limit)
    write_json(output / 'operation.json', marker)
    status = 'failed'
    try:
        result = callback()
        status = 'complete'
    finally:
        ended = clock_pair()
        elapsed = clock_elapsed(started, ended)
        record = dict(marker, active=False, ended=ended, elapsed=elapsed, status=status)
        write_json(output / 'operation.json', record)
        with (output / 'operations.jsonl').open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    if any(not duration_ns_ok(value, limit) for value in elapsed.values()) or peak_bytes() > LIMIT_BYTES:
        raise RuntimeError('Complete operation exceeded its monotonic/UTC/memory bound: ' + name)
    return result, elapsed


def validate_operation_evidence(output, summary):
    events = [json.loads(line) for line in (output / 'operations.jsonl').read_text().splitlines()]
    if len(events) != 2:
        raise ValueError('Exactly two complete atomic UTC/monotonic operation records are required')
    for event, name, field in zip(events, (FIT_OPERATION, QUERY_OPERATION), ('fit_elapsed', 'query_elapsed')):
        if (event.get('name') != name or event.get('active') is not False
                or event.get('status') != 'complete' or event.get('limit_seconds') != 300
                or event.get('elapsed') != clock_elapsed(event['started'], event['ended'])
                or summary.get(field) != event.get('elapsed')
                or any(not duration_ns_ok(value, 300) for value in event['elapsed'].values())):
            raise ValueError('Incomplete/inconsistent/out-of-bound complete operation evidence')
    if json.loads((output / 'operation.json').read_text()) != events[-1]:
        raise ValueError('Atomic operation marker differs from final complete event')


def worker(size, stage, output):
    verify()
    require_resource_admission()
    output.mkdir(parents=True, exist_ok=False)
    blocked = RuntimeError('Network disabled in the invented-only fixed GP worker')
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=blocked))
        stack.enter_context(patch.object(socket.socket, 'connect_ex', side_effect=blocked))
        stack.enter_context(patch.object(urllib.request, 'urlopen', side_effect=blocked))
        np, limits, thread_info = initialize()
        stack.enter_context(limits(limits=1))
        pools = thread_info()
        if any(pool['num_threads'] != 1 for pool in pools):
            raise ValueError('A numerical pool is not single threaded')
        write_json(output / 'threadpools.json', pools)
        data = synthetic_data(size)
        arrays_before = {name: array_record(value) for name, value in data.items()}
        for name, value in data.items():
            np.save(output / (name + '.npy'), value, allow_pickle=False)
        write_json(output / 'recipe.json', dict(shape=[CONTEXT_ROWS[size], 166], query_rows=QUERY_ROWS,
            query_batch=QUERY_SIZE, seed=42, generator_seed=740042, frame_spacing=5,
            class_order=['H', 'A'], calibration=None, source_contract=source_contract(),
            arrays=arrays_before, roles=training_roles(CONTEXT_ROWS[size]),
            preprocessing='Outer-train-only nonfiniteNaN, median/empty0, standardscale/constant1',
            gp='fixed amplitude1 RBF ell=sqrt(166*Z.var()/2), zero variance sqrt(.5); optimizerNone'))
        model, fit_elapsed = operation(output, FIT_OPERATION, FIT_SECONDS,
            lambda: fit_with_author_recording(data, output))
        final_before = state_digest(model)
        diagnostics, query_elapsed = operation(output, QUERY_OPERATION, PREDICT_SECONDS,
            lambda: query_and_reload_checks(model, data, output))
        if final_before != state_digest(model) or arrays_before != {name: array_record(value) for name, value in data.items()}:
            raise ValueError('Complete fitted state or invented raw arrays changed')
        exact_files = ['recipe.json', 'fit_record.json', 'full_state.json', 'posterior_traces.json'] + [
            name + '.npy' for name in ('original_probability', 'swapped_probability', 'owner')]
        if stage == 'rebuild':
            for name in exact_files:
                if (output / name).read_bytes() != (output.parent / 'first' / name).read_bytes():
                    raise ValueError('Independent fresh-process complete refit differs: ' + name)
        checks = dict(one_actual_author_fit=True, two_actual_posterior_calls=True,
            actual_convergence_observed=True, finite_normalized_probabilities=True,
            exact_first_repeat=True, exact_team_swap=True, raw_arrays_immutable=True,
            full_fitted_preprocessing_kernel_RNG_state_immutable=True,
            semantic_reload_exact=True, canonical_reload_exact=True,
            independent_refit_exact=stage == 'rebuild')
        summary = dict(passed=True, size=size, stage=stage, synthetic_only=True,
            native_arrays_read=False, checks=checks, fit_elapsed=fit_elapsed, query_elapsed=query_elapsed,
            diagnostics=diagnostics, peak_bytes=peak_bytes(), exact_artifacts=exact_files,
            artifact_sha256={path.name: digest(path) for path in output.iterdir()
                if path.is_file() and path.name not in ('operation.json', 'summary.json')})
        validate_operation_evidence(output, summary)
        write_json(output / 'summary.json', summary)
    verify()


def supervise(size, stage, output):
    started, peak, aborted = clock_pair(), 0, None
    destination = output / size / stage
    destination.parent.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = [destination.parent / (stage + suffix)
        for suffix in ('.stdout.log', '.stderr.log')]
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        process = subprocess.Popen([str(PYTHON), '-B', '-P', '-S', str(Path(__file__).resolve()),
            '--worker', '--size', size, '--stage', stage, '--output', str(destination)],
            cwd=ROOT, env=worker_environment(os.environ), stdout=stdout, stderr=stderr)
        while process.poll() is None:
            sample = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)],
                                    capture_output=True, text=True)
            if sample.returncode == 0 and sample.stdout.strip():
                peak = max(peak, int(sample.stdout.strip()) * 1024)
            marker_path = destination / 'operation.json'
            try:
                marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
                aborted = clock_abort_reason(started, clock_pair(), marker)
            except (ValueError, TypeError, KeyError):
                aborted = 'invalid_atomic_operation_or_clock_evidence'
            if peak > LIMIT_BYTES:
                aborted = 'rss_over_6GiB'
            if aborted:
                process.terminate()
                break
            time.sleep(.1)
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=10)
    ended = clock_pair()
    elapsed = clock_elapsed(started, ended)
    summary_path = destination / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if aborted is None:
        aborted = clock_abort_reason(started, ended, {})
    if aborted is None and max(peak, summary.get('peak_bytes', 0)) > LIMIT_BYTES:
        aborted = 'rss_over_6GiB'
    if aborted is None and code == 0 and summary.get('passed') is True:
        try:
            validate_operation_evidence(destination, summary)
        except (ValueError, KeyError, TypeError, OSError):
            aborted = 'missing_invalid_or_out_of_bound_complete_operation_evidence'
    record = dict(size=size, stage=stage, exit_code=code, aborted=aborted,
        passed=aborted is None and code == 0 and summary.get('passed') is True,
        largest_sampled_rss_bytes=peak, started=started, ended=ended, elapsed=elapsed,
        summary_sha256=digest(summary_path) if summary_path.exists() else None,
        stdout_sha256=digest(stdout_path), stderr_sha256=digest(stderr_path))
    write_json(destination.parent / (stage + '.supervision.json'), record)
    print(size, stage, 'passed', record['passed'], flush=True)
    return record


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare-metadata', 'preflight', 'freeze', 'verify', 'run', 'worker'):
        action.add_argument('--' + flag, action='store_true')
    parser.add_argument('--size', choices=tuple(CONTEXT_ROWS))
    parser.add_argument('--stage', choices=('first', 'rebuild'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROUND / 'runtime_assessment').resolve()
    if not output.is_relative_to(ROUND.resolve()) or output == ROUND.resolve():
        parser.error('All GP assessment outputs must be a new child of its own directory')
    if args.prepare_metadata:
        prepare_metadata()
    elif args.preflight:
        preflight()
    elif args.freeze:
        freeze()
    elif args.verify:
        verify()
        assert_source_only()
        print('GP frozen sources verified without scientific imports', flush=True)
    elif args.run:
        run_gate(output)
    else:
        if not args.size or not args.stage:
            parser.error('An admitted worker requires explicit size and stage')
        try:
            worker(args.size, args.stage, output)
        except BaseException as error:
            if output.exists():
                write_json(output / 'failure.json', dict(error=repr(error),
                    traceback=traceback.format_exc(), synthetic_only=True, native_arrays_read=False))
            raise


if __name__ == '__main__':
    cli()
