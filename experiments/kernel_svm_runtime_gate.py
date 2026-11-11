"""R72: source-only preparation and an explicitly admitted invented RBF-SVM gate.

Preparation, tests, preflight, freeze and verification use the standard library.
Only a reviewed, frozen, resource-admitted worker imports scientific packages.
No native observation, label, checkpoint, scorer or selected prediction is read.
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
ROUND = ROOT / 'data/control_kernel_svm_runtime_v1'
PLAN = ROOT / 'experiments/KERNEL_SVM_RUNTIME_PLAN.json'
PYTHON_ROOT = ROOT / 'data/control_isolated_python_v1'
PYTHON = PYTHON_ROOT / 'head_runtime/bin/python'
PACKAGES = PYTHON_ROOT / 'head_runtime/lib/python3.12/site-packages'
SCHEMA = ROOT / 'experiments/modernnca_schema_contract.py'
AUTHOR_REVISION = '866c0f51e7560ef0303cbcc5f159df5382ea9e3f'
AUTHOR_SOURCES = {
    'sklearn/svm/_classes.py': (68141, '06b89760e22b5df441820f531a9989edfdf22bd5e405a68c5a33fb39e60644b9'),
    'sklearn/svm/_base.py': (44790, 'be4d5a830d357be28e2a85d6c80c208d8aa8536858a62e17cad50abf0f7fe62f'),
    'sklearn/calibration.py': (61543, '52339970c65ed05602a8515ebbbdacb2fef1ef24fc683cbace7683fd964269e9'),
    'sklearn/svm/src/libsvm/svm.cpp': (69309, '8e9c41f2e01ac93e07ece96fe8943699cfb7f0323e071d7321fb0032a666c3d6'),
    'COPYING': (1532, '50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0'),
}
CONTEXT_ROWS = {'moderate': 600, 'full': 3000}
QUERY_ROWS = 7500
QUERY_SIZE = 512
PURGE_FRAMES = 250
FIT_SECONDS = 300
PREDICT_SECONDS = 300
WORKER_SECONDS = 900
LIMIT_BYTES = 6 * 1024**3
STAGES = (('moderate', 'first'), ('moderate', 'rebuild'), ('full', 'first'), ('full', 'rebuild'))
ENVIRONMENT = {name: '1' for name in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
ENVIRONMENT.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='',
    PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
SOURCE_FILES = ('experiments/kernel_svm_runtime_gate.py',
    'experiments/test_kernel_svm_source_contract.py',
    'experiments/KERNEL_SVM_RUNTIME_PLAN.json',
    'experiments/NEXT_TEMPORAL_LEARNER_RESEARCH_20261031.md')


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
    spec = importlib.util.spec_from_file_location('r72_literal_schema', SCHEMA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.schema_contract()


def chronological_splits(base_frames):
    """Three array_split-equivalent chunks, actual-frame purge, paired roles.

    Inputs are invented original centers; output indices address original rows
    followed by their mirrored rows. This pure seam is also reusable later.
    """
    frames = list(base_frames)
    if len(frames) < 3 or any(type(f) is not int for f in frames):
        raise ValueError('At least three integer original frame centers are required')
    if any(a >= b for a, b in zip(frames, frames[1:])):
        raise ValueError('Original frame centers must be strictly ascending and unique')
    count, start, result = len(frames), 0, []
    quotient, remainder = divmod(count, 3)
    for group in range(3):
        stop = start + quotient + (group < remainder)
        validation = list(range(start, stop))
        low, high = frames[start], frames[stop - 1]
        train = [i for i, frame in enumerate(frames)
                 if frame < low - PURGE_FRAMES or frame > high + PURGE_FRAMES]
        if not train or not validation:
            raise ValueError('A chronological inner fold is empty after the fixed frame purge')
        if any(abs(frames[t] - frames[v]) <= PURGE_FRAMES for t in train for v in validation):
            raise AssertionError('Actual-frame inner purge failed')
        result.append(dict(group=group, base_train=train, base_validation=validation,
            train=train + [i + count for i in train],
            validation=validation + [i + count for i in validation],
            validation_min_frame=low, validation_max_frame=high,
            purge_frames=PURGE_FRAMES))
        start = stop
    covered = sorted(i for fold in result for i in fold['validation'])
    if covered != list(range(2 * count)):
        raise AssertionError('Inner validation sets do not partition augmented rows exactly once')
    return result


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
    # whatever bytes happen to be present when R72 preparation starts.
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
    files, links = runtime_snapshot()
    write_json(ROUND / 'runtime_manifest.json', files)
    write_json(ROUND / 'runtime_symlinks.json', links)
    versions = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in metadata.distributions(path=[str(PACKAGES)])}
    write_json(ROUND / 'source_metadata.json', dict(author_revision=AUTHOR_REVISION,
        author_source_sha256=sources, runtime_files=len(files), runtime_symlinks=len(links),
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
    print('R72 source metadata prepared; no scientific imports, model calls or freeze', flush=True)


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
    schema = load_schema()
    if schema != acquired['schema']:
        raise ValueError('The exact 166-column source/swap contract changed')
    required = dict(cpu_threads=1, workers=1, fit_seconds=FIT_SECONDS,
        prediction_seconds=PREDICT_SECONDS, worker_seconds=WORKER_SECONDS, rss_bytes=LIMIT_BYTES)
    if plan['resources'] != required or plan['synthetic_shapes'] != {
            k: [n, 166] for k, n in CONTEXT_ROWS.items()}:
        raise ValueError('Prospective resources or invented shapes differ')
    return dict(author_revision=AUTHOR_REVISION, author_sources=AUTHOR_SOURCES,
        schema=schema, versions=acquired['versions'], resources=required)


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
    print('R72 source-only preflight passed; no model work started', flush=True)


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
    print('R72 reviewed source frozen; separate model resource admission remains required', flush=True)


def verify():
    folder = ROUND / 'protocol_freeze'
    proof = json.loads((folder / 'freeze.json').read_text())
    if PLAN.read_bytes() != (folder / 'plan.json').read_bytes():
        raise ValueError('R72 frozen plan changed')
    if digest(folder / 'implementation.json') != proof['implementation_manifest_sha256']:
        raise ValueError('R72 frozen manifest changed')
    for relative, expected in json.loads((folder / 'implementation.json').read_text()).items():
        if digest(ROOT / relative) != expected:
            raise ValueError('R72 frozen input changed: ' + relative)
    source_contract()
    verify_runtime()


def require_resource_admission():
    approval = json.loads((ROUND / 'resource_admission.json').read_text())
    if (approval.get('approved_for_model_calls') is not True or
            approval.get('scope') != 'invented_only_r72' or
            approval.get('predecessor_r65_closed_or_passed') is not True or
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
        raise ValueError('R72 is a CPU scikit-learn gate, without Torch')
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


def synthetic_data(size):
    import numpy as np
    schema = load_schema()
    rows = CONTEXT_ROWS[size]
    generator = np.random.default_rng(720042)
    base = generator.normal(size=(rows // 2, 166)).astype(np.float64)
    query = generator.normal(size=(QUERY_ROWS, 166)).astype(np.float64)
    permutation = np.asarray(schema['permutation'], dtype=np.int64)
    if schema['signs'] != [1] * 166:
        raise ValueError('Actual project team swap is not the expected pure permutation')
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
    base_frames = [72_000_000 + 5 * i for i in range(rows // 2)]
    folds = chronological_splits(base_frames)
    data = dict(raw_context=context, target=target, raw_query=np.ascontiguousarray(query),
        permutation=permutation, base_frames=np.asarray(base_frames, dtype=np.int64))
    if np.bincount(target).tolist() != [rows // 2, rows // 2]:
        raise ValueError('Invented paired context is not class balanced')
    for values in data.values():
        values.flags.writeable = False
    return data, folds


def make_estimator():
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.svm import SVC
    return Pipeline([
        ('nonfinite', FunctionTransformer(np.nan_to_num, validate=False,
            kw_args=dict(copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan))),
        ('imputer', SimpleImputer(strategy='median', add_indicator=False, keep_empty_features=True)),
        ('scaler', StandardScaler()),
        # Omitted deprecated probability argument has effective False in 1.9.1.
        ('svc', SVC(C=1., kernel='rbf', gamma='scale', shrinking=True, tol=1e-3,
            cache_size=200, class_weight=None, verbose=False, max_iter=-1,
            decision_function_shape='ovr', break_ties=False, random_state=42)),
    ])


def explicit_pipeline_state(pipeline):
    import numpy as np
    imputer, scaler, svc = [pipeline.named_steps[name] for name in ('imputer', 'scaler', 'svc')]
    if (svc.fit_status_ != 0 or svc._effective_probability is not False or
            not np.array_equal(svc.classes_, [0, 1]) or len(imputer.statistics_) != 166):
        raise ValueError('SVC fit status, internal probability, classes or fixed columns differ')
    if not all(np.isfinite(getattr(svc, name)).all() for name in
            ('support_vectors_', 'dual_coef_', 'intercept_')) or not math.isfinite(svc._gamma) or svc._gamma <= 0:
        raise ValueError('Nonfinite SVC learned values or invalid resolved gamma')
    if not all(np.isfinite(value).all() for value in
            (imputer.statistics_, scaler.mean_, scaler.var_, scaler.scale_)):
        raise ValueError('Nonfinite fitted preprocessing statistics')
    return dict(imputer_statistics=array_record(imputer.statistics_),
        scaler={name: array_record(getattr(scaler, name)) for name in
            ('mean_', 'var_', 'scale_', 'n_samples_seen_')},
        svc={name: array_record(getattr(svc, name)) for name in
            ('support_', 'support_vectors_', 'dual_coef_', 'intercept_', 'classes_', 'n_support_', 'n_iter_')},
        gamma_bits=struct.pack('>d', svc._gamma).hex(), fit_status=int(svc.fit_status_),
        effective_probability=svc._effective_probability, full_dict=semantic_state(pipeline))


def fit_with_author_recording(data, folds, output):
    """Delegate the exact author CV fit; observe actual fit/decision calls only."""
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    context, target = data['raw_context'], data['target']
    expected = [(fold['train'], fold['validation']) for fold in folds]
    expected.append((list(range(len(context))), None))
    original_fit, original_decision = Pipeline.fit, Pipeline.decision_function
    fit_records, margins, identities = [], {}, {}

    @functools.wraps(original_fit)
    def observed_fit(estimator, X, y=None, **params):
        ordinal = len(fit_records)
        if ordinal >= 4:
            raise ValueError('Unexpected extra classifier fit')
        train, validation = expected[ordinal]
        if array_record(X) != array_record(context[train]) or array_record(y) != array_record(target[train]):
            raise ValueError('Author inner/final fit did not receive its exact permitted raw rows')
        before = dict(X=array_record(X), y=array_record(y))
        result = original_fit(estimator, X, y, **params)
        if before != dict(X=array_record(X), y=array_record(y)):
            raise ValueError('Pipeline fit mutated its raw input or target')
        state = explicit_pipeline_state(estimator)
        identity = dict(ordinal=ordinal, role='final' if validation is None else 'inner',
            train_indices=train, validation_indices=validation, input_sha256=before,
            learned_state=state, semantic_sha256=state_digest(estimator))
        # Recorder data are external; no output paths or audit attributes are
        # added to the standard author Pipeline or its fitted children.
        identities[id(estimator)] = ordinal
        fit_records.append(identity)
        write_json(output / ('fit_' + str(ordinal) + '_state.json'), identity)
        with (output / ('fit_' + str(ordinal) + '.pkl')).open('xb') as handle:
            pickle.dump(estimator, handle, protocol=5)
        return result

    @functools.wraps(original_decision)
    def observed_decision(estimator, X, **params):
        ordinal = identities.get(id(estimator))
        if ordinal not in (0, 1, 2) or ordinal in margins:
            raise ValueError('Unexpected author out-of-fold margin call')
        validation = expected[ordinal][1]
        if array_record(X) != array_record(context[validation]):
            raise ValueError('Author calibration margin input differs from the held-out inner rows')
        before = state_digest(estimator)
        value = original_decision(estimator, X, **params)
        if before != state_digest(estimator) or value.shape != (len(validation),) or not np.isfinite(value).all():
            raise ValueError('Invalid or state-mutating author calibration margin call')
        margins[ordinal] = np.asarray(value).copy()
        np.save(output / ('inner_' + str(ordinal) + '_margin.npy'), value, allow_pickle=False)
        return value

    model = CalibratedClassifierCV(estimator=make_estimator(), method='sigmoid',
        cv=[(np.asarray(train), np.asarray(validation)) for train, validation in expected[:3]],
        ensemble=False, n_jobs=1)
    # wraps preserves __name__='decision_function': the author resolves this
    # string and passes it to cross_val_predict/getattr.
    with patch.object(Pipeline, 'fit', observed_fit), patch.object(Pipeline, 'decision_function', observed_decision):
        model.fit(context, target)
    if len(fit_records) != 4 or sorted(margins) != [0, 1, 2] or len(model.calibrated_classifiers_) != 1:
        raise ValueError('The complete author three-inner plus one-final fit did not occur')
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError('Calibrated H0/A1 class order differs')
    calibrated = model.calibrated_classifiers_[0]
    if len(calibrated.calibrators) != 1 or state_digest(calibrated.estimator) != fit_records[3]['semantic_sha256']:
        raise ValueError('The fitted final pipeline differs from the recorded fourth fit')
    oof = np.empty(len(context), dtype=np.float64)
    for ordinal, fold in enumerate(folds):
        oof[fold['validation']] = margins[ordinal]
    np.save(output / 'oof_margin.npy', oof, allow_pickle=False)
    write_json(output / 'fit_records.json', fit_records)
    write_json(output / 'calibrator_state.json', semantic_state(calibrated.calibrators[0]))
    write_json(output / 'full_state.json', semantic_state(model))
    return model


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


def operation(output, name, limit, callback):
    start = time.monotonic()
    write_json(output / 'operation.json', dict(name=name, active=True, started=start, limit_seconds=limit))
    result = callback()
    elapsed = time.monotonic() - start
    write_json(output / 'operation.json', dict(name=name, active=False, started=start,
        seconds=elapsed, limit_seconds=limit))
    if elapsed >= limit or peak_bytes() > LIMIT_BYTES:
        raise RuntimeError('Operation exceeded its prospective time or memory bound: ' + name)
    return result, elapsed


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
    # Independent fresh calls after the full schedule, in swapped-first order.
    first_swap = model.predict_proba(np.ascontiguousarray(query[:QUERY_SIZE, permutation]))
    first_original = model.predict_proba(np.ascontiguousarray(query[:QUERY_SIZE]))
    if not np.array_equal(first_swap, swapped[:QUERY_SIZE]) or not np.array_equal(first_original, original[:QUERY_SIZE]):
        raise ValueError('Repeated first canonical queries differ')
    independent_swapped_owner = (first_swap + first_original[:, ::-1]) / 2
    if not np.array_equal(independent_swapped_owner, owner[:QUERY_SIZE, ::-1]):
        raise ValueError('Independent team-swap symmetrization differs')
    prefix = model.predict_proba(np.ascontiguousarray(query[:8]))
    reverse = model.predict_proba(np.ascontiguousarray(query[:QUERY_SIZE][::-1]))[::-1]
    diagnostics = dict(prefix_max_abs=float(np.max(np.abs(prefix - original[:8]))),
        reverse_max_abs=float(np.max(np.abs(reverse - original[:QUERY_SIZE]))),
        selection_allowed=False)
    if state_digest(model) != before:
        raise ValueError('Prediction mutated complete calibrated estimator state')
    with (output / 'calibrated_model.pkl').open('xb') as handle:
        pickle.dump(model, handle, protocol=5)
    with (output / 'calibrated_model.pkl').open('rb') as handle:
        reloaded = pickle.load(handle)
    if state_digest(reloaded) != before:
        raise ValueError('Complete calibrated estimator semantic state differs on reload')
    reload_original = canonical_predict(reloaded, query)
    reload_swap = canonical_predict(reloaded, query[:, permutation])
    if not np.array_equal(reload_original, original) or not np.array_equal(reload_swap, swapped):
        raise ValueError('Canonical raw probabilities differ after reload')
    if state_digest(reloaded) != before:
        raise ValueError('Reloaded estimator state changed during prediction')
    # Each of the three actual inner estimators and the fourth final estimator
    # must also deserialize with every learned __dict__ field intact.
    records = json.loads((output / 'fit_records.json').read_text())
    for ordinal in range(4):
        with (output / ('fit_' + str(ordinal) + '.pkl')).open('rb') as handle:
            inner = pickle.load(handle)
        if state_digest(inner) != records[ordinal]['semantic_sha256']:
            raise ValueError('An actual inner/final fitted Pipeline differs on reload')
    return diagnostics


def worker(size, stage, output):
    verify()
    require_resource_admission()
    output.mkdir(parents=True, exist_ok=False)
    blocked = RuntimeError('Network disabled for the invented-only R72 worker')
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=blocked))
        stack.enter_context(patch.object(socket.socket, 'connect_ex', side_effect=blocked))
        stack.enter_context(patch.object(urllib.request, 'urlopen', side_effect=blocked))
        np, limits, thread_info = initialize()
        stack.enter_context(limits(limits=1))
        pools = thread_info()
        if any(pool['num_threads'] != 1 for pool in pools):
            raise ValueError('A bound numerical thread pool is not single threaded')
        write_json(output / 'threadpools.json', pools)
        data, folds = synthetic_data(size)
        arrays_before = {name: array_record(value) for name, value in data.items()}
        for name, value in data.items():
            np.save(output / (name + '.npy'), value, allow_pickle=False)
        write_json(output / 'inner_folds.json', folds)
        write_json(output / 'recipe.json', dict(shape=[CONTEXT_ROWS[size], 166], query_rows=QUERY_ROWS,
            query_batch=QUERY_SIZE, seed=42, generator_seed=720042, inner_groups=3,
            purge_frames=PURGE_FRAMES, frame_spacing=5, class_order=['H', 'A'],
            author_revision=AUTHOR_REVISION, ensemble=False, method='sigmoid',
            preprocessing='Nonfinite-to-NaN copy; inner-train-only median/empty0, standardscale/constant1',
            source_contract=source_contract(), arrays=arrays_before))
        model, fit_seconds = operation(output, 'complete_three_inner_and_final_fit', FIT_SECONDS,
            lambda: fit_with_author_recording(data, folds, output))
        final_before = state_digest(model)
        diagnostics, query_seconds = operation(output, 'all_queries_repeat_symmetry_reload', PREDICT_SECONDS,
            lambda: query_and_reload_checks(model, data, output))
        if final_before != state_digest(model) or arrays_before != {name: array_record(value) for name, value in data.items()}:
            raise ValueError('Complete fitted state or invented raw arrays changed')
        exact_files = ['recipe.json', 'inner_folds.json', 'fit_records.json', 'full_state.json',
            'calibrator_state.json'] + ['fit_' + str(i) + '_state.json' for i in range(4)] + [
            'inner_' + str(i) + '_margin.npy' for i in range(3)] + [
            name + '.npy' for name in ('oof_margin', 'original_probability', 'swapped_probability', 'owner')]
        if stage == 'rebuild':
            first = output.parent / 'first'
            for name in exact_files:
                if (output / name).read_bytes() != (first / name).read_bytes():
                    raise ValueError('Independent fresh-process complete refit differs: ' + name)
        checks = dict(four_actual_author_fits=True, three_purged_inner_margin_calls=True,
            single_final_classifier=True, finite_normalized_probabilities=True,
            exact_first_repeat=True, exact_team_swap=True, raw_arrays_immutable=True,
            full_fitted_state_immutable=True, semantic_reload_exact=True,
            canonical_reload_exact=True, independent_refit_exact=stage == 'rebuild')
        write_json(output / 'summary.json', dict(passed=True, size=size, stage=stage,
            synthetic_only=True, native_arrays_read=False, checks=checks,
            fit_seconds=fit_seconds, all_query_seconds=query_seconds,
            diagnostics=diagnostics, peak_bytes=peak_bytes(), exact_artifacts=exact_files,
            artifact_sha256={p.name: digest(p) for p in output.iterdir()
                if p.is_file() and p.name not in ('operation.json', 'summary.json')}))
    verify()


def supervise(size, stage, output):
    started, peak, aborted = time.monotonic(), 0, None
    destination = output / size / stage
    destination.parent.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = [destination.parent / (stage + suffix) for suffix in ('.stdout.log', '.stderr.log')]
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        process = subprocess.Popen([str(PYTHON), '-B', '-P', '-S', str(Path(__file__).resolve()),
            '--worker', '--size', size, '--stage', stage, '--output', str(destination)],
            cwd=ROOT, env=worker_environment(os.environ), stdout=stdout, stderr=stderr)
        while process.poll() is None:
            sample = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True)
            if sample.returncode == 0 and sample.stdout.strip():
                peak = max(peak, int(sample.stdout.strip()) * 1024)
            status_path = destination / 'operation.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if peak > LIMIT_BYTES:
                aborted = 'rss_over_6GiB'
            elif time.monotonic() - started >= WORKER_SECONDS:
                aborted = 'worker_over_900seconds'
            elif status.get('active') and time.monotonic() - status['started'] >= status['limit_seconds']:
                aborted = status['name'] + '_over_' + str(status['limit_seconds']) + 'seconds'
            if aborted:
                process.terminate()
                break
            time.sleep(.1)
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=10)
    summary_path = destination / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    elapsed = time.monotonic() - started
    if aborted is None and elapsed >= WORKER_SECONDS:
        aborted = 'worker_over_900seconds'
    if aborted is None and max(peak, summary.get('peak_bytes', 0)) > LIMIT_BYTES:
        aborted = 'rss_over_6GiB'
    record = dict(size=size, stage=stage, exit_code=code, aborted=aborted,
        passed=aborted is None and code == 0 and summary.get('passed', False),
        largest_sampled_rss_bytes=peak, wall_seconds=elapsed,
        summary_sha256=digest(summary_path) if summary_path.exists() else None,
        stdout_sha256=digest(stdout_path), stderr_sha256=digest(stderr_path))
    write_json(destination.parent / (stage + '.supervision.json'), record)
    print(size, stage, 'passed', record['passed'], flush=True)
    return record


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
        parser.error('All R72 assessment outputs must be a new child of its own directory')
    if args.prepare_metadata:
        prepare_metadata()
    elif args.preflight:
        preflight()
    elif args.freeze:
        freeze()
    elif args.verify:
        verify()
        assert_source_only()
        print('R72 frozen sources verified without scientific imports', flush=True)
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
