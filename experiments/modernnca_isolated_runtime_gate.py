"""R64 invented-only ModernNCA gate; root review/freeze precedes model calls.

Source acquisition, preflight, freeze and supervision use the standard library.
Only an explicitly dispatched, frozen worker imports NumPy/Torch. No native
input path, public target, possession scorer or pretrained weights are used.
"""
import argparse
import ast
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import socket
import subprocess
import sys
import time
import traceback
import types
from typing import Optional
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'data/control_modernnca_isolated_runtime_v1'
PLAN = ROOT / 'experiments/MODERNNCA_ISOLATED_RUNTIME_PLAN.json'
PYTHON = ROOT / 'data/control_isolated_python_v1/python/bin/python3.12'
DEPS = ROOT / 'data/control_tabpfn_mps_loader_repair_v1/runtime_deps'
AUTHOR = ROUND / 'public_sources'
CONTEXT_ROWS = {'moderate': 64, 'full': 3000}
STAGES = (('moderate', 'first'), ('moderate', 'rebuild'), ('full', 'first'), ('full', 'rebuild'))
EPOCHS = 200
BATCH_SIZE = 1024
QUERY_SIZE = 512
CPU_THREADS = 2
FIT_SECONDS = 1800
PREDICT_SECONDS = 300
WORKER_SECONDS = 2400
LIMIT_BYTES = 6 * 1024**3
ENVIRONMENT = {name: '2' for name in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
ENVIRONMENT.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='',
    PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
    HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled',
    PYTORCH_ENABLE_MPS_FALLBACK='0', TOKENIZERS_PARALLELISM='false')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for body in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(body)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def acquisition_module():
    return load_local('r64_acquisition_constants', ROOT / 'experiments/acquire_modernnca_sources.py')


def schema_module():
    return load_local('r64_schema_contract', ROOT / 'experiments/modernnca_schema_contract.py')


def definition_node(path, name, parent=None):
    nodes = ast.parse(Path(path).read_bytes()).body
    if parent:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent).body
    return next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def definition_record(path, name, parent=None):
    node = definition_node(path, name, parent)
    canonical = ast.dump(node, include_attributes=False)
    return dict(path=str(Path(path).relative_to(ROOT)), name=name, parent=parent,
        ast_sha256=hashlib.sha256(canonical.encode()).hexdigest())


def candidate_indices_literal(train_size, batch_ids):
    """Stdlib sentinel only; real fitting executes the unchanged author method."""
    if train_size <= 0 or len(set(batch_ids)) != len(batch_ids):
        raise ValueError('Invalid candidate fixture')
    if any(not isinstance(i, int) or i < 0 or i >= train_size for i in batch_ids):
        raise ValueError('Batch index outside training fixture')
    batch = set(batch_ids)
    return [i for i in range(train_size) if i not in batch]


def source_contract():
    acquired = acquisition_module()
    acquisition = json.loads((ROUND / 'acquisition.json').read_text())
    if acquisition['revision'] != acquired.REVISION or not acquisition['source_text_only']:
        raise ValueError('Unexpected source acquisition provenance')
    if digest(ROUND / 'acquisition.json') != json.loads(PLAN.read_text())['acquisition_sha256']:
        raise ValueError('Plan does not bind source acquisition provenance')
    for name, (count, expected) in acquired.SOURCES.items():
        source = AUTHOR / name
        if source.stat().st_size != count or digest(source) != expected:
            raise ValueError('Author source identity differs: ' + name)
    config = json.loads((AUTHOR / 'TALENT/configs/default/modernNCA.json').read_text())['modernNCA']
    general = json.loads((AUTHOR / 'TALENT/configs/deep_configs.json').read_text())
    expected_model = dict(dim=128, dropout=.1, d_block=512, n_blocks=0, temperature=1,
        num_embeddings=dict(type='PLREmbeddings', n_frequencies=77,
            frequency_scale=.04431360576139521, d_embedding=34, lite=True), sample_rate=.5)
    if config['model'] != expected_model or config['training'] != dict(lr=.01, weight_decay=.0002):
        raise ValueError('Author defaults changed')
    if general['max_epoch'] != EPOCHS or general['batch_size'] != BATCH_SIZE:
        raise ValueError('Author epoch/batch defaults differ')
    model = AUTHOR / 'TALENT/model/models/modernNCA.py'
    helper = AUTHOR / 'TALENT/model/lib/tabr/utils.py'
    methods = AUTHOR / 'TALENT/model/methods/modernNCA.py'
    utils = AUTHOR / 'TALENT/model/utils.py'
    definitions = {
        'ModernNCA': definition_record(model, 'ModernNCA'),
        'make_module': definition_record(helper, 'make_module'),
        'PLREmbeddings': definition_record(helper, 'PLREmbeddings'),
        'make_random_batches': definition_record(methods, 'make_random_batches'),
        'train_epoch': definition_record(methods, 'train_epoch', 'ModernNCAMethod'),
        'Averager': definition_record(utils, 'Averager'),
    }
    return dict(definitions=definitions, model_config=config['model'],
        training_config=config['training'], general_config=general,
        schema=schema_module().schema_contract(), source_count=len(acquired.SOURCES))


def runtime_references():
    """Reuse exact old wheel bytes, applying only the prior recorded loader copies."""
    plan = json.loads(PLAN.read_text())
    refs = plan['runtime_reference_sha256']
    for name, expected in refs.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Prior runtime reference changed: ' + name)
    original = json.loads((ROOT / plan['original_runtime_manifest']).read_text())
    repairs = json.loads((ROOT / plan['repair_manifest']).read_text())['records']
    replacement = {r['path']: r['new_sha256'] for r in repairs}
    old_prefix = 'data/control_tabpfn_mps_v1/runtime_deps/'
    files = {}
    for name, expected in original.items():
        if name.startswith(old_prefix):
            relative = name[len(old_prefix):]
            files[str((DEPS / relative).relative_to(ROOT))] = replacement.get(relative, expected)
    if len(files) < 20_000:
        raise ValueError('Incomplete inherited dependency file binding')
    links = {name: target for name, target in json.loads(
        (ROOT / plan['repair_symlinks']).read_text()).items()
        if name.startswith(str(DEPS.relative_to(ROOT)) + '/')}
    files.update(refs)
    files[str(PYTHON.resolve())] = plan['interpreter_sha256']
    isolated = ROOT / 'data/control_isolated_python_v1'
    identity = isolated / 'acquisition.json'
    if digest(identity) != plan['isolated_runtime_acquisition_sha256']:
        raise ValueError('Isolated Python acquisition changed')
    retained = json.loads(identity.read_text())
    for relative, expected in retained['files_sha256'].items():
        files[str((isolated / relative).relative_to(ROOT))] = expected
    for relative, target in retained['symlinks'].items():
        links[str((isolated / relative).relative_to(ROOT))] = target
    for environment in ('head_runtime', 'torch25_runtime'):
        relative = isolated / environment / 'pyvenv.cfg'
        files[str(relative.relative_to(ROOT))] = plan['isolated_venv_configuration_sha256'][environment]
    files[str(identity.relative_to(ROOT))] = plan['isolated_runtime_acquisition_sha256']
    return files, links


def verify_mapping(files, links):
    for name, expected in files.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Read-only source/runtime changed: ' + name)
    for name, target in links.items():
        path = ROOT / name
        if not path.is_symlink() or os.readlink(path) != target:
            raise ValueError('Read-only dependency link changed: ' + name)


def assert_no_scientific_imports():
    if any(name in sys.modules for name in ('torch', 'numpy', 'pandas', 'scipy', 'sklearn', 'TALENT')):
        raise ValueError('Scientific import occurred in a source-only action')


def preflight():
    started = time.monotonic()
    contract = source_contract()
    files, links = runtime_references()
    verify_mapping(files, links)
    for name in ('modernnca_isolated_runtime_gate.py', 'acquire_modernnca_sources.py',
                 'modernnca_schema_contract.py', 'test_modernnca_isolated_runtime.py'):
        path = ROOT / 'experiments' / name
        compile(path.read_bytes(), str(path), 'exec')
    versions = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in metadata.distributions(path=[str(DEPS)])}
    if versions.get('torch') != '2.15.0' or versions.get('numpy') != '2.1.2':
        raise ValueError('Isolated runtime metadata differs')
    assert_no_scientific_imports()
    write_json(ROUND / 'source_preflight.json', dict(passed=True,
        seconds=time.monotonic() - started, scientific_imports=False, model_calls=False,
        native_arrays_read=False, plan_sha256=digest(PLAN),
        implementation_sha256=digest(__file__), source_contract=contract,
        runtime_bound_files=len(files), runtime_bound_symlinks=len(links)))
    print('R64 source-only preflight passed; no scientific imports or model calls', flush=True)


def freeze():
    review = json.loads((ROUND / 'independent_review.json').read_text())
    sources = [Path(__file__), PLAN, ROOT / 'experiments/acquire_modernnca_sources.py',
        ROOT / 'experiments/modernnca_schema_contract.py', ROOT / 'experiments/test_modernnca_isolated_runtime.py']
    if not review['approved_for_freeze']:
        raise ValueError('Root independent review required')
    for path in sources:
        if review['source_sha256'][str(path.relative_to(ROOT))] != digest(path):
            raise ValueError('Review does not bind final source: ' + str(path))
    proof = json.loads((ROUND / 'source_preflight.json').read_text())
    if not proof['passed'] or proof['implementation_sha256'] != digest(__file__) or proof['plan_sha256'] != digest(PLAN):
        raise ValueError('Final source-only preflight is required')
    contract = source_contract()
    files, links = runtime_references()
    files.update({str(p.relative_to(ROOT)): digest(p) for p in sources})
    for path in sorted(ROUND.rglob('*')):
        if path.is_file() and 'protocol_freeze' not in path.parts and 'runtime_assessment' not in path.parts:
            if path.suffix == '.pyc' or '__pycache__' in path.parts:
                raise ValueError('Unexpected bytecode in new round')
            files[str(path.relative_to(ROOT))] = digest(path)
    files.update(contract['schema']['source_sha256'])
    verify_mapping(files, links)
    folder = ROUND / 'protocol_freeze'
    folder.mkdir(exist_ok=False)
    (folder / 'plan.json').write_bytes(PLAN.read_bytes())
    write_json(folder / 'implementation.json', files)
    write_json(folder / 'symlinks.json', links)
    write_json(folder / 'freeze.json', dict(plan_sha256=digest(PLAN),
        implementation_sha256=digest(__file__), source_contract=contract,
        scientific_imports=False, model_calls=False, native_arrays_read=False,
        independent_review_sha256=digest(ROUND / 'independent_review.json')))
    assert_no_scientific_imports()
    print('R64 frozen by reviewed source action; no model work started', flush=True)


def verify():
    folder = ROUND / 'protocol_freeze'
    if PLAN.read_bytes() != (folder / 'plan.json').read_bytes():
        raise ValueError('R64 frozen plan changed')
    verify_mapping(json.loads((folder / 'implementation.json').read_text()),
                   json.loads((folder / 'symlinks.json').read_text()))
    source_contract()


def runtime_paths():
    base = Path(sys.base_prefix) / 'lib'
    sys.path[:] = [str(DEPS), str(base / 'python312.zip'), str(base / 'python3.12'),
                   str(base / 'python3.12/lib-dynload')]
    sys.dont_write_bytecode = True


def initialize():
    for name, value in ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise ValueError('Missing fixed worker environment: ' + name)
    runtime_paths()
    import numpy as np
    import torch
    if torch.__version__ != '2.15.0' or np.__version__ != '2.1.2':
        raise ValueError('Pinned runtime version differs')
    for package, module in (('torch', torch), ('numpy', np)):
        if not Path(module.__file__).resolve().is_relative_to((DEPS / package).resolve()):
            raise ValueError('Scientific package escaped isolated dependency tree')
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    random.seed(42)
    np.random.seed(43)
    torch.manual_seed(44)
    return np, torch


def extracted_definitions(torch):
    """Compile exact author AST definitions without importing the broad toolkit."""
    namespace = dict(torch=torch, Optional=Optional, __name__='r64_exact_author_training')
    nodes = [definition_node(AUTHOR / 'TALENT/model/utils.py', 'Averager'),
        definition_node(AUTHOR / 'TALENT/model/methods/modernNCA.py', 'make_random_batches'),
        definition_node(AUTHOR / 'TALENT/model/methods/modernNCA.py', 'train_epoch', 'ModernNCAMethod')]
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<exact-pinned-author-training>', 'exec'), namespace)
    return namespace


def official_model_class():
    """Empty package namespaces resolve one unmodified author helper import only."""
    for name in ('TALENT', 'TALENT.model', 'TALENT.model.lib', 'TALENT.model.lib.tabr', 'TALENT.model.models'):
        if name in sys.modules:
            raise ValueError('Unexpected pre-existing TALENT namespace')
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    helper_name = 'TALENT.model.lib.tabr.utils'
    helper = load_local(helper_name, AUTHOR / 'TALENT/model/lib/tabr/utils.py')
    sys.modules[helper_name] = helper
    model_name = 'TALENT.model.models.modernNCA'
    module = load_local(model_name, AUTHOR / 'TALENT/model/models/modernNCA.py')
    sys.modules[model_name] = module
    return module.ModernNCA


def swap_array(values, permutation, signs):
    return values[:, permutation] * signs


def symmetric_statistics(context, permutation, signs):
    """Project choice: orbit-tied mean/scale, fitted only to paired training rows."""
    import numpy as np
    mean, scale = np.zeros(context.shape[1]), np.ones(context.shape[1])
    visited = set()
    for i, j in enumerate(permutation):
        if i in visited:
            continue
        values = context[:, i]
        finite = values[np.isfinite(values)]
        center = float(finite.mean()) if len(finite) else 0.
        if i == j and signs[i] == -1:
            center = 0.
        filled = np.where(np.isfinite(values), values, center)
        deviation = float(np.sqrt(np.mean((filled - center)**2)))
        mean[i], mean[j] = center, signs[i] * center
        scale[i] = scale[j] = deviation if deviation > 0 else 1.
        visited.update((i, int(j)))
    return mean, scale


def preprocess(values, mean, scale):
    import numpy as np
    return np.ascontiguousarray((np.where(np.isfinite(values), values, mean) - mean) / scale,
                               dtype=np.float64)


def synthetic_data(size):
    import numpy as np
    schema = schema_module().schema_contract()
    permutation, signs = np.asarray(schema['permutation']), np.asarray(schema['signs'], dtype=np.float64)
    rows, queries = CONTEXT_ROWS[size], 513 if size == 'moderate' else 512
    generator = np.random.default_rng(640042)
    base = generator.normal(size=(rows // 2, 166)).astype(np.float64)
    query = generator.normal(size=(queries, 166)).astype(np.float64)
    h = schema['columns'].index('h__distance_1')
    a = schema['columns'].index('a__distance_1')
    target = (base[:, h] - base[:, a] + .25 * base[:, 0] > 0).astype(np.int64)
    base.flat[::37], query.flat[::41] = np.nan, np.nan
    empty = schema['columns'].index('h__set_latent_31')
    constant = schema['columns'].index('h__set_latent_30')
    for value in (base, query):
        value[:, [empty, permutation[empty]]] = np.nan
        value[:, [constant, permutation[constant]]] = 1.
    raw_context = np.concatenate([base, swap_array(base, permutation, signs)])
    target = np.r_[target, 1 - target]
    mean, scale = symmetric_statistics(raw_context, permutation, signs)
    context = preprocess(raw_context, mean, scale)
    original = preprocess(query, mean, scale)
    swapped = preprocess(swap_array(query, permutation, signs), mean, scale)
    np.testing.assert_array_equal(swapped, swap_array(original, permutation, signs))
    np.testing.assert_array_equal(context[rows // 2:], swap_array(context[:rows // 2], permutation, signs))
    np.testing.assert_array_equal(np.bincount(target), [rows // 2, rows // 2])
    if context.shape != (rows, 166) or not np.isfinite(context).all() or not np.isfinite(original).all():
        raise ValueError('Invented feature/preprocessing schema differs')
    return dict(raw_context=raw_context, context=context, target=target, raw_query=query,
        original=original, swapped=swapped, mean=mean, scale=scale,
        permutation=permutation, signs=signs)


def array_record(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def state_record(model):
    return {name: array_record(value) for name, value in sorted(model.state_dict().items())}


def probability_checks(value, rows):
    import numpy as np
    return bool(value.shape == (rows, 2) and np.isfinite(value).all() and (value >= 0).all()
                and np.allclose(value.sum(axis=1), 1., atol=1e-12, rtol=0))


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def operation(output, name, kind, callback):
    limit = FIT_SECONDS if kind == 'fit' else PREDICT_SECONDS
    start = time.monotonic()
    write_json(output / 'operation.json', dict(name=name, kind=kind, active=True,
        started=start, limit_seconds=limit))
    result = callback()
    seconds = time.monotonic() - start
    write_json(output / 'operation.json', dict(name=name, kind=kind, active=False,
        started=start, seconds=seconds, limit_seconds=limit))
    if seconds > limit or peak_bytes() > LIMIT_BYTES:
        raise RuntimeError('Operation exceeded prospective resource bound: ' + name)
    return result, seconds


def train_complete(model, torch, context, target, contract, output):
    import numpy as np
    losses = []
    def criterion(prediction, labels):
        loss = torch.nn.functional.nll_loss(prediction, labels)
        scalar = loss.item()
        if not math.isfinite(scalar):
            raise ValueError('Nonfinite training loss')
        losses.append(scalar)
        return loss
    definition = extracted_definitions(torch)
    trainer = types.SimpleNamespace(model=model, N={'train': context}, C=None, y={'train': target},
        train_size=len(context), train_indices=torch.arange(len(context), device='cpu'),
        args=types.SimpleNamespace(device=torch.device('cpu'), batch_size=BATCH_SIZE,
            use_float=False, config={'training': contract['training_config']}),
        train_step=0, trlog={'train_loss': []}, train_loader=range(math.ceil(len(context) / BATCH_SIZE)),
        criterion=criterion, optimizer=torch.optim.AdamW(model.parameters(), **contract['training_config']))
    write_json(output / 'resolved_optimizer.json', trainer.optimizer.defaults)
    with (output / 'training.log').open('w') as log, redirect_stdout(log):
        for epoch in range(EPOCHS):
            definition['train_epoch'](trainer, epoch)
            log.flush()
    wanted = EPOCHS * math.ceil(len(context) / BATCH_SIZE)
    if trainer.train_step != wanted or len(losses) != wanted or len(trainer.trlog['train_loss']) != EPOCHS:
        raise ValueError('Incomplete final-200-epoch training')
    return np.asarray(losses, dtype=np.float64), np.asarray(trainer.trlog['train_loss'], dtype=np.float64)


def raw_predict(model, torch, context, target, query):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(query, dtype=torch.float64, device='cpu')
        log_prob = model(x=tensor, y=None, candidate_x=context, candidate_y=target, is_train=False)
        result = torch.softmax(log_prob, dim=1).detach().cpu().numpy()
    if not probability_checks(result, len(query)):
        raise ValueError('Invalid binary probability vector')
    return result


def canonical_predict(model, torch, context, target, query, output, name):
    import numpy as np
    values, timings = [], []
    for start in range(0, len(query), QUERY_SIZE):
        batch = query[start:start + QUERY_SIZE]
        count = len(batch)
        if count < QUERY_SIZE:
            batch = np.concatenate([batch, np.repeat(batch[-1:], QUERY_SIZE - count, axis=0)])
        value, seconds = operation(output, f'{name}_{start}', 'predict',
            lambda batch=batch: raw_predict(model, torch, context, target, batch))
        values.append(value[:count])
        timings.append(dict(name=name, start=start, real_rows=count, call_rows=QUERY_SIZE, seconds=seconds))
    return np.concatenate(values), timings


def signed_preprocessing_fixture():
    """Independent invented signed orbit verifies the generalized statistic rule."""
    import numpy as np
    base = np.array([[2., 5., 3., 8.], [np.nan, 7., -4., 8.], [6., np.nan, 2., 8.]])
    permutation, signs = np.array([1, 0, 2, 3]), np.array([-1., -1., -1., 1.])
    paired = np.concatenate([base, swap_array(base, permutation, signs)])
    mean, scale = symmetric_statistics(paired, permutation, signs)
    x = preprocess(base, mean, scale)
    y = preprocess(swap_array(base, permutation, signs), mean, scale)
    np.testing.assert_array_equal(y, swap_array(x, permutation, signs))
    return dict(passed=True, permutation=permutation.tolist(), signs=signs.tolist(),
                mean=mean.tolist(), scale=scale.tolist(), native_schema_modified=False)


def worker(size, stage, output):
    verify()
    output.mkdir(parents=True, exist_ok=False)
    blocked = RuntimeError('Network disabled for invented-only R64 worker')
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=blocked))
        stack.enter_context(patch.object(socket.socket, 'connect_ex', side_effect=blocked))
        stack.enter_context(patch.object(urllib.request, 'urlopen', side_effect=blocked))
        np, torch = initialize()
        contract = source_contract()
        data = synthetic_data(size)
        signed_fixture = signed_preprocessing_fixture()
        for name, value in data.items():
            np.save(output / (name + '.npy'), value)
            value.flags.writeable = False
        array_before = {name: array_record(value) for name, value in data.items()}
        context = torch.tensor(data['context'], dtype=torch.float64, device='cpu')
        target = torch.tensor(data['target'], dtype=torch.int64, device='cpu')
        tensor_before = dict(context=array_record(context), target=array_record(target))
        Model = official_model_class()
        model = Model(d_in=166, d_num=166, d_out=2, **contract['model_config']).to('cpu').double()
        initial = state_record(model)
        (losses, epoch_losses), fit_seconds = operation(output, 'complete_200_epoch_fit', 'fit',
            lambda: train_complete(model, torch, context, target, contract, output))
        np.save(output / 'training_losses.npy', losses)
        np.save(output / 'epoch_losses.npy', epoch_losses)
        state = state_record(model)
        torch.save({'params': model.state_dict()}, output / 'model.pt')
        write_json(output / 'initial_state_fingerprint.json', initial)
        write_json(output / 'state_fingerprint.json', state)
        original, timings = canonical_predict(model, torch, context, target, data['original'], output, 'original')
        swapped, more = canonical_predict(model, torch, context, target, data['swapped'], output, 'swapped')
        timings.extend(more)
        owner = .5 * (original + swapped[:, [1, 0]])
        reverse_original, more = canonical_predict(model, torch, context, target, data['swapped'], output, 'equivariant_original')
        timings.extend(more)
        reverse_swapped, more = canonical_predict(model, torch, context, target, data['original'], output, 'equivariant_swapped')
        timings.extend(more)
        np.testing.assert_array_equal(.5 * (reverse_original + reverse_swapped[:, [1, 0]]), owner[:, [1, 0]])
        first_batch = data['original'][:QUERY_SIZE]
        repeated, seconds = operation(output, 'repeat_first_query', 'predict',
            lambda: raw_predict(model, torch, context, target, first_batch))
        np.testing.assert_array_equal(repeated, original[:QUERY_SIZE])
        timings.append(dict(name='repeat_first_query', seconds=seconds))
        diagnostics = {}
        for name, index in (('prefix8', np.arange(8)), ('reverse512', np.arange(QUERY_SIZE)[::-1])):
            value, seconds = operation(output, 'diagnostic_' + name, 'predict',
                lambda index=index: raw_predict(model, torch, context, target, data['original'][index]))
            np.save(output / ('diagnostic_' + name + '.npy'), value)
            expected = original[index]
            diagnostics[name] = dict(exact=bool(np.array_equal(value, expected)),
                max_absolute_difference=float(np.max(np.abs(value - expected))), seconds=seconds,
                selection_allowed=False)
        reloaded = Model(d_in=166, d_num=166, d_out=2, **contract['model_config']).to('cpu').double()
        reloaded.load_state_dict(torch.load(output / 'model.pt', map_location='cpu', weights_only=True)['params'], strict=True)
        if state_record(reloaded) != state:
            raise ValueError('Strict checkpoint reload changed tensors')
        reload_probability, more = canonical_predict(reloaded, torch, context, target, data['original'], output, 'reload_original')
        timings.extend(more)
        np.testing.assert_array_equal(reload_probability, original)
        for name, value in (('original_probability', original), ('swapped_probability', swapped), ('owner', owner)):
            np.save(output / (name + '.npy'), value)
        if state_record(model) != state or tensor_before != dict(context=array_record(context), target=array_record(target)):
            raise ValueError('Inference/training changed model or training-context witness')
        if array_before != {name: array_record(value) for name, value in data.items()}:
            raise ValueError('Read-only invented arrays changed')
        recipe = dict(context_rows=CONTEXT_ROWS[size], features=166, class_order=['H', 'A'],
            epochs=EPOCHS, batches_per_epoch=math.ceil(CONTEXT_ROWS[size] / BATCH_SIZE),
            batch_size=BATCH_SIZE, precision='float64', backend='cpu', cpu_threads=CPU_THREADS,
            seed=dict(python=42, numpy_global=43, torch=44, invented_default_rng=640042),
            model_config=contract['model_config'], training_config=contract['training_config'],
            definitions=contract['definitions'], schema=contract['schema'], signed_fixture=signed_fixture,
            preprocessing='Train-only orbit-tied mean and population scale; all-empty zero; constant scale1',
            checkpoint='Final complete200epochs only; no validation/checkpoint selection',
            source_parity='Exact pinned model/helper modules and extracted author training definitions',
            arrays=array_before, context_tensors=tensor_before)
        write_json(output / 'recipe.json', recipe)
        if stage == 'rebuild':
            parent = output.parent / 'first'
            for name in ('initial_state_fingerprint.json', 'state_fingerprint.json', 'recipe.json', 'resolved_optimizer.json'):
                if (output / name).read_bytes() != (parent / name).read_bytes():
                    raise ValueError('Independent fresh refit differs: ' + name)
            for name in ('training_losses', 'epoch_losses', 'original_probability', 'swapped_probability', 'owner'):
                if (output / (name + '.npy')).read_bytes() != (parent / (name + '.npy')).read_bytes():
                    raise ValueError('Independent fresh refit differs: ' + name)
        checks = dict(complete_200_epochs=True, finite_loss=bool(np.isfinite(losses).all()),
            probabilities_valid=probability_checks(owner, len(data['original'])), exact_swap_equivariance=True,
            exact_repeat=True, strict_reload_exact=True, model_state_immutable=True,
            context_state_immutable=True, independent_refit_exact=stage == 'rebuild',
            independent_refit_required_at_final_stage=True, rss_within_limit=peak_bytes() <= LIMIT_BYTES)
        # First-stage workers establish the reference; only rebuild can certify refit.
        worker_pass = all(v for k, v in checks.items() if k != 'independent_refit_exact')
        write_json(output / 'summary.json', dict(passed=worker_pass, size=size, stage=stage,
            checks=checks, fit_seconds=fit_seconds, query_timings=timings,
            diagnostics=diagnostics, peak_bytes=peak_bytes(), synthetic_only=True,
            native_arrays_read=False, artifact_sha256={p.name: digest(p) for p in output.iterdir()
                if p.is_file() and p.name != 'operation.json'}))
    verify()


def supervise(size, stage, output):
    started, peak, aborted = time.monotonic(), 0, None
    destination = output / size / stage
    destination.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = destination.parent / (stage + '.stdout.log')
    stderr_path = destination.parent / (stage + '.stderr.log')
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        process = subprocess.Popen([str(PYTHON), '-B', '-P', '-S', str(Path(__file__).resolve()),
            '--worker', '--size', size, '--stage', stage, '--output', str(destination)], cwd=ROOT,
            env={**os.environ, **ENVIRONMENT}, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            sample = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True)
            if sample.returncode == 0 and sample.stdout.strip():
                peak = max(peak, int(sample.stdout.strip()) * 1024)
            status_path = destination / 'operation.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if peak > LIMIT_BYTES:
                aborted = 'rss_over_6GiB'
            elif time.monotonic() - started > WORKER_SECONDS:
                aborted = 'worker_over_2400seconds'
            elif status.get('active') and time.monotonic() - status['started'] > status['limit_seconds']:
                aborted = status['kind'] + '_over_' + str(status['limit_seconds']) + 'seconds'
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
    if aborted is None and elapsed > WORKER_SECONDS:
        aborted = 'worker_over_2400seconds'
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


def main(output):
    verify()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for size, stage in STAGES:
        records.append(supervise(size, stage, output))
        if not records[-1]['passed']:
            break
    passed = len(records) == len(STAGES) and all(r['passed'] for r in records)
    if passed:
        for size in CONTEXT_ROWS:
            summary = json.loads((output / size / 'rebuild/summary.json').read_text())
            if not summary['checks']['independent_refit_exact']:
                raise ValueError('Final gate lacks exact independent refit')
    write_json(output / 'summary.json', dict(passed=passed, records=records,
        plan_sha256=digest(PLAN), implementation_sha256=digest(__file__),
        synthetic_only=True, native_processing_started=False, native_plan_authorized=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--verify', action='store_true')
    action.add_argument('--worker', action='store_true')
    parser.add_argument('--size', choices=tuple(CONTEXT_ROWS))
    parser.add_argument('--stage', choices=('first', 'rebuild'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROUND / 'runtime_assessment').resolve()
    if not output.is_relative_to(ROUND.resolve()):
        parser.error('All R64 outputs must remain inside its new directory')
    if args.preflight:
        preflight()
    elif args.freeze:
        freeze()
    elif args.verify:
        verify()
        print('R64 frozen inputs verified; no scientific import', flush=True)
    elif args.worker:
        if not args.size or not args.stage:
            parser.error('Worker requires a size and stage')
        try:
            worker(args.size, args.stage, output)
        except BaseException as error:
            if output.exists():
                write_json(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                    synthetic_only=True, native_arrays_read=False))
            raise
    else:
        main(output)
