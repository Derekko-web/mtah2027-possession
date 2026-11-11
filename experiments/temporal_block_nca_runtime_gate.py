"""One invented-only temporal-block NCA gate; root review/freeze/admission required.

Only standard-library source inspection is available before explicit --run.
The pinned author forward/train_epoch are copied as ASTs with enumerated
metadata/mask/gradient additions. Inference calls the pristine author forward.
Complete applicable author licenses are embedded below and retained as notices.
"""
import argparse
import ast
import copy
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
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
ROUND = ROOT / 'data/control_temporal_block_nca_runtime_v1'
PLAN = ROOT / 'experiments/TEMPORAL_BLOCK_NCA_RUNTIME_PLAN.json'
BASIS_PATH = ROOT / 'experiments/modernnca_isolated_runtime_gate.py'
AUTHOR = ROOT / 'data/control_modernnca_isolated_runtime_v1/public_sources'
PYTHON = ROOT / 'data/control_isolated_python_v1/python/bin/python3.12'
CONTEXT_ROWS = {'moderate': 600, 'full': 3000}
QUERY_ROWS = {'moderate': 2500, 'full': 2500}
STAGES = (('moderate', 'first'), ('moderate', 'rebuild'), ('full', 'first'), ('full', 'rebuild'))
EPOCHS, BATCH_SIZE, QUERY_SIZE, CPU_THREADS = 200, 1024, 512, 2
FIT_SECONDS, PREDICT_SECONDS, WORKER_SECONDS = 1800, 300, 2400
LIMIT_BYTES, EXCLUSION = 6 * 1024**3, 250
SOURCE_FILES = ('experiments/temporal_block_nca_runtime_gate.py',
    'experiments/test_temporal_block_nca_source_contract.py',
    'experiments/TEMPORAL_BLOCK_NCA_RUNTIME_PLAN.json',
    'experiments/TEMPORAL_BLOCK_NCA_RESEARCH_20261101.md')
ENVIRONMENT = {name: '2' for name in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
ENVIRONMENT.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='',
    PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
    HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled',
    PYTORCH_ENABLE_MPS_FALLBACK='0', TOKENIZERS_PARALLELISM='false')
_BASIS = None


def basis_module():
    global _BASIS
    if _BASIS is None:
        spec = importlib.util.spec_from_file_location('temporal_nca_immutable_basis', BASIS_PATH)
        _BASIS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BASIS)
    return _BASIS


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for body in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(body)
    return value.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_name(Path(path).name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


class SupportError(ValueError):
    pass


def native_integer(value):
    if type(value) is not int or value < 0 or value > 2**62:
        raise ValueError('Frame metadata must contain nonnegative native integer centers')
    return value


def invented_frames(rows):
    if type(rows) is not int or rows <= 0 or rows % 2:
        raise ValueError('Invented context requires a positive even paired row count')
    base = tuple(10000 + 5 * i for i in range(rows // 2))
    return base + base


def validate_paired_metadata(frames, labels):
    frames, labels = list(frames), list(labels)
    if not frames or len(frames) % 2 or len(labels) != len(frames):
        raise ValueError('Paired metadata shape differs')
    frames = [native_integer(f) for f in frames]
    half = len(frames) // 2
    if frames[:half] != sorted(set(frames[:half])) or any(f % 5 for f in frames[:half]):
        raise ValueError('Original centers must be distinct ascending native 5-Hz centers')
    if frames[half:] != frames[:half]:
        raise ValueError('Original and swapped mates must share exact frame centers')
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError('Own-training labels must be binary H0/A1')
    if labels[half:] != [1-y for y in labels[:half]]:
        raise ValueError('Swapped own-training labels must be complemented')
    return True


def temporal_candidate_literal(total, batch_ids, sample_permutation):
    batch_ids, sample_permutation = list(batch_ids), list(sample_permutation)
    if type(total) is not int or total <= 0 or not batch_ids:
        raise ValueError('Invalid training size or empty mini-batch')
    if any(type(i) is not int or i < 0 or i >= total for i in batch_ids) or len(set(batch_ids)) != len(batch_ids):
        raise ValueError('Batch indices must be unique valid row positions')
    external = [i for i in range(total) if i not in set(batch_ids)]
    if (any(type(i) is not int for i in sample_permutation)
            or sorted(sample_permutation) != list(range(len(external)))):
        raise ValueError('Neighbor draw must be one full permutation of external row positions')
    sampled = [external[i] for i in sample_permutation[:len(external)//2]]
    return dict(external=external, sampled=sampled, candidates=batch_ids + sampled)


def temporal_mask_literal(anchor_frames, candidate_frames):
    anchors = [native_integer(f) for f in anchor_frames]
    candidates = [native_integer(f) for f in candidate_frames]
    if not anchors or not candidates:
        raise ValueError('Temporal mask requires nonempty anchors and combined candidates')
    return [[abs(i-j) > EXCLUSION for j in candidates] for i in anchors]


def support_literal(mask, anchor_labels, candidate_labels):
    anchor_labels, candidate_labels = list(anchor_labels), list(candidate_labels)
    if len(mask) != len(anchor_labels) or any(len(row) != len(candidate_labels) for row in mask):
        raise ValueError('Support mask and labels are misaligned')
    if any(type(y) is not int or y not in (0, 1) for y in anchor_labels + candidate_labels):
        raise ValueError('Support labels must be binary own-training labels')
    records = []
    for row, own in zip(mask, anchor_labels):
        if any(type(v) is not bool for v in row):
            raise ValueError('Support mask must be boolean')
        positive = sum(keep and label == own for keep, label in zip(row, candidate_labels))
        eligible = sum(row)
        if not eligible:
            raise SupportError('Zero eligible temporal neighbors; fixed method fails without resampling')
        if not positive:
            raise SupportError('Zero eligible own-class neighbors; fixed method fails without resampling')
        records.append(dict(eligible=eligible, positive=positive, opposite=eligible-positive))
    return records


def clock_sample():
    return dict(utc_ns=time.time_ns(), mono_ns=time.monotonic_ns())


def elapsed_ns(start, end):
    if set(start) != {'utc_ns', 'mono_ns'} or set(end) != set(start):
        raise ValueError('Dual integer-nanosecond clock witnesses are required')
    if any(type(v) is not int for v in list(start.values()) + list(end.values())):
        raise ValueError('Clock witnesses must be integer nanoseconds')
    elapsed = {key: end[key]-start[key] for key in start}
    if min(elapsed.values()) < 0:
        raise ValueError('Clock rollback invalidates resource evidence')
    return elapsed


def operation_violation(start, end, limit_seconds):
    elapsed = elapsed_ns(start, end)
    if elapsed['utc_ns'] > limit_seconds * 10**9:
        return 'utc_over_' + str(limit_seconds) + 'seconds'
    if elapsed['mono_ns'] > limit_seconds * 10**9:
        return 'monotonic_over_' + str(limit_seconds) + 'seconds'
    return None


def adapted_forward_node():
    """Copy the pinned author forward; these are its only enumerated changes."""
    node = copy.deepcopy(basis_module().definition_node(
        AUTHOR / 'TALENT/model/models/modernNCA.py', 'forward', 'ModernNCA'))
    node.args.args.extend([ast.arg(arg='anchor_frames'), ast.arg(arg='candidate_frames')])
    node.args.defaults.extend([ast.Constant(None), ast.Constant(None)])
    prefix = ast.parse('''
if not is_train:
    if y is not None or anchor_frames is not None or candidate_frames is not None:
        raise ValueError('Inference forbids query labels and temporal metadata')
    return self.pristine_author_forward(x, y, candidate_x, candidate_y, is_train)
validate_tensor_metadata(x, y, candidate_x, candidate_y, anchor_frames, candidate_frames)
''').body
    # The first author branch performs exactly one unchanged uniform randperm.
    first = node.body[0]
    first.body.extend(ast.parse('''
candidate_frames = candidate_frames[sample_idx]
self.sample_records.append(array_record(sample_idx))
''').body)
    append = next(n for n in node.body if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name) and n.test.id == 'is_train'
        and any(isinstance(c, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'candidate_x' for t in c.targets)
            and isinstance(c.value, ast.Call) and isinstance(c.value.func, ast.Attribute)
            and c.value.func.attr == 'cat' for c in n.body))
    append.body.extend(ast.parse('''
candidate_frames = torch.cat([anchor_frames, candidate_frames])
candidate_labels = candidate_y
''').body)
    diagonal = next(n for n in node.body if isinstance(n, ast.If)
        and any(isinstance(c, ast.Assign) and isinstance(c.value, ast.Call)
            and isinstance(c.value.func, ast.Attribute) and c.value.func.attr == 'fill_diagonal_' for c in n.body))
    diagonal.body.extend(ast.parse('''
distances = apply_temporal_mask(self, distances, anchor_frames, candidate_frames, y, candidate_labels)
''').body)
    node.body = prefix + node.body
    return ast.fix_missing_locations(node)


def adapted_train_epoch_node():
    node = copy.deepcopy(basis_module().definition_node(
        AUTHOR / 'TALENT/model/methods/modernNCA.py', 'train_epoch', 'ModernNCAMethod'))
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'model']
    if len(calls) != 1:
        raise ValueError('Pinned author training call changed')
    calls[0].keywords.extend([
        ast.keyword(arg='anchor_frames', value=ast.parse('self.frames[batch_idx]', mode='eval').body),
        ast.keyword(arg='candidate_frames', value=ast.parse('self.frames[candidate_indices]', mode='eval').body)])
    loop = next(n for n in node.body if isinstance(n, ast.For))
    backward = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == 'backward')
    loop.body[backward+1:backward+1] = ast.parse('finite_gradient_check(self)').body
    return ast.fix_missing_locations(node)


def ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def adapter_parity_contract():
    """Remove only the enumerated additions and recover both exact author ASTs."""
    basis = basis_module()
    original_forward = basis.definition_node(AUTHOR / 'TALENT/model/models/modernNCA.py', 'forward', 'ModernNCA')
    restored = copy.deepcopy(adapted_forward_node())
    restored.args = copy.deepcopy(original_forward.args)
    restored.body = restored.body[2:]
    restored.body[0].body = restored.body[0].body[:-2]
    append = next(n for n in restored.body if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name) and n.test.id == 'is_train'
        and any(isinstance(c, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'candidate_frames' for t in c.targets) for c in n.body))
    append.body = append.body[:-2]
    diagonal = next(n for n in restored.body if isinstance(n, ast.If)
        and any(isinstance(c, ast.Assign) and isinstance(c.value, ast.Call)
            and isinstance(c.value.func, ast.Name) and c.value.func.id == 'apply_temporal_mask' for c in n.body))
    diagonal.body = diagonal.body[:-1]
    original_train = basis.definition_node(AUTHOR / 'TALENT/model/methods/modernNCA.py', 'train_epoch', 'ModernNCAMethod')
    restored_train = copy.deepcopy(adapted_train_epoch_node())
    call = next(n for n in ast.walk(restored_train) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'model')
    call.keywords = call.keywords[:-2]
    loop = next(n for n in restored_train.body if isinstance(n, ast.For))
    loop.body = [n for n in loop.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == 'finite_gradient_check')]
    if ast_sha(restored) != ast_sha(original_forward) or ast_sha(restored_train) != ast_sha(original_train):
        raise ValueError('Adapter changes extend beyond the enumerated metadata/mask/gradient additions')
    return dict(restored_forward_exact_author_ast=True, restored_train_epoch_exact_author_ast=True,
        pristine_inference_forward_ast_sha256=ast_sha(original_forward),
        original_train_epoch_ast_sha256=ast_sha(original_train))


def source_contract():
    plan = json.loads(PLAN.read_text())
    for path, expected in plan['source_basis_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Immutable numerical/source basis changed: ' + path)
    basis = basis_module()
    original = basis.source_contract()
    if (basis.ENVIRONMENT != ENVIRONMENT or basis.EPOCHS != EPOCHS
            or basis.BATCH_SIZE != BATCH_SIZE or basis.CPU_THREADS != CPU_THREADS
            or basis.QUERY_SIZE != QUERY_SIZE or basis.PYTHON != PYTHON):
        raise ValueError('Unchanged NCA numerical/runtime contract differs')
    for name in ('LICENSE', 'TABR_LICENSE'):
        if (AUTHOR / name).read_text() not in LICENSE_TEXT:
            raise ValueError('Complete applicable MIT notice missing from new adapter source')
    return dict(original=original, model_config=original['model_config'],
        training_config=original['training_config'], schema=original['schema'],
        definitions=original['definitions'], adapted_forward_ast_sha256=ast_sha(adapted_forward_node()),
        adapted_train_epoch_ast_sha256=ast_sha(adapted_train_epoch_node()),
        adapter_parity=adapter_parity_contract(),
        inference='Pristine author forward identity; rejects query labels and frame metadata',
        deviation='Only training soft-neighbor eligibility abs(native_center_i-native_center_j)>250')


def runtime_references():
    files, links = basis_module().runtime_references()
    contract = source_contract()
    files.update(json.loads(PLAN.read_text())['source_basis_sha256'])
    files.update(contract['schema']['source_sha256'])
    for relative in basis_module().acquisition_module().SOURCES:
        path = AUTHOR / relative
        files[str(path.relative_to(ROOT))] = digest(path)
    return files, links


def assert_source_only():
    if any(name in sys.modules for name in ('torch', 'numpy', 'pandas', 'scipy', 'sklearn', 'TALENT')):
        raise ValueError('Scientific import occurred during source-only preparation')


def preflight():
    if (ROUND / 'protocol_freeze').exists() or (ROUND / 'runtime_assessment').exists():
        raise ValueError('Cannot replace preparation evidence after freeze or execution')
    ROUND.mkdir(parents=True, exist_ok=True)
    contract = source_contract()
    files, links = runtime_references()
    basis_module().verify_mapping(files, links)
    for relative in SOURCE_FILES:
        if relative.endswith('.py'):
            compile((ROOT / relative).read_bytes(), relative, 'exec')
    notices = {}
    for name in ('LICENSE', 'TABR_LICENSE'):
        destination = ROUND / 'notices' / name
        destination.parent.mkdir(exist_ok=True)
        body = (AUTHOR / name).read_bytes()
        if destination.exists() and destination.read_bytes() != body:
            raise ValueError('Previously retained author notice differs')
        if not destination.exists():
            destination.write_bytes(body)
        notices[str(destination.relative_to(ROOT))] = digest(destination)
    write_json(ROUND / 'runtime_files.json', files)
    write_json(ROUND / 'runtime_symlinks.json', links)
    write_json(ROUND / 'acquisition.json', dict(source_only=True, new_dependencies=False,
        immutable_author_revision='39b54024902955e51063a34bce17d514cd68ff42',
        inherited_complete_runtime=True, runtime_files=len(files), symlinks=len(links),
        inherited_notice_files=sum(Path(p).name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE', 'AUTHORS')) for p in files),
        retained_notices_sha256=notices, scientific_imports=False, native_arrays_read=False,
        model_calls=False))
    assert_source_only()
    write_json(ROUND / 'source_preflight.json', dict(passed=True, source_only=True,
        source_sha256={p: digest(ROOT / p) for p in SOURCE_FILES}, contract=contract,
        runtime_files_sha256=digest(ROUND / 'runtime_files.json'),
        runtime_symlinks_sha256=digest(ROUND / 'runtime_symlinks.json'),
        acquisition_sha256=digest(ROUND / 'acquisition.json'),
        scientific_imports=False, model_calls=False, native_arrays_read=False,
        pending_root_review=True, pending_freeze=True, pending_resource_admission=True))
    print('Temporal-block NCA source-only preflight passed; model work remains unauthorized', flush=True)


def freeze():
    assert_source_only()
    review = json.loads((ROUND / 'independent_review.json').read_text())
    proof = json.loads((ROUND / 'source_preflight.json').read_text())
    current = {p: digest(ROOT / p) for p in SOURCE_FILES}
    if (not review.get('approved_for_freeze') or review.get('unresolved', -1) != 0
            or review.get('source_sha256') != current or proof['source_sha256'] != current):
        raise ValueError('Final independent root source review and preflight must bind all four files')
    files, links = runtime_references()
    basis_module().verify_mapping(files, links)
    if files != json.loads((ROUND / 'runtime_files.json').read_text()) or links != json.loads((ROUND / 'runtime_symlinks.json').read_text()):
        raise ValueError('Prepared complete runtime binding differs')
    files.update(current)
    for name in ('runtime_files.json', 'runtime_symlinks.json', 'acquisition.json', 'source_preflight.json', 'independent_review.json'):
        files[str((ROUND / name).relative_to(ROOT))] = digest(ROUND / name)
    for path in (ROUND / 'notices').iterdir():
        files[str(path.relative_to(ROOT))] = digest(path)
    folder = ROUND / 'protocol_freeze'
    folder.mkdir(exist_ok=False)
    (folder / 'plan.json').write_bytes(PLAN.read_bytes())
    write_json(folder / 'implementation.json', files)
    write_json(folder / 'symlinks.json', links)
    write_json(folder / 'freeze.json', dict(source_sha256=current,
        implementation_sha256=digest(folder / 'implementation.json'),
        symlinks_sha256=digest(folder / 'symlinks.json'), scientific_imports=False,
        model_calls=False, native_arrays_read=False, resource_admission_required=True))


def verify():
    folder = ROUND / 'protocol_freeze'
    if (folder / 'plan.json').read_bytes() != PLAN.read_bytes():
        raise ValueError('Frozen new plan differs')
    frozen = json.loads((folder / 'freeze.json').read_text())
    if (digest(folder / 'implementation.json') != frozen['implementation_sha256']
            or digest(folder / 'symlinks.json') != frozen['symlinks_sha256']):
        raise ValueError('Frozen complete binding maps differ')
    basis_module().verify_mapping(json.loads((folder / 'implementation.json').read_text()),
        json.loads((folder / 'symlinks.json').read_text()))
    source_contract()


def require_admission():
    path = ROUND / 'resource_admission.json'
    if not path.exists():
        raise PermissionError('Pending root resource admission after GP closes or passes')
    value = json.loads(path.read_text())
    if (value.get('approved_for_model_calls') is not True
            or value.get('scope') != 'invented_only_temporal_block_nca_v1'
            or value.get('predecessor_gp_closed_or_passed') is not True
            or value.get('other_model_workers_stopped') is not True
            or value.get('freeze_sha256') != digest(ROUND / 'protocol_freeze/freeze.json')):
        raise PermissionError('Root resource admission does not bind this exact frozen invented-only gate')
    return value


def operation(output, name, kind, callback):
    if kind not in ('fit', 'predict'):
        raise ValueError('Unknown resource operation')
    limit = FIT_SECONDS if kind == 'fit' else PREDICT_SECONDS
    started = clock_sample()
    record = dict(name=name, kind=kind, active=True, started=started, limit_seconds=limit)
    write_json(output / 'operation.json', record)
    try:
        result = callback()
    except BaseException as error:
        record.update(active=False, finished=clock_sample(), failed=True, error=repr(error))
        write_json(output / 'operation.json', record)
        with (output / 'operations.jsonl').open('a') as log:
            log.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        raise
    finished = clock_sample()
    durations = elapsed_ns(started, finished)
    violation = operation_violation(started, finished, limit)
    peak = basis_module().peak_bytes()
    record.update(active=False, finished=finished, elapsed=durations, failed=bool(violation),
        violation=violation, peak_bytes=peak)
    write_json(output / 'operation.json', record)
    with (output / 'operations.jsonl').open('a') as log:
        log.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    if violation or peak > LIMIT_BYTES:
        raise RuntimeError('Prospective resource bound failed: ' + (violation or 'rss_over_6GiB'))
    return result, durations


def validate_tensor_metadata(x, y, candidate_x, candidate_y, anchor_frames, candidate_frames):
    if (anchor_frames is None or candidate_frames is None or y is None
            or anchor_frames.dtype != torch.int64 or candidate_frames.dtype != torch.int64
            or anchor_frames.ndim != 1 or candidate_frames.ndim != 1
            or len(anchor_frames) != len(x) or len(candidate_frames) != len(candidate_x)
            or len(y) != len(x) or len(candidate_y) != len(candidate_x)):
        raise ValueError('Training frame metadata must follow exact anchor/external row permutations')
    if (not len(x) or not bool(((anchor_frames >= 0) & (anchor_frames <= 2**62)).all())
            or not bool(((candidate_frames >= 0) & (candidate_frames <= 2**62)).all())):
        raise ValueError('Empty anchors or native frame metadata outside its integer range')


def array_record(value):
    return basis_module().array_record(value)


def apply_temporal_mask(model, distances, anchor_frames, candidate_frames, labels, candidate_labels):
    allowed = (anchor_frames[:, None] - candidate_frames[None, :]).abs() > EXCLUSION
    eligible = allowed.sum(dim=1)
    positive = (allowed & (labels[:, None] == candidate_labels[None, :])).sum(dim=1)
    if not bool((eligible > 0).all()):
        raise SupportError('Zero eligible temporal neighbors; no resampling or skipped anchor allowed')
    if not bool((positive > 0).all()):
        raise SupportError('Zero eligible own-class neighbors; no resampling or skipped anchor allowed')
    opposite = eligible-positive
    model.support_records.append(dict(anchor_frames=array_record(anchor_frames),
        candidate_frames=array_record(candidate_frames), eligible=array_record(eligible),
        positive=array_record(positive), opposite=array_record(opposite),
        eligible_counts=eligible.tolist(), positive_counts=positive.tolist(), opposite_counts=opposite.tolist(),
        anchors=len(labels), candidates=len(candidate_labels),
        minimum_eligible=int(eligible.min().item()), minimum_positive=int(positive.min().item()),
        missing_opposite=int((opposite == 0).sum().item())))
    return distances.masked_fill(~allowed, torch.inf)


def temporal_model_class(torch_module):
    global torch
    torch = torch_module
    AuthorModel = basis_module().official_model_class()
    namespace = dict(torch=torch, F=torch.nn.functional,
        validate_tensor_metadata=validate_tensor_metadata, apply_temporal_mask=apply_temporal_mask,
        array_record=array_record, __name__='temporal_block_nca_author_adapter')
    exec(compile(ast.Module(body=[adapted_forward_node()], type_ignores=[]),
        '<pinned-author-forward-with-fixed-temporal-mask>', 'exec'), namespace)
    Model = type('TemporalBlockNCA', (AuthorModel,), dict(forward=namespace['forward'],
        pristine_author_forward=AuthorModel.forward))
    if Model.pristine_author_forward is not AuthorModel.forward:
        raise ValueError('Inference must retain pristine author function identity')
    return Model


def finite_gradient_check(trainer):
    parameters = list(trainer.model.parameters())
    if not parameters or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError('Every trainable parameter requires a finite actual gradient')
    trainer.finite_gradient_steps += 1


def tree_record(value):
    if hasattr(value, 'detach'):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite fitted model or optimizer tensor state')
        return array_record(value)
    if isinstance(value, dict):
        return {str(k): tree_record(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [tree_record(v) for v in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('Unrecognized or nonfinite optimizer/runtime state')


def model_runtime_record(model):
    return dict(tensors=tree_record(model.state_dict()),
        gradients={name: tree_record(parameter.grad) for name, parameter in model.named_parameters()},
        training_flags={name: child.training for name, child in model.named_modules()},
        hyperparameters={name: getattr(model, name) for name in (
            'd_in', 'd_out', 'd_num', 'dim', 'dropout', 'd_block', 'n_blocks', 'T', 'sample_rate')},
        samples=copy.deepcopy(model.sample_records), supports=copy.deepcopy(model.support_records))


def optimizer_runtime_record(optimizer):
    return dict(state_dict=tree_record(optimizer.state_dict()), defaults=tree_record(optimizer.defaults))


def synthetic_data(size):
    import numpy as np
    basis = basis_module()
    schema = basis.schema_module().schema_contract()
    permutation, signs = np.asarray(schema['permutation']), np.asarray(schema['signs'], dtype=np.float64)
    rows, queries = CONTEXT_ROWS[size], QUERY_ROWS[size]
    generator = np.random.default_rng(640042)
    base = generator.normal(size=(rows//2, 166)).astype(np.float64)
    query = generator.normal(size=(queries, 166)).astype(np.float64)
    h, a = schema['columns'].index('h__distance_1'), schema['columns'].index('a__distance_1')
    target = (base[:, h]-base[:, a]+.25*base[:, 0] > 0).astype(np.int64)
    base.flat[::37], query.flat[::41] = np.nan, np.nan
    empty, constant = schema['columns'].index('h__set_latent_31'), schema['columns'].index('h__set_latent_30')
    for value in (base, query):
        value[:, [empty, permutation[empty]]] = np.nan
        value[:, [constant, permutation[constant]]] = 1.
    raw_context = np.concatenate([base, basis.swap_array(base, permutation, signs)])
    target = np.r_[target, 1-target]
    frames = np.asarray(invented_frames(rows), dtype=np.int64)
    validate_paired_metadata(frames.tolist(), target.tolist())
    mean, scale = basis.symmetric_statistics(raw_context, permutation, signs)
    context = basis.preprocess(raw_context, mean, scale)
    original = basis.preprocess(query, mean, scale)
    swapped = basis.preprocess(basis.swap_array(query, permutation, signs), mean, scale)
    np.testing.assert_array_equal(swapped, basis.swap_array(original, permutation, signs))
    np.testing.assert_array_equal(context[rows//2:], basis.swap_array(context[:rows//2], permutation, signs))
    if context.shape != (rows, 166) or not np.isfinite(context).all() or not np.isfinite(original).all():
        raise ValueError('Invented numerical schema differs')
    return dict(raw_context=raw_context, context=context, target=target, frame_ids=frames,
        raw_query=query, original=original, swapped=swapped, mean=mean, scale=scale,
        permutation=permutation, signs=signs)


def train_complete(model, torch_module, context, target, frames, contract, output):
    import numpy as np
    losses = []
    def criterion(prediction, labels):
        loss = torch_module.nn.functional.nll_loss(prediction, labels)
        if not math.isfinite(loss.item()):
            raise ValueError('Nonfinite actual training loss')
        losses.append(loss.item())
        return loss
    original = basis_module().extracted_definitions(torch_module)
    namespace = dict(torch=torch_module, Averager=original['Averager'],
        make_random_batches=original['make_random_batches'], finite_gradient_check=finite_gradient_check)
    exec(compile(ast.Module(body=[adapted_train_epoch_node()], type_ignores=[]),
        '<pinned-author-train-epoch-with-frame-metadata>', 'exec'), namespace)
    optimizer = torch_module.optim.AdamW(model.parameters(), **contract['training_config'])
    trainer = types.SimpleNamespace(model=model, N={'train': context}, C=None, y={'train': target},
        frames=frames, train_size=len(context), train_indices=torch_module.arange(len(context), device='cpu'),
        args=types.SimpleNamespace(device=torch_module.device('cpu'), batch_size=BATCH_SIZE,
            use_float=False, config={'training': contract['training_config']}), train_step=0,
        trlog={'train_loss': []}, train_loader=range(math.ceil(len(context)/BATCH_SIZE)),
        criterion=criterion, optimizer=optimizer, finite_gradient_steps=0)
    write_json(output / 'resolved_optimizer.json', tree_record(optimizer.defaults))
    with (output / 'training.log').open('w') as log, redirect_stdout(log):
        for epoch in range(EPOCHS):
            namespace['train_epoch'](trainer, epoch)
            log.flush()
    wanted = EPOCHS * math.ceil(len(context)/BATCH_SIZE)
    if (trainer.train_step != wanted or len(losses) != wanted
            or len(trainer.trlog['train_loss']) != EPOCHS or trainer.finite_gradient_steps != wanted
            or len(model.support_records) != wanted or len(model.sample_records) != wanted):
        raise ValueError('Incomplete final200 epoch/gradient/support evidence')
    return (np.asarray(losses, dtype=np.float64), np.asarray(trainer.trlog['train_loss'], dtype=np.float64),
        optimizer, trainer.finite_gradient_steps)


def canonical_predict(model, torch_module, context, target, query, output, name):
    import numpy as np
    values, timings = [], []
    for start in range(0, len(query), QUERY_SIZE):
        batch = query[start:start+QUERY_SIZE]
        count = len(batch)
        if count < QUERY_SIZE:
            batch = np.concatenate([batch, np.repeat(batch[-1:], QUERY_SIZE-count, axis=0)])
        value, durations = operation(output, f'{name}_{start}', 'predict',
            lambda batch=batch: basis_module().raw_predict(model, torch_module, context, target, batch))
        values.append(value[:count])
        timings.append(dict(name=name, start=start, real_rows=count, call_rows=QUERY_SIZE, elapsed=durations))
    return np.concatenate(values), timings


def worker(size, stage, output):
    verify()
    require_admission()
    output.mkdir(parents=True, exist_ok=False)
    blocked = RuntimeError('Network disabled for invented-only temporal-block NCA worker')
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=blocked))
        stack.enter_context(patch.object(socket.socket, 'connect_ex', side_effect=blocked))
        stack.enter_context(patch.object(urllib.request, 'urlopen', side_effect=blocked))
        basis = basis_module()
        np, torch_module = basis.initialize()
        contract, data = source_contract(), synthetic_data(size)
        signed_fixture = basis.signed_preprocessing_fixture()
        for name, value in data.items():
            np.save(output / (name+'.npy'), value)
            value.flags.writeable = False
        arrays_before = {name: array_record(value) for name, value in data.items()}
        context = torch_module.tensor(data['context'], dtype=torch_module.float64, device='cpu')
        target = torch_module.tensor(data['target'], dtype=torch_module.int64, device='cpu')
        frames = torch_module.tensor(data['frame_ids'], dtype=torch_module.int64, device='cpu')
        tensors_before = dict(context=array_record(context), target=array_record(target), frame_ids=array_record(frames))
        Model = temporal_model_class(torch_module)
        model = Model(d_in=166, d_num=166, d_out=2, **contract['model_config']).to('cpu').double()
        model.support_records, model.sample_records = [], []
        initial = basis.state_record(model)
        result, fit_elapsed = operation(output, 'complete_200_epoch_fit', 'fit',
            lambda: train_complete(model, torch_module, context, target, frames, contract, output))
        losses, epoch_losses, optimizer, gradient_steps = result
        model.eval()
        np.save(output / 'training_losses.npy', losses)
        np.save(output / 'epoch_losses.npy', epoch_losses)
        state, optimizer_state = model_runtime_record(model), optimizer_runtime_record(optimizer)
        write_json(output / 'initial_state_fingerprint.json', initial)
        write_json(output / 'state_fingerprint.json', state)
        write_json(output / 'optimizer_fingerprint.json', optimizer_state)
        write_json(output / 'realized_support.json', model.support_records)
        write_json(output / 'sampling_trace.json', model.sample_records)
        torch_module.save(dict(params=model.state_dict(), optimizer=optimizer.state_dict(),
            gradients={name: parameter.grad for name, parameter in model.named_parameters()}), output / 'model.pt')
        timings = []
        def predict(value, name, chosen=model):
            probability, more = canonical_predict(chosen, torch_module, context, target, value, output, name)
            timings.extend(more)
            return probability
        original = predict(data['original'], 'original')
        swapped = predict(data['swapped'], 'swapped')
        owner = .5*(original+swapped[:, [1, 0]])
        reverse_original = predict(data['swapped'], 'equivariant_original')
        reverse_swapped = predict(data['original'], 'equivariant_swapped')
        np.testing.assert_array_equal(.5*(reverse_original+reverse_swapped[:, [1, 0]]), owner[:, [1, 0]])
        repeated, repeat_elapsed = operation(output, 'repeat_first_query', 'predict',
            lambda: basis.raw_predict(model, torch_module, context, target, data['original'][:QUERY_SIZE]))
        np.testing.assert_array_equal(repeated, original[:QUERY_SIZE])
        np.save(output / 'repeat_probability.npy', repeated)
        timings.append(dict(name='repeat_first_query', elapsed=repeat_elapsed))
        diagnostics = {}
        for name, index in (('prefix8', np.arange(8)), ('reverse512', np.arange(QUERY_SIZE)[::-1])):
            value, durations = operation(output, 'diagnostic_'+name, 'predict',
                lambda index=index: basis.raw_predict(model, torch_module, context, target, data['original'][index]))
            np.save(output / ('diagnostic_'+name+'.npy'), value)
            diagnostics[name] = dict(exact=bool(np.array_equal(value, original[index])),
                max_absolute_difference=float(np.max(np.abs(value-original[index]))),
                elapsed=durations, selection_allowed=False)
        reloaded = Model(d_in=166, d_num=166, d_out=2, **contract['model_config']).to('cpu').double()
        saved = torch_module.load(output / 'model.pt', map_location='cpu', weights_only=True)
        reloaded.load_state_dict(saved['params'], strict=True)
        if set(saved['gradients']) != set(dict(reloaded.named_parameters())):
            raise ValueError('Strict gradient checkpoint parameter names differ')
        for name, parameter in reloaded.named_parameters():
            gradient = saved['gradients'][name]
            parameter.grad = None if gradient is None else gradient.clone()
        reloaded.eval()
        reload_optimizer = torch_module.optim.AdamW(reloaded.parameters(), **contract['training_config'])
        reload_optimizer.load_state_dict(saved['optimizer'])
        reloaded.support_records, reloaded.sample_records = copy.deepcopy(model.support_records), copy.deepcopy(model.sample_records)
        if model_runtime_record(reloaded) != state or optimizer_runtime_record(reload_optimizer) != optimizer_state:
            raise ValueError('Strict model/optimizer checkpoint reload differs')
        reload_original = predict(data['original'], 'reload_original', reloaded)
        reload_swapped = predict(data['swapped'], 'reload_swapped', reloaded)
        np.testing.assert_array_equal(reload_original, original)
        np.testing.assert_array_equal(reload_swapped, swapped)
        for name, value in (('original_probability', original), ('swapped_probability', swapped),
                ('owner', owner), ('reload_original_probability', reload_original), ('reload_swapped_probability', reload_swapped)):
            np.save(output / (name+'.npy'), value)
        if (model_runtime_record(model) != state or optimizer_runtime_record(optimizer) != optimizer_state
                or model_runtime_record(reloaded) != state or optimizer_runtime_record(reload_optimizer) != optimizer_state
                or tensors_before != dict(context=array_record(context), target=array_record(target), frame_ids=array_record(frames))
                or arrays_before != {name: array_record(value) for name, value in data.items()}):
            raise ValueError('Inference changed model/optimizer/context/metadata state')
        recipe = dict(context_rows=CONTEXT_ROWS[size], query_rows=QUERY_ROWS[size], features=166,
            class_order=['H', 'A'], epochs=EPOCHS, batch_size=BATCH_SIZE, query_size=QUERY_SIZE,
            batches_per_epoch=math.ceil(CONTEXT_ROWS[size]/BATCH_SIZE), backend='cpu', precision='float64',
            cpu_threads=CPU_THREADS, seed=dict(python=42, numpy_global=43, torch=44, invented_default_rng=640042),
            exclusion_native_frames=EXCLUSION, sampling='Unchanged uniform row floor(.5*(U minus B)); C=[B,S]',
            frame_metadata='Paired native centers outside all166 feature/preprocessing/inference inputs',
            objective='Mean NLL log(p_y+1e-7), unsquaredL2/T1, block mask before softmax/label aggregation',
            zero_support='Fail entire method on zeroeligible OR zeroownclass; no resample/skip/fallback',
            final_checkpoint_only=True, contract=contract, arrays=arrays_before, context_tensors=tensors_before,
            signed_fixture=signed_fixture, finite_gradient_steps=gradient_steps)
        write_json(output / 'recipe.json', recipe)
        if stage == 'rebuild':
            parent = output.parent / 'first'
            json_names = ('initial_state_fingerprint', 'state_fingerprint', 'optimizer_fingerprint',
                'resolved_optimizer', 'realized_support', 'sampling_trace', 'recipe')
            array_names = ('training_losses', 'epoch_losses', 'original_probability', 'swapped_probability',
                'owner', 'repeat_probability', 'reload_original_probability', 'reload_swapped_probability')
            for name in [n+'.json' for n in json_names] + [n+'.npy' for n in array_names] + [n+'.npy' for n in data]:
                if (output / name).read_bytes() != (parent / name).read_bytes():
                    raise ValueError('Independent fresh200 epoch refit differs: '+name)
        checks = dict(complete_200_epochs=True, finite_losses=bool(np.isfinite(losses).all()),
            finite_gradients=True, all_realized_support_positive=True, temporal_metadata_aligned=True,
            probabilities_valid=basis.probability_checks(owner, QUERY_ROWS[size]), exact_swap_equivariance=True,
            exact_repeat=True, strict_model_optimizer_reload_exact=True, full_original_swapped_reload_exact=True,
            model_optimizer_context_metadata_immutable=True, inference_pristine_author_function=True,
            independent_refit_exact=stage == 'rebuild', rss_within_limit=basis.peak_bytes() <= LIMIT_BYTES,
            dual_clock_operations_within_limit=True)
        worker_pass = all(v for k, v in checks.items() if k != 'independent_refit_exact')
        write_json(output / 'summary.json', dict(passed=worker_pass, size=size, stage=stage, checks=checks,
            fit_elapsed=fit_elapsed, query_timings=timings, diagnostics=diagnostics,
            peak_bytes=basis.peak_bytes(), synthetic_only=True, native_arrays_read=False,
            artifact_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'operation.json'}))
    verify()


def supervision_violation(started, now, status, peak):
    if peak > LIMIT_BYTES:
        return 'rss_over_6GiB'
    exceeded = operation_violation(started, now, WORKER_SECONDS)
    if exceeded:
        return 'worker_'+exceeded
    if status.get('active'):
        kind = status.get('kind')
        expected = FIT_SECONDS if kind == 'fit' else PREDICT_SECONDS if kind == 'predict' else None
        if expected is None or status.get('limit_seconds') != expected:
            raise ValueError('Unknown or altered active operation resource contract')
        exceeded = operation_violation(status['started'], now, expected)
        if exceeded:
            return kind+'_'+exceeded
    return None


def supervise(size, stage, output):
    started, peak, aborted = clock_sample(), 0, None
    destination = output / size / stage
    destination.parent.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = destination.parent / (stage+'.stdout.log'), destination.parent / (stage+'.stderr.log')
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        process = subprocess.Popen([str(PYTHON), '-B', '-P', '-S', str(Path(__file__).resolve()),
            '--worker', '--size', size, '--stage', stage, '--output', str(destination)], cwd=ROOT,
            env={**os.environ, **ENVIRONMENT}, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            sample = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True)
            if sample.returncode == 0 and sample.stdout.strip():
                peak = max(peak, int(sample.stdout.strip())*1024)
            status_path = destination / 'operation.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            try:
                aborted = supervision_violation(started, clock_sample(), status, peak)
            except (KeyError, TypeError, ValueError) as error:
                aborted = 'invalid_clock_or_operation_witness:'+repr(error)
            if aborted:
                process.terminate()
                break
            time.sleep(.1)
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=10)
    finished = clock_sample()
    summary_path = destination / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if aborted is None:
        try:
            aborted = supervision_violation(started, finished, {}, max(peak, summary.get('peak_bytes', 0)))
        except ValueError as error:
            aborted = 'invalid_clock_witness:'+repr(error)
    try:
        durations, clock_error = elapsed_ns(started, finished), None
    except ValueError as error:
        durations, clock_error = None, repr(error)
        aborted = aborted or 'invalid_clock_witness:'+clock_error
    record = dict(size=size, stage=stage, started=started, finished=finished,
        elapsed=durations, clock_error=clock_error, exit_code=code, aborted=aborted,
        passed=aborted is None and code == 0 and summary.get('passed', False),
        largest_sampled_rss_bytes=peak, summary_sha256=digest(summary_path) if summary_path.exists() else None,
        stdout_sha256=digest(stdout_path), stderr_sha256=digest(stderr_path))
    write_json(destination.parent / (stage+'.supervision.json'), record)
    print(size, stage, 'passed', record['passed'], flush=True)
    return record


def main(output):
    verify()
    admission = require_admission()
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
                raise ValueError('Final gate lacks exact independent fresh refit')
    write_json(output / 'summary.json', dict(passed=passed, records=records,
        plan_sha256=digest(PLAN), implementation_sha256=digest(__file__), admission=admission,
        synthetic_only=True, native_processing_started=False, native_plan_authorized=False))
    if not passed:
        raise SystemExit(1)


LICENSE_TEXT = 'MIT License\n\nCopyright (c) 2024 Siyang Liu, Haorun Cai, Qile Zhou, Han-Jia Ye\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n\nCopyright (c) 2023 Authors of "TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning"\n\nPermission is hereby granted, free of charge, to any person obtaining\na copy of this software and associated documentation files (the\n"Software"), to deal in the Software without restriction, including\nwithout limitation the rights to use, copy, modify, merge, publish,\ndistribute, sublicense, and/or sell copies of the Software, and to\npermit persons to whom the Software is furnished to do so, subject to\nthe following conditions:\n\nThe above copyright notice and this permission notice shall be\nincluded in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,\nEXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF\nMERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND\nNONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE\nLIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION\nOF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION\nWITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('preflight', 'freeze', 'verify', 'run', 'worker'):
        action.add_argument('--'+name, action='store_true')
    parser.add_argument('--size', choices=tuple(CONTEXT_ROWS))
    parser.add_argument('--stage', choices=('first', 'rebuild'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROUND / 'runtime_assessment').resolve()
    if not output.is_relative_to(ROUND.resolve()):
        parser.error('All new gate outputs must remain within its new round')
    if args.preflight:
        preflight()
    elif args.freeze:
        freeze()
    elif args.verify:
        verify()
        assert_source_only()
        print('Frozen new temporal-block NCA sources verified; no model calls', flush=True)
    elif args.worker:
        if not args.size or not args.stage:
            parser.error('Worker requires size and stage')
        try:
            worker(args.size, args.stage, output)
        except BaseException as error:
            if output.exists():
                write_json(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                    synthetic_only=True, native_arrays_read=False, no_retry=True))
            raise
    elif args.run:
        main(output)
