"""R67 public-API release of unused MPS allocator/framework graph cache.

All model, preprocessing, query, fingerprint, timing, dispatch and supervision
functions are inherited from frozen R66. Only the callback supplied to the
original timed operation gains synchronize+empty_cache before predictions.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'data/control_tabpfn_mps_allocator_flush_v1'
PLAN = ROOT / 'experiments/TABPFN_MPS_ALLOCATOR_FLUSH_PLAN.json'
TEST = ROOT / 'experiments/test_tabpfn_mps_allocator_flush.py'
PREVIOUS = ROOT / 'data/control_tabpfn_mps_balanced_budget_v1'
SOURCE = ROOT / 'experiments/tabpfn_mps_balanced_budget_gate.py'
SOURCE_SHA = '9f5ae761ecf7a9d2748345d5a116f5769893f26610145f169dc477ebe4c32478'
TORCH_REVISION = 'd0f2af8b66b20da4baa50644c0cd3f887a2eae2e'
GIB = 1024**3
SOURCE_HASHES = {
    'torch/mps/__init__.py': 'c9ec34971a58ab7e170b4181dde4307a87d6ef7d5b4dfb87b3c925c85f4f1fa9',
    'torch/csrc/mps/Module.cpp': '9f02b982a2f3a82de8be3f1c45e709c9921cd203e354fedcb17710dbfe26035d',
    'aten/src/ATen/mps/MPSAllocator.mm': 'c715eb3f3d7202075e5b54eb425af8e11940f8088259d0edb4bb7a775ca6df2f',
    'aten/src/ATen/mps/MPSAllocator.h': 'a795c35b2c8bd150c332fa659eabbea5c89345593c4daa6ba0da99c0a5a9ed05',
    'aten/src/ATen/mps/MPSHooks.mm': 'c6c8764211acce1621a88dafab04c2fb95f8320316e9d62e91394d07412539ec',
    'aten/src/ATen/native/mps/OperationUtils.h': 'b0cabc60d1522bebd3b60d3794c74dccd78e3149c32ca13f00198c9a0af54a0f',
    'aten/src/ATen/native/mps/OperationUtils.mm': 'e803190bab2c5716f5c65bfafdca3c53b27a8bfb2c554fdce448a0a7107b3cac',
    'LICENSE': 'bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43',
}
_ORIGINAL_OPERATION = None


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def previous():
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen R66 source changed')
    spec = importlib.util.spec_from_file_location('r67_frozen_r66', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpp_body(text, signature):
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 0
    for i in range(opening, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[opening:i + 1]
    raise ValueError('Incomplete pinned C++ definition: ' + signature)


def source_contract():
    records = json.loads((ROUND / 'source_proof.json').read_text())
    if {r['path'] for r in records} != set(SOURCE_HASHES):
        raise ValueError('Unexpected allocator source proof file set')
    for record in records:
        path = ROUND / 'public_sources' / record['path']
        expected = SOURCE_HASHES[record['path']]
        url = f'https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_REVISION}/{record["path"]}'
        if (record['sha256'] != expected or digest(path) != expected
                or record['revision'] != TORCH_REVISION or record['url'] != url):
            raise ValueError('Pinned Torch source identity differs: ' + record['path'])
    installed = ROOT / 'data/control_tabpfn_mps_loader_repair_v1/runtime_deps/torch/mps/__init__.py'
    if digest(installed) != SOURCE_HASHES['torch/mps/__init__.py']:
        raise ValueError('Installed MPS Python API differs from pinned official source')
    version_file = installed.parents[1] / 'version.py'
    version = {n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(version_file.read_bytes()).body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('__version__', 'git_version')}
    if version != {'__version__': '2.15.0', 'git_version': TORCH_REVISION}:
        raise ValueError('Installed Torch version/official source revision differs')
    tree = ast.parse((ROUND / 'public_sources/torch/mps/__init__.py').read_bytes())
    definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    empty = definitions['empty_cache']
    sync = definitions['synchronize']
    expected_empty = ast.parse('torch._C._mps_emptyCache()', mode='eval').body
    expected_sync = ast.parse('torch._C._mps_deviceSynchronize()', mode='eval').body
    if (len(empty.body) != 2 or not isinstance(empty.body[1], ast.Expr)
            or ast.dump(empty.body[1].value) != ast.dump(expected_empty)):
        raise ValueError('Official empty_cache no longer only delegates to allocator')
    if (len(sync.body) != 2 or not isinstance(sync.body[1], ast.Return)
            or ast.dump(sync.body[1].value) != ast.dump(expected_sync)):
        raise ValueError('Official synchronize implementation differs')
    allocator = (ROUND / 'public_sources/aten/src/ATen/mps/MPSAllocator.mm').read_text()
    cpp = {
        'emptyCache': cpp_body(allocator, 'void MPSHeapAllocatorImpl::emptyCache()'),
        'release_cached_buffers': cpp_body(allocator, 'bool MPSHeapAllocatorImpl::release_cached_buffers()'),
        'release_free_heaps': cpp_body(allocator, 'size_t MPSHeapAllocatorImpl::release_free_heaps('),
        'release_heap': cpp_body(allocator, 'void MPSHeapAllocatorImpl::release_heap('),
        'alloc_buffer_block': cpp_body(allocator, 'BufferBlock* MPSHeapAllocatorImpl::alloc_buffer_block('),
    }
    required = {
        'emptyCache': ['release_cached_buffers();'],
        'release_cached_buffers': ['synchronizeAllMPSStreams(SyncType::COMMIT_AND_WAIT)', 'release_free_heaps('],
        'release_free_heaps': ['heap->free_bytes != heap->total_size', 'block->retainCount() > 1', 'release_heap(pool, heap)'],
        'release_heap': ['TORCH_INTERNAL_ASSERT(!block->in_use)'],
        'alloc_buffer_block': ['release_cached_buffers() && (get_free_buffer(params) || alloc_heap(params))'],
    }
    for name, markers in required.items():
        if any(marker not in cpp[name] for marker in markers):
            raise ValueError('Pinned allocator source contract differs: ' + name)
    hooks = (ROUND / 'public_sources/aten/src/ATen/mps/MPSHooks.mm').read_text()
    module = (ROUND / 'public_sources/torch/csrc/mps/Module.cpp').read_text()
    hook = cpp_body(hooks, 'void MPSHooks::emptyCache()')
    if ('getIMPSAllocator()->emptyCache()' not in hook
            or 'MPSGraphCache::getInstance()->clear()' not in hook
            or 'getMPSHooks().emptyCache()' not in cpp_body(module, 'static PyObject* MPSModule_emptyCache(')):
        raise ValueError('Official C++ API-to-allocator bridge differs')
    graph_source = (ROUND / 'public_sources/aten/src/ATen/native/mps/OperationUtils.h').read_text()
    graph_cache = cpp_body(graph_source, 'struct MPSGraphCache {')
    graph_clear = cpp_body(graph_cache, 'void clear()')
    if any(marker not in graph_clear for marker in ('dispatch_sync_with_rethrow(serialQueue_',
            'delete i.second.cachedGraph_', 'cache_.clear()')):
        raise ValueError('Official framework graph-cache release differs')
    return dict(torch_version='2.15.0', revision=TORCH_REVISION, installed_python_api_exact=True,
        allocator_release_path_releases_only_wholly_free_unretained_heaps=True,
        live_buffer_release_forbidden=True, allocator_already_retries_release_on_oom=True,
        public_api_also_clears_framework_MPSGraph_compilation_cache=True,
        framework_graph_cache_is_not_TabPFN_fitted_preprocessing_or_training_context=True,
        driver_counter_includes_non_allocator_MPSGraph_memory=True,
        no_guarantee_that_flush_prevents_oom=True,
        official_python_definitions={name: dict(lines=[definitions[name].lineno, definitions[name].end_lineno],
            ast_sha256=hashlib.sha256(ast.dump(definitions[name]).encode()).hexdigest())
            for name in ('empty_cache', 'synchronize', 'current_allocated_memory', 'driver_allocated_memory')},
        cpp_definition_sha256={name: hashlib.sha256(body.encode()).hexdigest() for name, body in cpp.items()})


def is_prediction(name):
    return not name.startswith(('fit_', 'fingerprint_'))


def memory_counters(mps):
    result = dict(mps_driver_bytes=int(mps.driver_allocated_memory()),
                  mps_tensor_bytes=int(mps.current_allocated_memory()))
    if min(result.values()) < 0:
        raise ValueError('Invalid MPS memory counter')
    return result


def flushed_callback(callback, mps, emit):
    """No model/context references, RNG operations, deletions or retries occur here."""
    def wrapped():
        before = None
        try:
            mps.synchronize()
            before = memory_counters(mps)
            mps.empty_cache()
            after = memory_counters(mps)
            unchanged = before['mps_tensor_bytes'] == after['mps_tensor_bytes']
            event = dict(status='complete', before=before, after=after,
                live_tensor_bytes_unchanged=unchanged,
                driver_bytes_released=before['mps_driver_bytes'] - after['mps_driver_bytes'],
                synchronization_calls=1, empty_cache_calls=1,
                model_context_or_rng_mutation_apis_called=False,
                official_api_also_clears_framework_MPSGraph_compilation_cache=True,
                driver_reduction_required=False)
        except BaseException as error:
            emit(dict(status='failed', before=before, error=repr(error), callback_called=False))
            raise
        emit(event)
        if not unchanged:
            raise ValueError('Allocator flush changed live MPS tensor bytes')
        return callback()
    return wrapped


def record_flush(output, name, event):
    value = dict(operation=name, **event)
    with (output / 'allocator_flush.jsonl').open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def allocator_flush_operation(output, name, callback):
    if _ORIGINAL_OPERATION is None:
        raise ValueError('Original timed operation has not been bound')
    if not is_prediction(name):
        return _ORIGINAL_OPERATION(output, name, callback)
    # Scientific runtime is already initialized by the unchanged inherited worker.
    # Importing here does not move cleanup outside the original timed callback.
    import torch
    cleaned = flushed_callback(callback, torch.mps, lambda event: record_flush(output, name, event))
    return _ORIGINAL_OPERATION(output, name, cleaned)


def core():
    global _ORIGINAL_OPERATION
    execution = previous().core()
    execution.ROUND = ROUND
    execution.PLAN = PLAN
    execution.__file__ = str(Path(__file__).resolve())
    execution.verify = verify
    _ORIGINAL_OPERATION = execution.operation
    execution.operation = allocator_flush_operation
    if (execution.LIMIT_BYTES, execution.OPERATION_SECONDS, execution.WORKER_SECONDS) != (6 * GIB, 300, 900):
        raise ValueError('Original overall resource limits changed')
    return execution


def immutable_contract():
    if any((ROUND / name).read_bytes() != (PREVIOUS / name).read_bytes()
           for name in ('hardware_probe.json', 'acquisition.json')):
        raise ValueError('R66 hardware/model/dependency identity changed')
    hardware = json.loads((ROUND / 'hardware_probe.json').read_text())
    if (hardware['mps_allocator_limit_bytes'], hardware['reserved_owned_rss_budget_bytes']) != (4 * GIB, 2 * GIB):
        raise ValueError('Original allocator/RSS planning allocation changed')
    return hardware


def preflight():
    started = time.monotonic()
    compile(Path(__file__).read_bytes(), str(Path(__file__)), 'exec')
    compile(TEST.read_bytes(), str(TEST), 'exec')
    prior = previous()
    prior.verify()
    proof = source_contract()
    hardware = immutable_contract()
    execution = core()
    names = ('worker', 'main', 'supervise', 'canonical_predict', 'checkpoint_fingerprint',
             'synthetic_data', 'fixed_environment', 'runtime_paths', 'MemoryMonitor', 'NativeDispatchGuard')
    for name in names:
        value = getattr(execution, name)
        code = value.__init__.__code__ if isinstance(value, type) else value.__code__
        if Path(code.co_filename).resolve() != ROOT / 'experiments/tabpfn_mps_runtime_gate.py':
            raise ValueError('Original runtime/math function changed: ' + name)
    if Path(_ORIGINAL_OPERATION.__code__.co_filename).resolve() != ROOT / 'experiments/tabpfn_mps_runtime_gate.py':
        raise ValueError('Original operation timer/monitor/dispatch was replaced')
    if Path(execution.preprocessing_fingerprint.__code__.co_filename).resolve() != ROOT / 'experiments/tabpfn_mps_validation_init_gate.py':
        raise ValueError('Full R63 fingerprint boundary changed')
    if execution.operation is not allocator_flush_operation:
        raise ValueError('Prediction-only allocator hook not bound')
    if execution.fixed_environment()['TABPFN_MPS_MEMORY_FRACTION'] != hardware['tabpfn_mps_memory_fraction']:
        raise ValueError('Pre-import allocation fraction changed')
    if any(name in sys.modules for name in ('torch', 'numpy', 'scipy', 'sklearn', 'tabpfn', 'joblib')):
        raise ValueError('Scientific import occurred in source-only preflight')
    save(ROUND / 'source_preflight.json', dict(passed=True, seconds=time.monotonic() - started,
        implementation_sha256=digest(__file__), plan_sha256=digest(PLAN), test_sha256=digest(TEST),
        source_proof_sha256=digest(ROUND / 'source_proof.json'), source_contract=proof,
        original_function_paths_verified=list(names), original_timed_operation_retained=True,
        full_raw_cache_checkpoint_fingerprints_retained=True,
        allocator_limit_bytes=4 * GIB, combined_limit_bytes=6 * GIB,
        previous_R66_remains_failed=True, scientific_imports=False, model_calls=False))
    print('R67 source-only allocator preflight passed; no model call', flush=True)


def freeze():
    review = json.loads((ROUND / 'independent_review.json').read_text())
    pre = json.loads((ROUND / 'source_preflight.json').read_text())
    for record in (review, pre):
        if (record['implementation_sha256'] != digest(__file__) or record['plan_sha256'] != digest(PLAN)
                or record['test_sha256'] != digest(TEST)
                or record['source_proof_sha256'] != digest(ROUND / 'source_proof.json')):
            raise ValueError('Review/preflight must bind final wrapper/plan/tests/source proof')
    if not review['approved_for_freeze'] or not pre['passed']:
        raise ValueError('ROOT review and source-only preflight required')
    previous().verify()
    source_contract()
    immutable_contract()
    checks = {str(Path(__file__).relative_to(ROOT)): digest(__file__),
        str(PLAN.relative_to(ROOT)): digest(PLAN), str(TEST.relative_to(ROOT)): digest(TEST)}
    for path in sorted(ROUND.rglob('*')):
        if 'protocol_freeze' not in path.parts and 'runtime_assessment' not in path.parts and path.is_file():
            if path.suffix == '.pyc' or '__pycache__' in path.parts:
                raise ValueError('Unexpected generated bytecode')
            checks[str(path.relative_to(ROOT))] = digest(path)
    for path in sorted((PREVIOUS / 'runtime_assessment').rglob('*')):
        if path.is_file():
            checks[str(path.relative_to(ROOT))] = digest(path)
    for name in ('protocol_freeze/implementation.json', 'protocol_freeze/freeze.json', 'runtime_gate.log'):
        path = PREVIOUS / name
        checks[str(path.relative_to(ROOT))] = digest(path)
    report = ROOT / 'experiments/TABPFN_MPS_BALANCED_BUDGET_REPORT.md'
    checks[str(report.relative_to(ROOT))] = digest(report)
    folder = ROUND / 'protocol_freeze'
    folder.mkdir(exist_ok=False)
    (folder / 'plan.json').write_bytes(PLAN.read_bytes())
    save(folder / 'implementation.json', checks)
    save(folder / 'freeze.json', dict(implementation_sha256=digest(__file__), plan_sha256=digest(PLAN),
        test_sha256=digest(TEST), source_proof_sha256=digest(ROUND / 'source_proof.json'),
        bound_files=len(checks), only_change='pre-prediction sync+empty_cache inside original timed callback',
        original_math_and_full_raw_checks=True, previous_R66_remains_failed=True,
        model_calls_started=False, native_inputs_read=False))
    print('R67 frozen after ROOT review; no model call started', flush=True)


def verify():
    folder = ROUND / 'protocol_freeze'
    if PLAN.read_bytes() != (folder / 'plan.json').read_bytes():
        raise ValueError('Frozen R67 plan changed')
    for path, expected in json.loads((folder / 'implementation.json').read_text()).items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen R67 input changed: ' + path)
    previous().verify()
    source_contract()
    immutable_contract()


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
        parser.error('Artifacts must remain in R67')
    if args.preflight:
        preflight()
    elif args.freeze:
        freeze()
    elif args.verify:
        verify()
        print('R67 and all prior frozen evidence verified', flush=True)
    else:
        execution = core()
        if args.worker:
            if not args.size or not args.stage:
                parser.error('Worker requires size and stage')
            try:
                execution.worker(args.size, args.stage, output)
            except BaseException as error:
                if output.exists():
                    save(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                        synthetic_only=True, competition_inputs_read=False, prior_failures_retained=True))
                raise
        else:
            execution.main(output)
