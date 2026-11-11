"""Training-only football context and isolated official TabICLv2 inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'data/control_tabicl_native_v1'
PLAN = ROOT / 'experiments/TABICL_NATIVE_CONTEXT_PLAN.json'
PUBLIC = np.arange(11185, 18685)
_LIVE_WORKERS = {}
_WORKER_LOCK = threading.Lock()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def context_record(inputs, labels, train, validation):
    """Select only permitted 5Hz live training rows; impute after augmentation."""
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from rich_features import swap_teams
    train, validation = np.asarray(train, int), np.asarray(validation, int)
    if (not len(train) or np.intersect1d(train, validation).size
            or not np.isin(train, PUBLIC).all() or not np.isin(validation, PUBLIC).all()
            or len(np.unique(train)) != len(train)
            or not np.isin(labels[train], ['H', 'A', 'D']).all()):
        raise ValueError('Invalid public training split')
    frames = train[(train % 5 == 0) & np.isin(labels[train], ['H', 'A'])]
    if not len(frames) or 2 * len(frames) > 3000:
        raise ValueError('Invalid fixed-size live context')
    original = inputs.iloc[frames]
    values = pd.concat([original, swap_teams(original)], ignore_index=True)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=False)
    context = imputer.fit_transform(values).astype(np.float32)
    target = (labels[frames] == 'A').astype(np.int64)
    target = np.r_[target, 1 - target]
    if (not np.isfinite(context).all() or context.shape[1] != inputs.shape[1]
            or not np.array_equal(np.bincount(target), [len(frames), len(frames)])):
        raise ValueError('Context imputation or balanced augmentation failed')
    record = dict(split_train=train, validation=validation, context_frames=frames,
                  target=target, weight=np.ones(len(target)), median=imputer.statistics_)
    return context, target, imputer, record


def query_indices(length):
    if length <= 0:
        raise ValueError('Empty query set')
    for start in range(0, length, 512):
        end = min(start + 512, length)
        index = np.arange(start, end)
        if len(index) < 512:
            index = np.r_[index, np.repeat(index[-1], 512 - len(index))]
        yield start, end, index


def worker(job_path, output):
    import tabicl_context_rebuild as rebuild
    import tabicl_serial_runtime as serial
    serial.verify()
    summary = serial.ROUND / 'runtime_assessment/summary.json'
    if not summary.exists() or not json.loads(summary.read_text())['passed']:
        raise ValueError('Independent context reconstruction has not passed')
    serial.initialize()
    from tabicl_cached_runtime_gate import (make_classifier, cache_fingerprint,
                                           checkpoint_check, probability_checks)
    output.mkdir(parents=True, exist_ok=False)
    job = json.loads(job_path.read_text())
    if set(job) - {'context', 'target', 'original', 'swapped', 'seed', 'input_sha256', 'parent'}:
        raise ValueError('Unexpected worker input; query labels are prohibited')
    for name, expected in job['input_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Worker input changed')
    context, target, original, swapped = [np.load(job[key], mmap_mode='r')
                                         for key in ('context', 'target', 'original', 'swapped')]
    if (context.ndim != 2 or context.shape[1] != 166 or len(context) > 3000
            or original.shape != swapped.shape or original.shape[1] != 166
            or not np.isfinite(context).all() or not np.isfinite(original).all()
            or not np.isfinite(swapped).all() or not np.array_equal(np.unique(target), [0, 1])):
        raise ValueError('Native worker schema differs')
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('Network disabled')):
        model = make_classifier(True)
        model.set_params(random_state=int(job['seed']), n_jobs=1)
        _, fit_seconds = rebuild.operation(output, 'context_fit', lambda: model.fit(context, target))
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError('Unexpected H/A class order')
        columns = int(model.ensemble_generator_.n_features_in_)
        before = cache_fingerprint(model, len(context), columns)
        values, timings = [], []
        for view, inputs in (('original', original), ('swapped', swapped)):
            parts = []
            for start, end, index in query_indices(len(inputs)):
                prediction, elapsed = rebuild.operation(output, view + '_' + str(start),
                    lambda index=index, inputs=inputs: model.predict_proba(np.asarray(inputs[index])))
                if not all(probability_checks(prediction, 512).values()):
                    raise ValueError('Invalid H/A probabilities')
                parts.append(prediction[:end-start])
                timings.append(dict(view=view, start=start, end=end, seconds=elapsed))
            values.append(np.concatenate(parts))
        first_index = next(query_indices(len(original)))[2]
        repeated, _ = rebuild.operation(output, 'repeat_first_query',
            lambda: model.predict_proba(np.asarray(original[first_index])))
        np.testing.assert_array_equal(repeated[:min(len(original), 512)], values[0][:512])
        owner = (values[0] + values[1][:, [1, 0]]) / 2.
        if cache_fingerprint(model, len(context), columns) != before:
            raise ValueError('Query inference changed cached training context')
        checkpoint_check(model)
        if job.get('parent'):
            parent = Path(job['parent'])
            np.testing.assert_array_equal(owner, np.load(parent / 'owner.npy'))
            if before != json.loads((parent / 'cache_fingerprint.json').read_text()):
                raise ValueError('Independent context regeneration changed tensors')
        for name, value in (('original_probability', values[0]), ('swapped_probability', values[1]),
                            ('owner', owner)):
            np.save(output / (name + '.npy'), value)
        write_json(output / 'cache_fingerprint.json', before)
        write_json(output / 'recipe.json', dict(parameters={k: str(v) for k, v in model.get_params().items()},
            context_sha256=digest(job['context']), target_sha256=digest(job['target']),
            original_features=166, library_filtered_features=columns,
            feature_keep_mask=model.ensemble_generator_.unique_filter_.features_to_keep_.tolist(),
            reconstruction='Fresh official fit on saved imputed context and targets',
            fitted_object_pickle_used=False))
        write_json(output / 'summary.json', dict(passed=True, seed=int(job['seed']),
            context_rows=len(context), query_rows=len(original), fit_seconds=fit_seconds,
            peak_bytes=rebuild.cached.original.peak_bytes(), query_timings=timings,
            original_D_mass_supplied_to_worker=False, query_labels_supplied=False,
            class_order=['H', 'A'], context_cache_immutable=True, checkpoint_exact=True,
            artifact_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()}))


def supervise(job, output):
    import sys
    try:
        import psutil
    except ImportError:
        sys.path.insert(0, str(ROOT / 'data/control_tabicl_v1/runtime_deps'))
        import psutil
    import subprocess
    import tabicl_runtime_gate as runtime
    from tabicl_serial_runtime import SERIAL_ENV
    start, peak, aborted = time.monotonic(), 0, None
    with (output.parent / 'worker.stdout.log').open('wb') as stdout, (output.parent / 'worker.stderr.log').open('wb') as stderr:
        process = subprocess.Popen([str(runtime.PYTHON), '-B', str(Path(__file__).resolve()),
            '--job', str(job), '--output', str(output)], cwd=ROOT,
            env={**os.environ, **SERIAL_ENV}, stdout=stdout, stderr=stderr)
        handle = psutil.Process(process.pid)
        with _WORKER_LOCK:
            _LIVE_WORKERS[process.pid] = handle
        combined_peak = 0
        while process.poll() is None:
            try:
                rss = handle.memory_info().rss
            except psutil.NoSuchProcess:
                break
            peak = max(peak, rss)
            with _WORKER_LOCK:
                combined = 0
                for peer in _LIVE_WORKERS.values():
                    try:
                        combined += peer.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
            combined_peak = max(combined_peak, combined)
            status = output / 'operation.json'
            current = json.loads(status.read_text()) if status.exists() else {}
            if rss > 6 * 1024**3:
                aborted = 'rss_over_6GiB'
            elif combined > int(9.5 * 1024**3):
                aborted = 'combined_workers_over_9.5GiB'
            elif current.get('active') and time.monotonic() - current['started'] > 300:
                aborted = 'operation_over_300seconds'
            elif time.monotonic() - start > 900:
                aborted = 'worker_over_900seconds'
            if aborted:
                process.terminate()
                break
            time.sleep(.1)
        code = process.wait(timeout=10)
        with _WORKER_LOCK:
            _LIVE_WORKERS.pop(process.pid, None)
    record = dict(exit_code=code, aborted=aborted, peak_sampled_bytes=peak,
                  combined_workers_peak_bytes=combined_peak, elapsed_seconds=time.monotonic() - start)
    write_json(output.parent / 'supervision.json', record)
    if code != 0 or aborted:
        raise RuntimeError('Native worker failed; retained logs: ' + str(output.parent))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    worker(args.job, args.output)
