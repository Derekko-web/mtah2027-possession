"""R51: regenerate the official cached estimator from saved context, not pickle.

The model has no fitted neural weights. Repeating its public fit(context, y)
reconstructs preprocessing and caches through the same supported execution path.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
from unittest.mock import patch

import tabicl_cached_runtime_gate as cached

ROOT = cached.ROOT
ROUND = ROOT / 'data/control_tabicl_context_rebuild_v1'
PLAN = ROOT / 'experiments/TABICL_CONTEXT_REBUILD_PLAN.json'
SOURCE = cached.ROUND / 'runtime_assessment/large'
digest, write_json = cached.digest, cached.write_json


def verify():
    cached.verify()
    freeze = ROUND / 'protocol_freeze'
    if PLAN.read_bytes() != (freeze / 'plan.json').read_bytes():
        raise ValueError('Context reconstruction plan changed')
    for name, expected in json.loads((freeze / 'implementation.json').read_text()).items():
        if digest(ROOT / name) != expected:
            raise ValueError('Context reconstruction dependency changed: ' + name)


def initialize():
    cached.original.runtime_paths()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if torch.__version__ != '2.5.0' or np.__version__ != '2.1.2':
        raise ValueError('Pinned runtime differs')


def operation(output, name, callback):
    status = output / 'operation.json'
    temporary = output / 'operation.tmp'
    write_json(temporary, dict(name=name, started=time.monotonic(), active=True))
    temporary.replace(status)
    start = time.monotonic()
    print('Start', name, flush=True)
    value = callback()
    elapsed = time.monotonic() - start
    write_json(temporary, dict(name=name, started=start, active=False, elapsed=elapsed))
    temporary.replace(status)
    if elapsed > 300:
        raise ValueError('Operation exceeded 300 seconds: ' + name)
    print('Complete', name, round(elapsed, 3), 'seconds', flush=True)
    return value, elapsed


def worker(stage, output):
    verify()
    initialize()
    import numpy as np
    output.mkdir(parents=True, exist_ok=False)
    train, target, query = [np.load(SOURCE / (name + '.npy'))
                            for name in ('train', 'target', 'query')]
    if train.shape != (3000, 166) or query.shape != (512, 166):
        raise ValueError('Frozen workload differs')
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('Network disabled')):
        estimator = cached.make_classifier(True)
        _, fit_time = operation(output, 'fresh_context_fit', lambda: estimator.fit(train, target))
        fingerprint = cached.cache_fingerprint(estimator, 3000, 166)
        if fingerprint != json.loads((SOURCE / 'cache_fingerprint.json').read_text()):
            raise ValueError('Context reconstruction changed cached tensors')
        actual, predict_time = operation(output, 'predict', lambda: estimator.predict_proba(query))
        repeated, repeat_time = operation(output, 'repeat', lambda: estimator.predict_proba(query))
        np.testing.assert_array_equal(actual, repeated)
        np.testing.assert_array_equal(actual, np.load(SOURCE / 'cached_first.npy'))
        if stage == 'rebuild':
            np.testing.assert_array_equal(actual, np.load(output.parent / 'first/probability.npy'))
        if fingerprint != cached.cache_fingerprint(estimator, 3000, 166):
            raise ValueError('Prediction changed context caches')
        cached.checkpoint_check(estimator)
        np.save(output / 'probability.npy', actual)
        write_json(output / 'cache_fingerprint.json', fingerprint)
        checks = cached.probability_checks(actual, 512)
        checks.update(cached_tensors_exact=True, original_R49_probability_exact=True,
                      independent_context_reconstruction_exact=True,
                      checkpoint_tensors_exact=True,
                      peak_within_6GiB=cached.original.peak_bytes() <= cached.LIMIT_BYTES)
        write_json(output / 'summary.json', dict(passed=all(checks.values()), checks=checks,
            times=dict(fit=fit_time, predict=predict_time, repeat=repeat_time),
            peak_bytes=cached.original.peak_bytes(), model_implementation_unchanged=True,
            context_shape=[3000, 166], query_shape=[512, 166],
            reconstructed_from='Saved train.npy, target.npy and fixed public estimator parameters',
            pickle_regeneration_used=False, synthetic_only=True, native_processing=False,
            artifact_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()}))


def supervise(stage, output):
    initialize()
    import psutil
    log, error = output / (stage + '.stdout.log'), output / (stage + '.stderr.log')
    destination = output / stage
    start = time.monotonic()
    aborted, peak = None, 0
    with log.open('wb') as stdout, error.open('wb') as stderr:
        process = subprocess.Popen([str(cached.PYTHON), '-B', str(Path(__file__).resolve()),
            '--worker', stage, '--output', str(destination)], cwd=ROOT, stdout=stdout, stderr=stderr)
        handle = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss = handle.memory_info().rss
            except psutil.NoSuchProcess:
                break
            peak = max(peak, rss)
            status = destination / 'operation.json'
            record = json.loads(status.read_text()) if status.exists() else {}
            if rss > cached.LIMIT_BYTES:
                aborted = dict(reason='rss_over_6GiB', observed_bytes=rss)
            elif record.get('active') and time.monotonic() - record['started'] > 300:
                aborted = dict(reason='operation_over_300seconds', operation=record['name'])
            elif time.monotonic() - start > 900:
                aborted = dict(reason='worker_over_900seconds')
            if aborted:
                process.terminate()
                break
            time.sleep(.1)
        code = process.wait(timeout=10)
    path = destination / 'summary.json'
    summary = json.loads(path.read_text()) if path.exists() else {}
    record = dict(stage=stage, exit_code=code, aborted=aborted,
                  largest_sampled_rss_bytes=peak, passed=code == 0 and summary.get('passed', False),
                  elapsed_wall_seconds=time.monotonic() - start,
                  summary_sha256=digest(path) if path.exists() else None,
                  stdout_sha256=digest(log), stderr_sha256=digest(error))
    write_json(output / (stage + '.supervision.json'), record)
    print(stage, 'passed', record['passed'], 'peak GiB', round(peak / 1024**3, 3), flush=True)
    return record


def main(output):
    verify()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for stage in ('first', 'rebuild'):
        records.append(supervise(stage, output))
        if not records[-1]['passed']:
            break
    passed = len(records) == 2 and all(r['passed'] for r in records)
    write_json(output / 'summary.json', dict(passed=passed, records=records,
        plan_sha256=digest(PLAN), script_sha256=digest(__file__),
        synthetic_only=True, native_processing_started=False, prior_R49_remains_failed=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('first', 'rebuild'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
    else:
        main(args.output or ROUND / 'runtime_assessment')
