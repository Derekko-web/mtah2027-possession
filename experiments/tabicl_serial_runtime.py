"""R53: single-thread official cached TabICL, with independent exact reconstruction."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from unittest.mock import patch

import tabicl_cached_runtime_gate as cached

ROOT = cached.ROOT
ROUND = ROOT / 'data/control_tabicl_serial_v1'
PLAN = ROOT / 'experiments/TABICL_SERIAL_RUNTIME_PLAN.json'
SOURCE = cached.ROUND / 'runtime_assessment/large'
digest, write_json = cached.digest, cached.write_json
SERIAL_ENV = {name: '1' for name in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
SERIAL_ENV['PYTHONHASHSEED'] = '0'


def verify():
    cached.verify()
    freeze = ROUND / 'protocol_freeze'
    if PLAN.read_bytes() != (freeze / 'plan.json').read_bytes():
        raise ValueError('Serial execution plan changed')
    for name, expected in json.loads((freeze / 'implementation.json').read_text()).items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen serial dependency changed: ' + name)


def initialize():
    for name, value in SERIAL_ENV.items():
        if os.environ.get(name) != value:
            raise ValueError('Missing fixed execution environment: ' + name)
    cached.original.runtime_paths()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    if torch.__version__ != '2.5.0' or np.__version__ != '2.1.2':
        raise ValueError('Pinned runtime changed')


def worker(stage, output):
    verify()
    initialize()
    import numpy as np
    from tabicl_context_rebuild import operation
    output.mkdir(parents=True, exist_ok=False)
    train, target, query = [np.load(SOURCE / (name + '.npy'))
                            for name in ('train', 'target', 'query')]
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('Network disabled')):
        model = cached.make_classifier(True)
        model.set_params(n_jobs=1)
        _, fit_time = operation(output, 'context_fit', lambda: model.fit(train, target))
        cached.checkpoint_check(model)
        fingerprint = cached.cache_fingerprint(model, 3000, 166)
        actual, predict_time = operation(output, 'predict', lambda: model.predict_proba(query))
        repeated, repeat_time = operation(output, 'repeat', lambda: model.predict_proba(query))
        np.save(output / 'probability.npy', actual)
        np.save(output / 'repeat.npy', repeated)
        write_json(output / 'cache_fingerprint.json', fingerprint)
        np.testing.assert_array_equal(actual, repeated)
        baseline = np.load(SOURCE / 'cached_first.npy')
        parity = cached.compare(baseline, actual)
        if not parity['passed']:
            raise ValueError('Fixed numerical parity to original backend failed')
        if fingerprint != cached.cache_fingerprint(model, 3000, 166):
            raise ValueError('Prediction mutated context cache')
        if stage == 'rebuild':
            parent = output.parent / 'first'
            np.testing.assert_array_equal(actual, np.load(parent / 'probability.npy'))
            if fingerprint != json.loads((parent / 'cache_fingerprint.json').read_text()):
                raise ValueError('Independent serial cache reconstruction differs')
        checks = cached.probability_checks(actual, 512)
        checks.update(repeat_exact=True, original_backend_parity=parity['passed'],
            independent_rebuild_exact=True, cached_context_immutable=True,
            checkpoint_exact=True, peak_within_6GiB=cached.original.peak_bytes() <= cached.LIMIT_BYTES)
        write_json(output / 'summary.json', dict(passed=all(checks.values()), checks=checks,
            parity=parity, times=dict(fit=fit_time, predict=predict_time, repeat=repeat_time),
            peak_bytes=cached.original.peak_bytes(), synthetic_only=True,
            context_shape=[3000, 166], query_shape=[512, 166],
            serial_environment=SERIAL_ENV, implementation_unchanged=True,
            artifact_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()}))


def supervise(stage, output):
    cached.original.runtime_paths()
    import psutil
    start, peak, aborted = time.monotonic(), 0, None
    destination = output / stage
    log, error = output / (stage + '.stdout.log'), output / (stage + '.stderr.log')
    with log.open('wb') as stdout, error.open('wb') as stderr:
        process = subprocess.Popen([str(cached.PYTHON), '-B', str(Path(__file__).resolve()),
            '--worker', stage, '--output', str(destination)], cwd=ROOT,
            env={**os.environ, **SERIAL_ENV}, stdout=stdout, stderr=stderr)
        handle = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss = handle.memory_info().rss
            except psutil.NoSuchProcess:
                break
            peak = max(peak, rss)
            path = destination / 'operation.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            if rss > cached.LIMIT_BYTES:
                aborted = 'rss_over_6GiB'
            elif status.get('active') and time.monotonic() - status['started'] > 300:
                aborted = 'operation_over_300seconds'
            elif time.monotonic() - start > 900:
                aborted = 'worker_over_900seconds'
            if aborted:
                process.terminate()
                break
            time.sleep(.1)
        code = process.wait(timeout=10)
    path = destination / 'summary.json'
    summary = json.loads(path.read_text()) if path.exists() else {}
    record = dict(stage=stage, exit_code=code, aborted=aborted,
        passed=code == 0 and summary.get('passed', False), largest_sampled_rss_bytes=peak,
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
        plan_sha256=digest(PLAN), script_sha256=digest(__file__), synthetic_only=True,
        native_processing_started=False, prior_R49_R51_remain_failed=True))
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
