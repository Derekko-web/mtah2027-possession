"""Resume preserved production jobs after loss of their external controller.

No numerical source, frozen plan, inputs or successful worker is modified.
An orphan worker cannot supply the missing external resource supervision;
retain its complete attempt and rerun that job under the original supervisor.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabpfn_full_match_native_v3 as native
import run_tabpfn_full_completion as workflow
from tabpfn_host_interruption import is_host_sleep_interruption


def main():
    root, round_path = native.ROOT, native.ROUND
    native.verify()
    native.require_admission()
    policy = native.read(round_path / 'finalization_plan.json')
    for name, sha in policy['source_sha256'].items():
        if native.digest(root / name) != sha:
            raise ValueError('Frozen completion source changed: ' + name)
    plan = native.read(native.PLAN)
    recovery = round_path / 'controller_loss_recovery'
    recovery.mkdir(exist_ok=False)
    native.write_json(recovery / 'started.json', dict(
        pid=os.getpid(), utc_ns=time.time_ns(), source_sha256=native.digest(Path(__file__)),
        reason='Original workflow and prediction controller absent; native worker orphaned',
        original_worker_outputs_never_accepted_without_external_supervision=True))
    keep_awake = subprocess.Popen(['/usr/bin/caffeinate', '-is'])
    try:
        for key in sorted(plan['jobs'], key=lambda k: (k.split('/')[0], int(k.split('/block')[1]))):
            folder = round_path / 'jobs' / key
            supervision = folder / 'supervision.json'
            if supervision.exists():
                record = native.read(supervision)
                if not record['passed'] or record['exit_code'] != 0:
                    raise ValueError('Existing supervised failure: ' + key)
                if record['summary_sha256'] != native.digest(folder / 'worker/summary.json'):
                    raise ValueError('Saved successful summary changed: ' + key)
                for kind in ('stdout', 'stderr'):
                    if record[kind + '_sha256'] != native.digest(folder / ('worker.' + kind + '.log')):
                        raise ValueError('Saved successful log changed: ' + key)
                print('Preserved passed worker', key, flush=True)
                continue
            if (folder / 'worker').exists():
                processes = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,args='], text=True)
                for line in processes.splitlines():
                    fields = line.strip().split(maxsplit=2)
                    if len(fields) != 3:
                        continue
                    pid, parent, command = fields
                    if (Path(command.split()[0]).name.startswith('python')
                            and '--worker' in command and str(folder / 'job.json') in command):
                        if parent != '1':
                            raise ValueError('Worker still has a controller; refusing duplicate')
                        native.write_json(recovery / 'orphan_process.json', dict(
                            pid=int(pid), ppid=int(parent), command=command, stopped_utc_ns=time.time_ns()))
                        os.killpg(int(pid), signal.SIGTERM)
                        for _ in range(100):
                            try:
                                os.kill(int(pid), 0)
                            except ProcessLookupError:
                                break
                            time.sleep(.1)
                        else:
                            raise ValueError('Orphan did not stop')
                archive = recovery / key
                archive.mkdir(parents=True)
                shutil.copy2(folder / 'job.json', archive / 'job.json')
                for name in ('worker', 'worker.stdout.log', 'worker.stderr.log'):
                    if (folder / name).exists():
                        (folder / name).rename(archive / name)
                native.write_json(archive / 'controller_interruption.json', dict(
                    reason='External supervisor lost; this attempt is excluded from final outputs',
                    not_a_model_or_memory_failure=True, input_job_sha256=native.digest(folder / 'job.json')))
            print('Original supervised fresh fit', key, flush=True)
            for attempt in range(1, 4):
                try:
                    native.supervise(folder / 'job.json', folder / 'worker')
                    break
                except native.NativeMethodFailure:
                    record = native.read(supervision)
                    error = (folder / 'worker.stderr.log').read_text()
                    if attempt == 3 or not is_host_sleep_interruption(record, error):
                        raise
                    archive = folder / 'interrupted_attempts' / ('attempt' + str(attempt))
                    archive.mkdir(parents=True, exist_ok=False)
                    shutil.copy2(folder / 'job.json', archive / 'job.json')
                    for name in ('worker', 'worker.stdout.log', 'worker.stderr.log', 'supervision.json'):
                        (folder / name).rename(archive / name)
        native.verify()
        native.write_json(round_path / 'production_prediction_complete.json', dict(
            passed=True, seed=42, jobs=len(plan['jobs']), query_rows=plan['query_rows'],
            all_fresh_fits=True, canonical512_queries=True,
            controller_loss_recovery=str(recovery.relative_to(round_path)),
            interrupted_attempts=[str(p.relative_to(round_path)) for p in (round_path / 'jobs').glob('*/*/interrupted_attempts/attempt*')],
            supervision_sha256={key: native.digest(round_path / 'jobs' / key / 'supervision.json') for key in plan['jobs']}))
        for name in ('export1', 'export2'):
            workflow.stage(name, workflow.DRIVER, '--export', round_path / 'candidates' / name)
        workflow.stage('final_audit_and_selection', workflow.FINALIZER)
        native.write_json(recovery / 'complete.json', dict(passed=True, utc_ns=time.time_ns()))
    finally:
        keep_awake.terminate()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        native.write_json(native.ROUND / 'controller_recovery_failure.json', dict(
            error=repr(error), traceback=traceback.format_exc()))
        raise
