"""Continue allocation proof, full inference, audits and authorized local selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import tabpfn_full_match_native_v3 as native

ROOT,ROUND=native.ROOT,native.ROUND
PYTHON=ROOT/'data/control_isolated_python_v1/head_runtime/bin/python'
DRIVER='experiments/build_tabpfn_full_candidate_v3.py'
FINALIZER='experiments/finalize_tabpfn_full_candidate.py'


def stage(label,script,*args,source_only=False):
    print('Starting',label,flush=True)
    command=[str(PYTHON),'-B']
    if source_only:command+=['-I','-S']
    command+=[script,*map(str,args)]
    with (ROUND/(label+'.log')).open('x') as handle:
        subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    print('Completed',label,flush=True)


def wait_for_gate(pid):
    gate=native.memory_gate.ROUND
    print('Waiting for exact full-context and five-main-fold allocation proof',flush=True)
    while not (gate/'gate_passed.json').exists():
        if (gate/'gate_failure.json').exists():raise ValueError('Allocation or saved-main-score proof failed')
        try:os.kill(pid,0)
        except ProcessLookupError:raise ValueError('Allocation controller stopped without a passed result')
        time.sleep(5)
    native.verify_memory_gate()


def freeze_production():
    authorization=native.read(ROUND/'completion_authorization.json')
    for name,sha in authorization['source_sha256'].items():
        if native.digest(ROOT/name)!=sha:raise ValueError('Completion source changed before admission')
    if native.digest(ROOT/'experiments/SELECTED_MODEL.md')!=authorization['previous_selected_document_sha256']:
        raise ValueError('Selection changed during the allocation experiment')
    for item in ('context.npy','target.npy'):
        if (ROUND/'training'/item).read_bytes()!=(native.memory_gate.ROUND/'fixed5_first'/item).read_bytes():
            raise ValueError('Production full-context payload differs from passed allocation proof')
    if (ROUND/'hardware_probe.json').read_bytes()!=(native.memory_gate.ROUND/'fixed5_first/hardware_probe.json').read_bytes():
        raise ValueError('Production allocation differs from tested allocation')
    policy=dict(user_authorization=authorization['user_authorization'],
        previous_selected_document_sha256=authorization['previous_selected_document_sha256'],
        previous_candidate_sha256=authorization['previous_candidate_sha256'],
        source_sha256=authorization['source_sha256'],
        required='Complete40 native workers, canonical full queries, exact fitted state and repeats, unchanged main0.777142857, two identical full CSV exports, official validation, preserved D/X and original candidate; then local selection',
        no_remote_main_or_team_default_change=True)
    smoke=native.read(ROUND/'full_decoder_smoke.json')
    if not smoke['passed'] or not smoke['reference_decode_exact'] or not smoke['unsupported_owner_keeps_baseline_exact']:
        raise ValueError('Full-size decoder compatibility check incomplete')
    native.write_json(ROUND/'finalization_plan.json',policy)
    plan=native.read(native.PLAN)
    (ROUND/'prepared_plan_before_review.json').write_bytes(native.PLAN.read_bytes())
    plan.update(round='round81_fixed_allocation_full_match',purpose='Full-match construction of the unchanged0.777142857 seed42 model after actual allocation and main replay proof',
        gate=policy['required'],hypothesis='No model or decoder selection; same numerical learner with allocation-only repair',
        failure='Preserve any failed native attempt; no partial-model rescue, parameter tuning or silent retries',
        full_build='40 fresh fits of identical2352-row public context, original512-row batch schedule, two independently loaded full CSV exports; entire full query inference is not duplicated',
        time_extension_only=False,automatic_promotion=True,conditional_local_selection_authorized=True,
        memory_policy=dict(gpu_bytes=5*1024**3,reserved_host_bytes=1024**3,combined_limit_bytes=6*1024**3),
        interrupted_worker_policy='At most two fresh identical-input retries only for captured host-suspension clock failures; numerical/memory/source failures never retried; all failed attempts retained',
        source_sha256=native.source_identity(),finalization_policy_sha256=native.digest(ROUND/'finalization_plan.json'))
    bindings=[ROUND/'completion_authorization.json',ROUND/'finalization_plan.json',ROUND/'full_decoder_smoke.json',
        native.memory_gate.ROUND/'gate_passed.json',native.memory_gate.ROUND/'baseline_reproduced.json',
        native.memory_gate.ROUND/'protocol_freeze/freeze.json',native.memory_gate.PLAN,
        ROOT/'experiments/run_tabpfn_full_completion.py',ROOT/FINALIZER]
    bindings.extend(ROOT/name for name in native.memory_gate.SOURCES)
    for path in bindings:plan['input_sha256'][str(path.relative_to(ROOT))]=native.digest(path)
    native.PLAN.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    review=dict(approved_for_freeze=True,review_kind='root-reviewed fixed source and prospective conditional selection policy; no independent agent claimed',
        source_sha256=native.source_identity(),plan_sha256=native.digest(native.PLAN),
        passed_actual_memory_and_main_replay_gate=True,no_numerical_model_change=True,
        conditional_local_selection_explicitly_authorized=True,old_candidate_preserved=True,
        finalization_policy_source_sha256=authorization['source_sha256'],unresolved_findings=0)
    native.write_json(ROUND/'production_root_review.json',review)
    stage('production_freeze','experiments/tabpfn_full_match_native_v3.py','--freeze',source_only=True)
    for line in subprocess.check_output(['ps','-axo','pid=,args='],text=True).splitlines():
        fields=line.strip().split(maxsplit=1)
        program=fields[1].split(maxsplit=1)[0] if len(fields)==2 else ''
        if len(fields)==2 and Path(program).name.lower().startswith('python') and 'tabpfn' in fields[1] and '--worker' in fields[1]:
            raise ValueError('Another native model worker is active')
    native.write_json(ROUND/'resource_admission.json',dict(approved_for_extended_evaluation=True,
        other_model_workers_stopped=True,freeze_sha256=native.digest(ROUND/'protocol_freeze/freeze.json'),
        combined_limit_bytes=6*1024**3,gpu_limit_bytes=5*1024**3,serial_native_workers=True,
        operation_seconds=900,worker_seconds=3600,admitted_utc_ns=time.time_ns()))


def main(gate_pid):
    with (ROUND/'workflow_started.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(),gate_pid=gate_pid,started_utc_ns=time.time_ns()),handle)
    wait_for_gate(gate_pid)
    freeze_production()
    stage('full_prediction',DRIVER,'--predict')
    for name in ('export1','export2'):stage(name,DRIVER,'--export',ROUND/'candidates'/name)
    stage('final_audit_and_selection',FINALIZER)
    if (ROUND/'promotion_complete.json').exists():print('Verified full-match fixed TabPFN is now selected; previous model preserved',flush=True)
    else:print('Full candidate completed; selection deferred by stop marker',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gate-pid',type=int,required=True);args=parser.parse_args()
    try:main(args.gate_pid)
    except BaseException as error:
        path=ROUND/'workflow_failure.json'
        if not path.exists():native.write_json(path,dict(error=repr(error),traceback=traceback.format_exc(),selected_model_may_only_change_after_final_audit=True))
        raise
