"""Independently audit the full candidate and conditionally update local selection."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import tabpfn_full_match_native_v3 as native
from tabpfn_host_interruption import is_host_sleep_interruption

ROOT,ROUND=native.ROOT,native.ROUND


def require(condition,message):
    if not condition:raise ValueError(message)


def audit():
    native.verify();gate=native.verify_memory_gate()
    policy=native.read(ROUND/'finalization_plan.json')
    for name,sha in policy['source_sha256'].items():
        require(native.digest(ROOT/name)==sha,'Finalization source changed: '+name)
    require(gate['main_corrected_labels_exact'] and abs(gate['main_spellF1']-.7771428571428571)<1e-12,'Saved main result not preserved')
    complete=native.read(ROUND/'production_prediction_complete.json');plan=native.read(native.PLAN)
    require(complete['passed'] and complete['jobs']==40 and len(plan['jobs'])==40,'Incomplete full native cohort')
    import numpy as np
    import pandas as pd
    sys.path.insert(0,str(ROOT))
    from games import GAME,half_video_blocks,source_frames
    from evaluation import possession_metrics as pm
    n=source_frames();expected=np.zeros(n,bool)
    for _,start,end in half_video_blocks(GAME):expected[start:end+1]=True
    seen=np.zeros(n,bool);workers=[];interruptions=[]
    for key,entry in plan['jobs'].items():
        folder=ROUND/'jobs'/key;worker=folder/'worker'
        native.validate_job(native.read(folder/'job.json'))
        summary=native.read(worker/'summary.json');s=native.read(folder/'supervision.json')
        require(s['passed'] and s['exit_code']==0 and s['aborted'] is None and summary['passed'],'Failed full worker')
        require(s['summary_sha256']==native.digest(worker/'summary.json'),'Worker summary changed')
        require(s['elapsed']==native.elapsed(s['started'],s['ended']) and native.within(s['elapsed'],3600),'Full worker clock evidence invalid')
        require(s['peak_owned_rss_plus_full_driver_bytes']<=6*1024**3,'Full worker exceeded combined6GiB')
        require(all(summary.get(k) is True for k in ('complete_raw_preprocessing_immutable','checkpoint_exact',
            'resolved_model_configuration_immutable','repeat_byte_exact','no_network_attempts','native_wrapper_exclusive')),'Full worker integrity conjunction incomplete')
        artifacts={p.name for p in worker.iterdir() if p.is_file()}-{'summary.json','memory.json','memory.tmp'}
        require(set(summary['artifact_sha256'])==artifacts,'Incomplete full worker artifact closure')
        for name,sha in summary['artifact_sha256'].items():
            require(native.digest(worker/name)==sha,'Worker artifact changed: '+name)
        for kind in ('stdout','stderr'):
            require(s[kind+'_sha256']==native.digest(folder/('worker.'+kind+'.log')),'Worker log changed')
        native.validate_operations(worker)
        operations=[json.loads(line) for line in (worker/'operations.jsonl').read_text().splitlines()]
        predictions=[op for op in operations if op['name'].startswith(('original_','swapped_','repeat_first_'))]
        flushes=[json.loads(line) for line in (worker/'allocator_flush.jsonl').read_text().splitlines()]
        require([op['name'] for op in predictions]==[f['operation'] for f in flushes],'Canonical prediction flush schedule differs')
        require(all(op['native_wrapper_calls']==op['native_dispatches']>0 for op in predictions),'Non-native prediction dispatch')
        require(all(f['status']=='complete' and f['live_tensor_bytes_unchanged'] and not f['model_context_or_rng_mutation_apis_called'] for f in flushes),'Allocation flush changed fitted state')
        for name in ('checkpoint_fingerprint.json','preprocessing_fingerprint.json'):
            require((worker/name).read_bytes()==(native.memory_gate.ROUND/'fixed5_first/worker'/name).read_bytes(),'Full fitted state differs from memory gate')
        start,end=entry['start'],entry['end'];frames=np.load(folder/'query_frames.npy',allow_pickle=False)
        np.testing.assert_array_equal(frames,np.arange(start,end));require(not seen[start:end].any(),'Duplicate full query coverage')
        seen[start:end]=True
        p0,p1,owner=[np.load(worker/(name+'.npy'),allow_pickle=False) for name in ('original_probability','swapped_probability','owner')]
        for value in (p0,p1,owner):
            require(value.shape==(end-start,2) and np.isfinite(value).all() and (value>=0).all()
                and (value<=1).all() and np.max(np.abs(value.sum(1)-1))<1e-6,'Invalid full probability matrix')
        np.testing.assert_array_equal(owner,(p0+p1[:,::-1])/2)
        np.testing.assert_array_equal(np.load(worker/'repeat_first.npy',allow_pickle=False),p0[:512])
        workers.append(dict(job=key,peak_combined_bytes=s['peak_owned_rss_plus_full_driver_bytes'],worker_utc_seconds=s['elapsed']['utc_ns']/1e9))
        for attempt in folder.glob('interrupted_attempts/attempt*'):
            interrupted=native.read(attempt/'supervision.json');error=(attempt/'worker.stderr.log').read_text()
            require(is_host_sleep_interruption(interrupted,error),'An ineligible failed attempt was retried')
            require((attempt/'job.json').read_bytes()==(folder/'job.json').read_bytes(),'Interrupted refit inputs changed')
            for kind in ('stdout','stderr'):
                require(interrupted[kind+'_sha256']==native.digest(attempt/('worker.'+kind+'.log')),'Interrupted attempt log changed')
            interruptions.append(str(attempt.relative_to(ROUND)))
    np.testing.assert_array_equal(seen,expected)
    with np.load(ROUND/'training/fitting.npz',allow_pickle=False) as record:
        centers=record['context_frames'];train=record['split_train'];targets=record['target']
        np.testing.assert_array_equal(train,np.arange(11185,18685))
        require((centers%5==0).all() and (centers-79>=11185).all() and (centers+79<18685).all(),'Invalid complete public context footprint')
        truth=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
        require(np.isin(truth[centers],['H','A']).all(),'Non-live context target')
        y=(truth[centers]=='A').astype(np.int64)
        np.testing.assert_array_equal(targets,np.r_[y,1-y])
        require(len(targets)==2352,'Full context changed')
    reference=pd.read_csv(ROOT/'data/candidate_motion_dead_trim/candidate.csv')
    first=ROUND/'candidates/export1';second=ROUND/'candidates/export2'
    for name in ('candidate.csv','owner_probability.npy','restored_events.json'):
        require((first/name).read_bytes()==(second/name).read_bytes(),'Separate full exports differ: '+name)
    for output in (first,second):
        manifest=native.read(output/'manifest.json');require(manifest['passed'],'Full export incomplete')
        for name,sha in manifest['cache_sha256'].items():require(native.digest(output/name)==sha,'Full export artifact changed')
        table=pd.read_csv(output/'candidate.csv');reason=pm.validate_submission(table,n);require(not reason,str(reason))
        for label in ('D','X'):np.testing.assert_array_equal(table.label.to_numpy()==label,reference.label.to_numpy()==label)
        require(np.isin(table.label,['H','A','D','X']).all(),'Invalid full candidate label')
        result=subprocess.run([sys.executable,'-B',str(ROOT/'evaluation/validate_submission.py'),str(output/'candidate.csv')],cwd=ROOT,text=True,capture_output=True)
        (ROUND/('official_'+output.name+'_validation.log')).write_text(result.stdout+result.stderr)
        require(result.returncode==0,'Official full CSV validator rejected candidate')
    result=dict(passed=True,full_frame_rows=n,official_half_queries=int(seen.sum()),native_workers=len(workers),
        full_public_context_rows=2352,main_spellF1=gate['main_spellF1'],main_labels_and_native_outputs_preserved=True,
        both_full_exports_byte_exact=True,official_validator_passed_twice=True,retained_D_X_exact=True,
        all_completed_native_clocks_integrity_and6GiB_bounds_passed=True,workers=workers,
        discarded_host_suspension_attempts=interruptions,other_failed_attempts_not_retried=True,
        candidate_sha256=native.digest(first/'candidate.csv'),source_sha256=policy['source_sha256'])
    native.write_json(ROUND/'final_audit.json',result)
    return result,policy,first


def promote():
    result,policy,source=audit()
    if (ROUND/'STOP_AFTER_BUILD').exists():
        native.write_json(ROUND/'ready_without_selection.json',dict(passed=True,reason='Selection deferred by local stop marker'))
        return
    selected=ROOT/'experiments/SELECTED_MODEL.md'
    require(native.digest(selected)==policy['previous_selected_document_sha256'],'Current selection changed during the build')
    require(not (ROOT/'experiments/SELECTED_MODEL.json').exists(),'Another machine-readable selection already exists')
    require(native.digest(ROOT/'data/candidate_motion_dead_trim/candidate.csv')==policy['previous_candidate_sha256'],'Previous candidate changed')
    output=ROOT/'data/candidate_tabpfn_short_control_v1';output.mkdir(exist_ok=False)
    for name in ('candidate.csv','owner_probability.npy','restored_events.json'):
        shutil.copy2(source/name,output/name)
    for name in ('context.npy','target.npy','fitting.npz','provenance.json'):
        shutil.copy2(ROUND/'training'/name,output/name)
    baseline=ROOT/'data/candidate_motion_dead_trim'
    for p in baseline.iterdir():
        if p.suffix=='.ubj' or p.name.endswith('_probabilities.npy'):shutil.copy2(p,output/p.name)
    shutil.copy2(baseline/'manifest.json',output/'baseline_manifest.json')
    shutil.copy2(source/'manifest.json',output/'full_export_manifest.json')
    shutil.copy2(ROUND/'final_audit.json',output/'final_audit.json')
    shutil.copy2(native.memory_gate.ROUND/'gate_passed.json',output/'allocation_gate.json')
    model=native.native_core().MODEL;shutil.copy2(model,output/'pretrained_tabpfn_v2.ckpt')
    require(native.digest(model)==native.digest(output/'pretrained_tabpfn_v2.ckpt'),'Pretrained weights copy differs')
    require(native.digest(output/'candidate.csv')==result['candidate_sha256'],'Packaged full CSV differs')
    native.write_json(output/'manifest.json',dict(model='Fixed TabPFN seed42 ownership with preserved motion D/X and supported short controls',
        seed=42,csv_sha256=result['candidate_sha256'],full_frames=result['full_frame_rows'],public_OOF_spellF1=result['main_spellF1'],
        full_match_private_accuracy_unknown=True,previous_csv_sha256=policy['previous_candidate_sha256'],
        production_round=str(ROUND.relative_to(ROOT)),production_plan_sha256=native.digest(native.PLAN),
        source_sha256=policy['source_sha256'],allocation_change_only=True,combined_limit_bytes=6*1024**3,
        full_query_inference_repeated_twice=False,independent_full_csv_exports=True,
        cache_sha256={p.name:native.digest(p) for p in output.iterdir() if p.is_file()}))
    native.verify()
    backup=ROOT/'experiments/SELECTED_MODEL_BEFORE_TABPFN_20261102.md'
    with backup.open('xb') as handle:handle.write(selected.read_bytes())
    text=f'''# Selected model: fixed TabPFN ownership with short-control preservation

The selected full-match candidate is `data/candidate_tabpfn_short_control_v1/candidate.csv`.
It contains {result['full_frame_rows']:,} rows and has SHA256 `{result['candidate_sha256']}`.

Prespecified seed42 retains the confirmed public main SpellF1 **0.7771428571**, compared with0.7411764706 for the previous motion model. Purged and longer-block SpellF1 are0.7314285714 and0.6810810811. All nine matched comparisons across seeds42/43/44 passed; seed44's higher main result did not change the prespecified production seed. Independent reconstruction and complete five-main-fold replay under the revised allocation reproduced probabilities and labels exactly.

The learner,2352-row full paired context, eight views,float32 precision,512-row batches and frozen short-control correction are unchanged. The memory fix assigns5GiB to MPS and reserves1GiB for the host within the same6GiB combined limit. Full-match inference completed40 fresh context fits over158857 official-half frames. All worker integrity/resource checks passed, both separately loaded full CSV exports match exactly, and both pass the official validator. D/X predictions remain exact to the previous full candidate.

The package manifest, full probability evidence and final audit are under `data/candidate_tabpfn_short_control_v1/`; detailed production evidence is under `{ROUND.relative_to(ROOT)}`. Two CSV exports were rebuilt from the completed full probabilities; the entire full-match native query inference was not performed twice. The public validation evidence reuses7500 same-match labels; private full-match accuracy remains unknown.

The previous full candidate and all17 protected artifacts remain unchanged under `data/candidate_motion_dead_trim/`. The exact preceding selected-model document is preserved in [SELECTED_MODEL_BEFORE_TABPFN_20261102.md](SELECTED_MODEL_BEFORE_TABPFN_20261102.md).
'''
    temporary=selected.with_name('SELECTED_MODEL.tabpfn.tmp');temporary.write_text(text)
    pointer=dict(model='tabpfn_short_control_seed42',candidate_path=str((output/'candidate.csv').relative_to(ROOT)),
        candidate_sha256=result['candidate_sha256'],manifest_path=str((output/'manifest.json').relative_to(ROOT)))
    pointer_tmp=ROOT/'experiments/SELECTED_MODEL.tabpfn.json.tmp';native.write_json(pointer_tmp,pointer)
    if (ROUND/'STOP_AFTER_BUILD').exists():raise ValueError('Selection was deferred before pointer update')
    os.replace(temporary,selected);os.replace(pointer_tmp,ROOT/'experiments/SELECTED_MODEL.json')
    with (ROOT/'experiments/WORK_HISTORY.md').open('a') as handle:
        handle.write(f"\n\nFixed TabPFN full-match production selected after exact main0.777142857 replay, verified allocation-only memory repair,40 full context fits, identical full CSV exports and official validation. CSV `{output.relative_to(ROOT)}/candidate.csv`, SHA256 `{result['candidate_sha256']}`. Prior candidate and selected-model document are preserved. Private full-match accuracy remains unknown.\n")
    native.write_json(ROUND/'promotion_complete.json',dict(passed=True,selected=True,previous_candidate_preserved=True,
        selected_candidate=str((output/'candidate.csv').relative_to(ROOT)),csv_sha256=result['candidate_sha256'],
        selected_document_sha256=native.digest(selected),final_audit_sha256=native.digest(ROUND/'final_audit.json'),
        authorization=policy['user_authorization']))
    print('Selected verified full-match fixed TabPFN candidate',output/'candidate.csv',flush=True)


if __name__=='__main__':promote()
