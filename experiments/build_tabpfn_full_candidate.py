"""Prepare, predict and independently export the frozen full-match candidate."""
import argparse
import json
from pathlib import Path
import sys
import traceback

import tabpfn_full_match_native as native

ROOT,ROUND,PLAN=native.ROOT,native.ROUND,native.PLAN
sys.path.insert(0,str(ROOT))


def audit_cache(folder):
    record=native.read(folder/'manifest.json')
    for name,sha in record['cache_sha256'].items():
        if native.digest(folder/name)!=sha:raise ValueError('Full cache changed: '+name)
    return record


def prepare():
    native.qualification.verify()
    native.qualification.require_private_runtime(sys.prefix,sys.base_prefix,sys.executable,controller=True)
    if not native.read(native.qualification.ROUND/'final_selection.json')['passed'] or not native.read(native.qualification.ROUND/'replay_complete.json')['passed']:
        raise ValueError('Fresh qualification/replay incomplete')
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from evaluation import possession_metrics as pm
    from games import GAME,half_video_blocks,source_frames
    from rich_features import swap_teams
    from tabpfn_full_features import full_context_frames
    source=ROUND/'full_inputs_v3';m=audit_cache(source);audit_cache(ROUND/'encodings')
    if m['columns']!=166 or not m['public_original102_and_raw_exact'] or not m['fixed64_encoder_public_parity']:
        raise ValueError('Full/public input parity incomplete')
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    centers,observed=full_context_frames(labels)
    inputs=pd.read_pickle(source/'inputs.pkl')
    if inputs.shape!=(source_frames(),166):raise ValueError('Full feature schema differs')
    training=ROUND/'training';training.mkdir(exist_ok=False)
    values=inputs.iloc[centers];paired=pd.concat([values,swap_teams(values)],ignore_index=True)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=False)
    context=imputer.fit_transform(paired).astype(np.float32)
    targets=(labels[centers]=='A').astype(np.int64);targets=np.r_[targets,1-targets]
    if not np.isfinite(context).all() or context.shape!=(len(targets),166):raise ValueError('Invalid paired full context')
    np.save(training/'context.npy',context);np.save(training/'target.npy',targets)
    np.savez_compressed(training/'fitting.npz',context_frames=centers,observation_eligible=observed,
        split_train=np.arange(11185,18685),median=imputer.statistics_,target=targets,footprint_radius=np.array(79))
    native.write_json(training/'provenance.json',dict(seed=42,full_public_training=True,
        feature_columns=list(inputs.columns),context_rows=len(context),complete79_public_footprint=True,
        query_labels_used=False,paired_median_only=True,context_selection_unchanged_except_no_outer_holdout=True))
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)];jobs={}
    for protocol,block,start,end in native.partitions(blocks):
        folder=ROUND/'jobs'/protocol/f'block{block}';folder.mkdir(parents=True)
        original=imputer.transform(inputs.iloc[start:end]).astype(np.float32)
        swapped=imputer.transform(swap_teams(inputs.iloc[start:end])).astype(np.float32)
        for name,value in (('original',original),('swapped',swapped)):
            if not np.isfinite(value).all():raise ValueError('Invalid full query values')
            np.save(folder/(name+'.npy'),value)
        np.save(folder/'query_frames.npy',np.arange(start,end))
        paths=dict(context=training/'context.npy',target=training/'target.npy',
                   original=folder/'original.npy',swapped=folder/'swapped.npy')
        job={name:str(p) for name,p in paths.items()}
        job.update(seed=42,protocol=protocol,block=block,input_sha256={str(p):native.digest(p) for p in paths.values()})
        native.write_json(folder/'job.json',job)
        jobs[f'{protocol}/block{block}']=dict(start=start,end=end,job_sha256=native.digest(folder/'job.json'))
    plan=native.read(native.qualification.PLAN)
    plan.update(round='round79_tabpfn_full_match',purpose='Full production inference after all9 primary and exact reconstruction pass',
        selected_seed=42,jobs=jobs,half_blocks=blocks,context_rows=len(context),query_rows=sum(b-a for a,b in blocks),
        source_sha256=native.source_identity(),same_native_worker=True,operation_seconds=900,worker_seconds=3600,
        scientific_model_unchanged=True,decoder_correction_frozen=True,best_seed_selection=False,
        full_probability_prediction_repeated_twice=False,
        reproducibility='40 fresh fits of identical full context; identical preprocessing/weights; repeated first512 per worker; two full CSV exports in separate processes',
        automatic_promotion=False,all_sources_and_resources_review_required=True)
    tracked=[]
    for folder in (source,ROUND/'encodings',training,ROUND/'jobs'):
        tracked.extend(p for p in folder.rglob('*') if p.is_file())
    for name in ('final_selection.json','replay_complete.json'):
        tracked.append(native.qualification.ROUND/name)
    tracked.extend([ROUND/'feature_plan.json',ROUND/'input_preparation_plan_v3.json',
                    ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',
                    ROOT/'data/control_broadcast_v1/full_inputs/combined_inputs.pkl',
                    ROOT/'data/control_pair_motion_v1/full_inputs/motion_info.pkl'])
    for name in ('hardware_probe.json','acquisition.json'):
        (ROUND/name).write_bytes((native.qualification.ROUND/name).read_bytes());tracked.append(ROUND/name)
    plan['input_sha256']={str(p.relative_to(ROOT)):native.digest(p) for p in tracked}
    with PLAN.open('x') as handle:handle.write(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    native.write_json(ROUND/'production_preparation.json',dict(passed=True,model_calls=False,context_rows=len(context),
        full_query_rows=plan['query_rows'],native_jobs=len(jobs),canonical_query_partition_size=4096,
        all_public_HA_context_only=True,query_labels_used=False,plan_sha256=native.digest(PLAN)))
    print('Prepared',len(jobs),'full-match jobs;',len(context),'paired context rows;',plan['query_rows'],'queries',flush=True)


def predict():
    plan=native.verify();native.require_admission()
    for key in sorted(plan['jobs'],key=lambda k:(k.split('/')[0],int(k.split('/block')[1]))):
        folder=ROUND/'jobs'/key
        print('Fresh full-context fit and prediction',key,flush=True)
        native.supervise(folder/'job.json',folder/'worker')
    native.verify()
    native.write_json(ROUND/'production_prediction_complete.json',dict(passed=True,seed=42,
        jobs=len(plan['jobs']),query_rows=plan['query_rows'],all_fresh_fits=True,canonical512_queries=True,
        supervision_sha256={key:native.digest(ROUND/'jobs'/key/'supervision.json') for key in plan['jobs']}))
    print('All full-match native workers complete',flush=True)


def export(output):
    plan=native.verify()
    if not native.read(ROUND/'production_prediction_complete.json')['passed']:raise ValueError('Full native predictions incomplete')
    import numpy as np
    import pandas as pd
    from games import source_frames
    from evaluation import possession_metrics as pm
    from motion_dead_trim import decode_motion_dead_trim
    from tabpfn_short_control_inference import decode_tabpfn_short_control
    source=ROUND/'full_inputs_v3';audit_cache(source)
    n=source_frames();owner=np.full((n,2),np.nan);seen=np.zeros(n,bool);fingerprint=None;weights=None
    workers=[]
    for key,entry in plan['jobs'].items():
        folder=ROUND/'jobs'/key;worker=folder/'worker';native.validate_job(native.read(folder/'job.json'))
        supervision=native.read(folder/'supervision.json');summary=native.read(worker/'summary.json')
        if not supervision['passed'] or supervision['exit_code']!=0 or supervision['aborted'] is not None or not summary['passed']:
            raise ValueError('Full worker incomplete')
        if supervision['summary_sha256']!=native.digest(worker/'summary.json') or not native.within(supervision['elapsed'],3600):
            raise ValueError('Full worker time or summary binding differs')
        if supervision['peak_owned_rss_plus_full_driver_bytes']>native.LIMIT_BYTES:raise ValueError('Full resource bound exceeded')
        native.validate_operations(worker)
        for name,sha in summary['artifact_sha256'].items():
            if native.digest(worker/name)!=sha:raise ValueError('Full worker artifact changed')
        fp=(worker/'preprocessing_fingerprint.json').read_bytes();wp=(worker/'checkpoint_fingerprint.json').read_bytes()
        if fingerprint is None:fingerprint,weights=fp,wp
        if fp!=fingerprint or wp!=weights:raise ValueError('Independent full-context fits differ')
        start,end=entry['start'],entry['end'];frames=np.load(folder/'query_frames.npy',allow_pickle=False)
        np.testing.assert_array_equal(frames,np.arange(start,end))
        if seen[start:end].any():raise ValueError('Duplicate full queries')
        original,swapped,value=[np.load(worker/(name+'.npy'),allow_pickle=False) for name in ('original_probability','swapped_probability','owner')]
        np.testing.assert_array_equal(value,(original+swapped[:,::-1])/2)
        np.testing.assert_array_equal(np.load(worker/'repeat_first.npy',allow_pickle=False),original[:512])
        if value.shape!=(end-start,2) or not np.isfinite(value).all():raise ValueError('Incomplete full probability rows')
        owner[start:end]=value;seen[start:end]=True
        workers.append(dict(job=key,summary_sha256=native.digest(worker/'summary.json'),supervision_sha256=native.digest(folder/'supervision.json')))
    expected=np.zeros(n,bool)
    for start,end in plan['half_blocks']:expected[start:end]=True
    np.testing.assert_array_equal(seen,expected)
    baseline=ROOT/'data/candidate_motion_dead_trim'
    arrays=[np.load(baseline/(name+'_probabilities.npy'),allow_pickle=False) for name in ('current','previous','fast','heads','broadcast','motion')]
    raw=pd.read_pickle(source/'raw.pkl')
    broadcast=pd.read_pickle(ROOT/'data/control_broadcast_v1/full_inputs/combined_inputs.pkl')
    motion=pd.read_pickle(ROOT/'data/control_pair_motion_v1/full_inputs/motion_info.pkl')
    blocks=plan['half_blocks'];eligible=np.load(ROUND/'encodings/full_eligible.npy',allow_pickle=False)
    reference=decode_motion_dead_trim(*arrays,broadcast,motion,raw,blocks)
    np.testing.assert_array_equal(reference,pd.read_csv(baseline/'candidate.csv').label.to_numpy())
    prediction,events=decode_tabpfn_short_control(arrays,owner,eligible,broadcast,motion,raw,blocks)
    for label in ('D','X'):np.testing.assert_array_equal(prediction==label,reference==label)
    table=pd.DataFrame(dict(frame=np.arange(n),label=prediction));reason=pm.validate_submission(table,n)
    if reason:raise ValueError(reason)
    output=output.resolve()
    if not output.is_relative_to(ROUND.resolve()):raise ValueError('Export must remain in the full production round')
    output.mkdir(parents=True,exist_ok=False);table.to_csv(output/'candidate.csv',index=False)
    np.save(output/'owner_probability.npy',owner)
    native.write_json(output/'restored_events.json',events)
    native.verify()
    native.write_json(output/'manifest.json',dict(passed=True,seed=42,full_frames=n,query_rows=int(seen.sum()),
        candidate_sha256=native.digest(output/'candidate.csv'),prior_candidate_sha256=native.digest(baseline/'candidate.csv'),
        retained_D_X_exact=True,reference_full_decode_exact=True,canonical512_full_half_queries=True,
        all_full_context_preprocessing_and_checkpoint_fingerprints_exact=True,within_worker_first512_repeats_exact=True,
        original_full_native_prediction_not_rerun_twice=True,changed_frames=int((prediction!=reference).sum()),
        class_counts=pd.Series(prediction).value_counts().to_dict(),training_accuracy_not_used_for_selection=True,
        qualified_OOF_main_spellF1=.7771428571428571,private_full_match_accuracy_unknown=True,
        plan_sha256=native.digest(PLAN),workers=workers,
        cache_sha256={p.name:native.digest(p) for p in output.iterdir() if p.is_file()}))
    print('Validated full candidate',output,'SHA256',native.digest(output/'candidate.csv'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--predict',action='store_true');parser.add_argument('--export',type=Path);args=parser.parse_args()
    try:
        if args.prepare:prepare()
        elif args.predict:predict()
        elif args.export:export(args.export)
        else:parser.error('Choose preparation, prediction or export')
    except BaseException as error:
        path=ROUND/('preparation_failure.json' if args.prepare else 'production_failure.json')
        if not path.exists():native.write_json(path,dict(error=repr(error),traceback=traceback.format_exc()))
        raise
