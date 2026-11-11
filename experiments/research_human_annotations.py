"""Frozen reviewed-subset ownership pilot and independent saved-model replay."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from human_annotation_training import digest,reviewed_rows,fitting_arrays,fit_human
from dfl_signal_audit import matching,keep_reference_dead_unknown
from finalize_broadcast_closure import audit_manifest,official_scores
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions
from temporal_candidate import predict_models

ROUND=ROOT/'data/control_human_annotations_v1'
PLAN=ROOT/'experiments/HUMAN_ANNOTATION_PLAN.json'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')


def verify_inputs():
    freeze=ROUND/'protocol_freeze';plan=json.loads(PLAN.read_text())
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Frozen plan differs')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen snapshot/input differs: '+name)
    for name,expected in json.loads((freeze/'protected_artifacts.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Selected artifact differs')
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((freeze/'ledger_before.csv').read_bytes()):
        raise ValueError('Prior ledger bytes changed')
    for name,target in json.loads((freeze/'symlinks.json').read_text()).items():
        p=ROOT/name
        if not p.is_symlink() or str(p.readlink())!=target:raise ValueError('Source symlink differs')
    for name,expected in json.loads((freeze/'implementation.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen implementation differs: '+name)
    manifest=json.loads((ROUND/'inputs/manifest.json').read_text())
    if manifest['plan_sha256']!=digest(PLAN) or not manifest['actual_public_and_human_prefix_parity']:
        raise ValueError('Feature provenance/parity missing')
    for key,base in (('source_sha256',ROOT),('input_sha256',ROOT),('cache_sha256',ROUND/'inputs')):
        for name,expected in manifest[key].items():
            if digest(base/name)!=expected:raise ValueError('Feature dependency differs: '+name)
    return plan


def human_snapshot():
    folder=ROUND/'annotation_snapshot'
    metadata=json.loads((folder/'windows.json').read_text());media=json.loads((folder/'media_manifest.json').read_text())
    snapshot=json.loads((folder/'snapshot.json').read_text());windows={w['window_id']:w for w in metadata['windows']}
    rows=[]
    for item in snapshot['included']:
        key=item['window_id'];path=folder/(key+'.json')
        if digest(path)!=item['annotation_sha256']:raise ValueError('Annotation snapshot changed')
        rows.extend(reviewed_rows(json.loads(path.read_text()),windows[key],snapshot['source_freeze_sha256'],media['clips'][key]['sha256']))
    actual=pd.DataFrame(sorted(rows),columns=['frame','label','window_id'])
    saved=pd.read_csv(folder/'training_labels.csv')
    pd.testing.assert_frame_equal(actual,saved)
    frames=saved.frame.to_numpy();labels=saved.label.to_numpy()
    if len(frames)!=2250 or len(np.unique(frames))!=len(frames) or np.any((frames>=11185-250)&(frames<=18684+250)):
        raise ValueError('Reviewed supervision overlaps public context or is incomplete')
    for window in metadata['windows']:
        if window['split']=='audit' and np.any((frames>=window['start']-250)&(frames<=window['end']+250)):
            raise ValueError('Reviewed supervision reaches sealed audit context')
    return frames,labels


def gates(scores,segments,official,states,seed):
    pairs=[]
    for protocol in scores.protocol.unique():
        old_name,new_name=f'current_seed{seed}',f'human_current_seed{seed}'
        index=scores[scores.protocol==protocol].set_index('model')
        old,new=index.loc[old_name],index.loc[new_name]
        counts=official[official.protocol==protocol].set_index('model')
        a,b=counts.loc[old_name],counts.loc[new_name]
        before,after=states[protocol]
        np.testing.assert_array_equal(before['truth'],after['truth'])
        short=np.array([end-start<=125 for start,end,_ in before['truth']])
        short_gain=int(after['gt_flags'][short].sum()-before['gt_flags'][short].sum())
        short_rows=segments[(segments.protocol==protocol)&(segments.label!='D')&(segments.frames<=125)]
        s0=short_rows[short_rows.model==old_name].set_index(['start','end','label'])
        s1=short_rows[short_rows.model==new_name].set_index(['start','end','label'])
        pd.testing.assert_index_equal(s0.index,s1.index)
        for state,count in ((before,a),(after,b)):
            if int(state['gt_flags'].sum())!=int(count.matched_iou50) or int((~state['pred_flags']).sum())!=int(count.spurious_iou50):
                raise ValueError('Official matching count differs')
        flags=dict(primary=bool(new.spellF1>=old.spellF1-1e-12),mean_fold_primary=bool(new.mean_minute_spellF1>=old.mean_minute_spellF1-1e-12),
            macro=bool(new.macroF1>=old.macroF1-.001-1e-12),short_iou=bool(new.short_live_mean_best_iou>=old.short_live_mean_best_iou-1e-12),
            retain_previous_matches=bool(np.all(after['gt_flags'][before['gt_flags']])),
            retain_previous_short_recall=bool((s1.recall[s0.recall>=.5]>=.5).all()),
            no_extra_spurious=bool(b.spurious_iou50<=a.spurious_iou50),no_extra_missed=bool(b.missed_iou50<=a.missed_iou50))
        if protocol=='buffered_minutes':
            flags['substantial_primary']=bool(new.spellF1>=old.spellF1+.04-1e-12)
            flags['extra_short_match']=bool(short_gain>=1)
        pairs.append(dict(protocol=protocol,seed=seed,checks=flags,spellF1_delta=float(new.spellF1-old.spellF1),
                          macroF1_delta=float(new.macroF1-old.macroF1),official_short_match_delta=short_gain))
    return dict(passed=all(all(pair['checks'].values()) for pair in pairs),pairs=pairs)


def run(seed=42,verify_only=False):
    plan=verify_inputs();stage='screen' if seed==42 else f'confirmation_seed{seed}'
    if seed!=42 and not json.loads((ROUND/'screen/summary.json').read_text())['passed']:
        raise ValueError('Seed42 continuation gate failed')
    saved=ROUND/stage;output=ROUND/(stage+'_replay') if verify_only else saved
    if verify_only:
        previous=json.loads((saved/'manifest.json').read_text())
        for name,expected in previous['cache_sha256'].items():
            if digest(saved/name)!=expected:raise ValueError('Saved pilot artifact changed: '+name)
    output.mkdir(exist_ok=False)
    (output/'plan.json').write_bytes(PLAN.read_bytes())
    human_frames,human_labels=human_snapshot()
    features=pd.read_pickle(ROUND/'inputs/ordered.pkl')
    public=np.arange(11185,18685)
    pd.testing.assert_frame_equal(features.iloc[public],pd.read_pickle(ROOT/'data/advanced_validation/ordered.pkl').iloc[public],check_exact=True)
    if np.max(human_frames)>=len(features):raise ValueError('Human feature rows unavailable')
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl');n=len(raw)
    camera=ROOT/'data/control_broadcast_v1/inputs';motion=ROOT/'data/control_pair_motion_v1'
    audit_manifest(camera);audit_manifest(motion/'inputs');audit_manifest(motion/'dead_trim_confirmation')
    bc=pd.read_pickle(camera/'combined_inputs.pkl').iloc[:n];info=pd.read_pickle(motion/'inputs/motion_info.pkl').iloc[:n]
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels!='X'),public)
    baselines={};input_hashes={};control_reloads=0
    # Verify all unchanged controls before the first alternate fit.
    for protocol in plan['screen']['protocols']:
        folds=[(s['block'],np.array(s['train']),np.array(s['validation'])) for s in splits if s['protocol']==protocol]
        blocks=[(int(v[0]),int(v[-1])+1) for _,_,v in folds]
        arrays,identities=selected_arrays(protocol,seed,n);input_hashes.update(identities)
        reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
        expected=pd.read_csv(motion/'dead_trim_confirmation'/protocol/'oof_predictions.csv')[f'motion_dead_trim_seed{seed}'].to_numpy()
        np.testing.assert_array_equal(reference[public],expected)
        control=np.full((n,3),np.nan)
        for block,train,val in folds:
            path=ROOT/'data/advanced_validation'/protocol/f'ordered_xgboost_frame_seed{seed}_block{block}_0.ubj'
            model=XGBClassifier();model.load_model(path)
            if model.get_booster().feature_names!=list(features) or model.get_booster().num_boosted_rounds()!=160:
                raise ValueError('Original ordered model schema differs')
            control[val]=predict_models([model],features.iloc[val])
            np.testing.assert_array_equal(control[val],arrays[0][val]);control_reloads+=1
            input_hashes[str(path.relative_to(ROOT))]=digest(path)
        identity=list(arrays);identity[0]=control
        np.testing.assert_array_equal(decode_motion_dead_trim(*identity,bc,info,raw,blocks),reference)
        baselines[protocol]=(folds,blocks,arrays,reference)
    print('All13 original models reload exactly; complete selected8 identity controls pass',flush=True)
    rows=[];details=[];segments=[];states={};new_reloads=0
    for protocol,(folds,blocks,arrays,reference) in baselines.items():
        folder=output/protocol;folder.mkdir()
        table=pd.DataFrame(dict(frame=public,ground_truth=labels[public]))
        human_probability=np.full((n,3),np.nan)
        for block,train,val in folds:
            fitting=fitting_arrays(labels,train,human_frames,human_labels,val,plan['model']['human_to_public_weight_mass'])
            masked=np.full_like(labels,'X');masked[train]=labels[train]
            isolated=fitting_arrays(masked,train,human_frames,human_labels,val,plan['model']['human_to_public_weight_mass'])
            for key in fitting:np.testing.assert_array_equal(fitting[key],isolated[key])
            prefix=folder/f'human_seed{seed}_block{block}'
            if verify_only:
                original=saved/protocol/prefix.name
                with np.load(str(original)+'_fitting.npz') as record:
                    for key,value in {**fitting,'validation':val}.items():np.testing.assert_array_equal(record[key],value)
                model=XGBClassifier();model.load_model(str(original)+'.ubj')
            else:
                model=fit_human(features,fitting,seed);model.save_model(str(prefix)+'.ubj')
                np.savez_compressed(str(prefix)+'_fitting.npz',**fitting,validation=val)
                loaded=XGBClassifier();loaded.load_model(str(prefix)+'.ubj')
                np.testing.assert_array_equal(predict_models([model],features.iloc[val]),predict_models([loaded],features.iloc[val]))
                model=loaded
            if model.get_booster().feature_names!=list(features) or model.get_booster().num_boosted_rounds()!=160:raise ValueError('New model schema differs')
            human_probability[val]=predict_models([model],features.iloc[val]);new_reloads+=1
            print(('Replay' if verify_only else 'Fit')+' reviewed-human head: '+protocol+' block '+str(block),flush=True)
        np.save(folder/f'human_seed{seed}_probabilities.npy',human_probability[public])
        if verify_only:np.testing.assert_array_equal(human_probability[public],np.load(saved/protocol/f'human_seed{seed}_probabilities.npy'))
        alternate_arrays=list(arrays);alternate_arrays[0]=human_probability
        alternate=decode_motion_dead_trim(*alternate_arrays,bc,info,raw,blocks)
        prediction=keep_reference_dead_unknown(reference,alternate)
        np.testing.assert_array_equal(prediction=='D',reference=='D');np.testing.assert_array_equal(prediction=='X',reference=='X')
        for name,values in ((f'current_seed{seed}',reference),(f'human_current_seed{seed}',prediction)):
            table[name]=values[public]
            record_predictions(output,protocol,name,public,labels,folds,values[public],rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,name)
        table.to_csv(folder/'oof_predictions.csv',index=False)
        if verify_only:
            if (folder/'oof_predictions.csv').read_bytes()!=(saved/protocol/'oof_predictions.csv').read_bytes():raise ValueError('Prediction CSV replay differs')
        states[protocol]=(matching(labels[public],reference[public]),matching(labels[public],prediction[public]))
    add_short_metrics(rows,segments)
    scores=pd.DataFrame(rows);score_path=output/'scores.csv';scores.to_csv(score_path,index=False)
    pd.DataFrame(details).to_csv(output/'fold_scores.csv',index=False)
    segment_table=pd.DataFrame(segments);segment_table.to_csv(output/'segment_diagnostics.csv',index=False)
    official=official_scores(output,scores)
    summary=gates(scores,segment_table,official,states,seed)
    summary.update(reviewed_clips=3,reviewed_training_frames=2250,source_mass_ratio=.5,new_models=0 if verify_only else new_reloads,
                   checkpoint_reloads=new_reloads,control_reloads=control_reloads,all_identity_controls_passed=True,all_final_D_X_exact=True,
                   decision='eligible_for_confirmation' if summary['passed'] else 'rejected_fixed_human_augmentation',audit_remains_sealed=True)
    write_json(output/'summary.json',summary)
    if verify_only:
        for name in ('scores.csv','fold_scores.csv','segment_diagnostics.csv','official_comparison.csv'):
            if (output/name).read_bytes()!=(saved/name).read_bytes():raise ValueError('Score replay differs: '+name)
    else:
        manifest=dict(plan_sha256=digest(PLAN),input_sha256=input_hashes,source_sha256=json.loads((ROUND/'protocol_freeze/implementation.json').read_text()),
                      cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},new_models=new_reloads,
                      matched_control_identity=True,held_labels_not_fitting_inputs=True,human_source='Reviewed native video annotations, not official corrected tracking')
        write_json(output/'manifest.json',manifest)
    verify_inputs()
    print(scores[['protocol','model','spellF1','macroF1','short_live_matched_iou50']].to_string(index=False),flush=True)
    print('Frozen continuation gate:',summary['passed'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    parser.add_argument('--verify-only',action='store_true');args=parser.parse_args();run(args.seed,args.verify_only)
