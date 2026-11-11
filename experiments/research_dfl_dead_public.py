"""Paired original/teacher-augmented D heads through complete selected8 inference."""
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
from build_temporal_candidate import digest
from dfl_signal_audit import matching
from finalize_broadcast_closure import audit_manifest,official_scores
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_dfl_dead_external import validate as external_validate,ROUND
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions
from research_temporal_dead import assess,write_json
from stoppage_context import fit_context

PLAN=ROOT/'experiments/DFL_DEAD_PUBLIC_PLAN.json'


def preserve_owners(reference,alternative):
    reference=np.asarray(reference);alternative=np.asarray(alternative)
    if reference.shape!=alternative.shape:raise ValueError('Unaligned D alternative')
    result=reference.copy();result[(reference!='X')&(alternative=='D')]='D'
    use=(reference=='D')&np.isin(alternative,['H','A']);result[use]=alternative[use]
    np.testing.assert_array_equal(result=='X',reference=='X')
    shared=np.isin(result,['H','A'])&np.isin(reference,['H','A']);np.testing.assert_array_equal(result[shared],reference[shared])
    return result


def validate():
    external_validate();freeze=ROUND/'protocol_freeze_C';plan=json.loads(PLAN.read_text())
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Public plan changed')
    for group in (plan['input_sha256'],json.loads((freeze/'implementation.json').read_text())):
        for name,expected in group.items():
            if digest(ROOT/name)!=expected:raise ValueError('Public dependency changed: '+name)
    if not json.loads((ROUND/'external_trial/summary.json').read_text())['passed']:raise ValueError('External D gate failed')
    m=json.loads((ROUND/'native_inputs/manifest.json').read_text())
    if not m['public_features_exact'] or not m['public_teacher_exact'] or m['targets_loaded']:raise ValueError('Native feature provenance invalid')
    for group,base in (('cache_sha256',ROUND/'native_inputs'),('input_sha256',ROOT)):
        for name,expected in m[group].items():
            if digest(base/name)!=expected:raise ValueError('Native feature dependency changed')
    return plan


def fit_arrays(labels,train,weak_frames,weak_target):
    if np.intersect1d(train,weak_frames).size:raise ValueError('Weak/public overlap')
    target=(labels[train]=='D').astype(int);weight=len(target)/(2*np.bincount(target)[target])
    other=len(target)*.5/(2*np.bincount(weak_target)[weak_target])
    return np.r_[target,weak_target],np.r_[weight,other]


def run(seed=42,replay=False,output_override=None):
    plan=validate();stage='public_trial' if seed==42 else f'confirmation_seed{seed}'
    if seed!=42 and not json.loads((ROUND/'public_trial/selection.json').read_text())['passed']:raise ValueError('No substantial seed42 improvement')
    saved=ROUND/stage;output=output_override or ROUND/(stage+'_replay' if replay else stage);output.mkdir(parents=True,exist_ok=False)
    if replay:
        for name,expected in json.loads((saved/'manifest.json').read_text())['cache_sha256'].items():
            if digest(saved/name)!=expected:raise ValueError('Saved public artifact changed')
    (output/'plan.json').write_bytes(PLAN.read_bytes());public=np.arange(11185,18685)
    cache=ROOT/'data/control_renewal_v1/stoppage_inputs';raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl');n=len(raw)
    views={k:pd.read_pickle(cache/(name+'.pkl')) for k,name in (('physical','physical'),('restart','restart'))}
    teacher=np.load(ROUND/'native_inputs/teacher_full.npy');weak=np.load(cache/'weak_samples.npz');weak_frames,weak_target=weak['frames'],weak['target']
    if ((weak_frames>=11185-250)&(weak_frames<=18684+250)).any():raise ValueError('Weak fitting touches public buffer')
    windows=json.loads((ROUND/'protocol_freeze_A/reserved_windows.json').read_text())
    for w in windows['windows'] if isinstance(windows,dict) else windows:
        if ((weak_frames>=w['start']-250)&(weak_frames<=w['end']+250)).any():raise ValueError('Weak fitting touches reserved window')
    if any(len(v)!=len(teacher) for v in views.values()):raise ValueError('Unaligned full teacher/physical inputs')
    bc=pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:n]
    info=pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[:n]
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels!='X'),public)
    baselines={};hashes={};controls={};models=0
    for protocol in plan['protocols']:
        folder=output/protocol;folder.mkdir()
        folds=[(s['block'],np.asarray(s['train']),np.asarray(s['validation'])) for s in splits if s['protocol']==protocol]
        blocks=[(int(v[0]),int(v[-1])+1) for _,_,v in folds];arrays,h=selected_arrays(protocol,seed,n);hashes.update(h)
        reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
        expected=pd.read_csv(ROOT/'data/control_pair_motion_v1/dead_trim_confirmation'/protocol/'oof_predictions.csv')[f'motion_dead_trim_seed{seed}'].to_numpy()
        np.testing.assert_array_equal(reference[public],expected);baselines[protocol]=(folds,blocks,arrays,reference)
        head=np.full((n,2),np.nan)
        for index,(name,features) in enumerate(views.items()):
            for block,train,val in folds:
                path=folder/f'control_{name}_block{block}.ubj';target,weight=fit_arrays(labels,train,weak_frames,weak_target)
                if replay:model=XGBClassifier();model.load_model(saved/protocol/path.name)
                else:
                    model=fit_context(features,labels,train,weak_frames,weak_target,.5,seed);model.save_model(path);models+=1
                head[val,index]=model.predict_proba(features.iloc[val])[:,1]
        np.testing.assert_array_equal(head[public],arrays[3][public])
        alternate=list(arrays);alternate[3]=head
        np.testing.assert_array_equal(decode_motion_dead_trim(*alternate,bc,info,raw,blocks),reference)
        controls[protocol]=head
    print('All26 original D-head refits and complete selected8 identities pass',flush=True)
    rows=[];details=[];segments=[];states={};changes={}
    for protocol,(folds,blocks,arrays,reference) in baselines.items():
        folder=output/protocol;heads=np.full((n,2),np.nan)
        for index,(name,features) in enumerate(views.items()):
            features=features.assign(external_dead=teacher)
            for block,train,val in folds:
                path=folder/f'teacher_{name}_block{block}.ubj';target,weight=fit_arrays(labels,train,weak_frames,weak_target)
                masked=np.full_like(labels,'X');masked[train]=labels[train]
                t,w=fit_arrays(masked,train,weak_frames,weak_target);np.testing.assert_array_equal(t,target);np.testing.assert_array_equal(w,weight)
                fit_path=folder/f'teacher_{name}_block{block}_fitting.npz'
                np.savez_compressed(fit_path,train=train,validation=val,weak_frames=weak_frames,target=target,weight=weight)
                if replay:
                    model=XGBClassifier();model.load_model(saved/protocol/path.name)
                    with np.load(saved/protocol/fit_path.name) as d:
                        for k,v in dict(train=train,validation=val,weak_frames=weak_frames,target=target,weight=weight).items():np.testing.assert_array_equal(d[k],v)
                else:
                    print('Fit teacher-augmented D head',protocol,name,block,flush=True)
                    model=fit_context(features,labels,train,weak_frames,weak_target,.5,seed);model.save_model(path);models+=1
                p=model.predict_proba(features.iloc[val])[:,1]
                loaded=XGBClassifier();loaded.load_model((saved/protocol/path.name) if replay else path)
                np.testing.assert_array_equal(p,loaded.predict_proba(features.iloc[val])[:,1]);heads[val,index]=p
        np.save(folder/'teacher_heads.npy',heads[public]);np.save(folder/'control_heads.npy',controls[protocol][public])
        alternate=list(arrays);alternate[3]=heads
        prediction=preserve_owners(reference,decode_motion_dead_trim(*alternate,bc,info,raw,blocks))
        table=pd.DataFrame(dict(frame=public,ground_truth=labels[public]));name=f'teacher_heads_seed{seed}'
        for method,value in ((f'current_seed{seed}',reference),(f'matched_control_seed{seed}',reference),(name,prediction)):
            table[method]=value[public];record_predictions(output,protocol,method,public,labels,folds,value[public],rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,method)
        table.to_csv(folder/'oof_predictions.csv',index=False)
        for method,value in ((f'matched_control_seed{seed}',reference),(name,prediction)):
            states[(protocol,method)]=(matching(labels[public],reference[public]),matching(labels[public],value[public]))
            changes[(protocol,method)]=int((value[public]!=reference[public]).sum())
        if replay and (folder/'oof_predictions.csv').read_bytes()!=(saved/protocol/'oof_predictions.csv').read_bytes():raise ValueError('Public inference replay differs')
    add_short_metrics(rows,segments);scores=pd.DataFrame(rows);diagnostics=pd.DataFrame(segments)
    scores.to_csv(output/'scores.csv',index=False);pd.DataFrame(details).to_csv(output/'fold_scores.csv',index=False);diagnostics.to_csv(output/'segment_diagnostics.csv',index=False)
    official=official_scores(output,scores);selection=assess(scores,diagnostics,official,states,changes,seed)
    for pair in selection['pairs']:
        if pair['protocol']=='buffered_minutes':
            pair['checks'].pop('incremental_primary_gain')
            pair['checks']['substantial_primary_gain']=pair['historical_substantial_gain']
            pair['checks']['extra_short_match']=pair['historical_extra_short_match']
    name=f'teacher_heads_seed{seed}';selection['passed']=all(all(p['checks'].values()) for p in selection['pairs'] if p['model']==name)
    selection['eligible']=[name] if selection['passed'] else [];selection['selected']=name if selection['passed'] else None
    write_json(output/'selection.json',selection)
    if replay:
        for name in ('scores.csv','fold_scores.csv','segment_diagnostics.csv','official_comparison.csv','selection.json'):
            if (output/name).read_bytes()!=(saved/name).read_bytes():raise ValueError('Public metric replay differs')
    else:write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),input_sha256=hashes,new_models=models,manual_labels_used=False,full_selected_identity=True,
        cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    validate();print(scores[['protocol','model','spellF1','macroF1','false_dead_frames','dead_recall']].to_string(index=False),flush=True)
    print('Substantial improvement passes:',selection['passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[42,43,44],default=42);p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.seed,a.replay,a.output)
