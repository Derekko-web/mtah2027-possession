"""Public-only temporal D experiment, canonical inference and saved-model replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from dfl_signal_audit import matching
from finalize_broadcast_closure import audit_manifest,official_scores
from games import GAME,half_video_blocks
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions
from temporal_dead import normalization,transform,binary_targets,replace_dead_probability,dead_overlay

ROUND=ROOT/'data/control_temporal_dead_v1'
PLAN=ROOT/'experiments/TEMPORAL_DEAD_PLAN.json'
RUNTIME=ROOT/'data/control_sports_action_v1/runtime/bin/python'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')


def check_inputs():
    freeze=ROUND/'protocol_freeze';plan=json.loads(PLAN.read_text())
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Plan changed')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen automatic input differs: '+name)
    for name,expected in json.loads((freeze/'protected_artifacts.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Selected artifact differs')
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((freeze/'EXPERIMENT_LEDGER.csv').read_bytes()):
        raise ValueError('Previous ledger bytes changed')
    for name,target in json.loads((freeze/'symlinks.json').read_text()).items():
        path=ROOT/name
        if not path.is_symlink() or str(path.readlink())!=target:raise ValueError('Source symlink changed')
    for name,expected in json.loads((freeze/'implementation.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen implementation changed: '+name)
    manifest=json.loads((ROUND/'inputs/manifest.json').read_text())
    if not manifest['public_plus31_prefix_exact'] or manifest['targets_loaded']:raise ValueError('Feature provenance invalid')
    for group,base in (('source_sha256',ROOT),('input_sha256',ROOT),('cache_sha256',ROUND/'inputs')):
        for name,expected in manifest[group].items():
            if digest(base/name)!=expected:raise ValueError('Feature dependency changed')
    if not json.loads((ROUND/'synthetic_preflight_final/summary.json').read_text())['passed']:
        raise ValueError('Synthetic temporal/runtime gate failed')
    return plan


def predict_variant(arrays,dead,bc,info,raw,blocks,mode,public):
    reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
    alternate=list(arrays)
    if mode=='trim':alternate[0]=replace_dead_probability(arrays[0],dead,public)
    elif mode=='union':
        alternate[3]=arrays[3].copy();alternate[3][public,0]=np.maximum(arrays[3][public,0],dead)
    else:raise ValueError('Unknown frozen decoder')
    changed=decode_motion_dead_trim(*alternate,bc,info,raw,blocks)
    return dead_overlay(reference,changed,mode)


def assess(scores,segments,official,states,changes,seed):
    rows=[];reference_name=f'current_seed{seed}'
    for name in scores.model.unique():
        if name==reference_name:continue
        for protocol in scores.protocol.unique():
            index=scores[scores.protocol==protocol].set_index('model');old,new=index.loc[reference_name],index.loc[name]
            counts=official[official.protocol==protocol].set_index('model');a,b=counts.loc[reference_name],counts.loc[name]
            before,after=states[(protocol,name)]
            np.testing.assert_array_equal(before['truth'],after['truth'])
            for state,count in ((before,a),(after,b)):
                if state['gt_flags'].sum()!=count.matched_iou50 or (~state['pred_flags']).sum()!=count.spurious_iou50:
                    raise ValueError('Official match accounting differs')
            short=segments[(segments.protocol==protocol)&(segments.label!='D')&(segments.frames<=125)]
            s0=short[short.model==reference_name].set_index(['start','end','label']);s1=short[short.model==name].set_index(['start','end','label'])
            pd.testing.assert_index_equal(s0.index,s1.index)
            flags=dict(primary=bool(new.spellF1>=old.spellF1-1e-12),mean_fold_primary=bool(new.mean_minute_spellF1>=old.mean_minute_spellF1-1e-12),
                macro=bool(new.macroF1>=old.macroF1-.001-1e-12),short_iou=bool(new.short_live_mean_best_iou>=old.short_live_mean_best_iou-1e-12),
                prior_matched=bool(np.all(after['gt_flags'][before['gt_flags']])),prior_short_recall=bool((s1.recall[s0.recall>=.5]>=.5).all()),
                no_extra_spurious=bool(b.spurious_iou50<=a.spurious_iou50),no_extra_missed=bool(b.missed_iou50<=a.missed_iou50),
                D_recall=bool(new.dead_recall>=old.dead_recall-1e-12),false_D=bool(new.false_dead_frames<=old.false_dead_frames+10))
            if protocol=='buffered_minutes':flags['incremental_primary_gain']=bool(new.spellF1>=old.spellF1+.01-1e-12)
            rows.append(dict(protocol=protocol,model=name,seed=seed,checks=flags,spellF1_delta=float(new.spellF1-old.spellF1),
                changed_frames=changes[(protocol,name)],historical_substantial_gain=bool(new.spellF1>=old.spellF1+.04-1e-12),
                historical_extra_short_match=bool(new.short_live_matched_iou50>=old.short_live_matched_iou50+1)))
    passing=[name for name in scores.model.unique() if name!=reference_name and all(all(p['checks'].values()) for p in rows if p['model']==name)]
    def rank(name):
        values=[p for p in rows if p['model']==name]
        buffered=next(p for p in values if p['protocol']=='buffered_minutes')
        return (-buffered['spellF1_delta'],-min(p['spellF1_delta'] for p in values if p['protocol']!='buffered_minutes'),
                sum(p['changed_frames'] for p in values),0 if name.startswith('temporal') else 1,0 if '_trim_' in name else 1)
    passing.sort(key=rank)
    return dict(passed=bool(passing),eligible=passing,selected=passing[0] if passing else None,pairs=rows)


def run(seed=42,replay=False,output_override=None):
    plan=check_inputs();stage='screen' if seed==42 else f'confirmation_seed{seed}'
    heads=['frame','temporal'];decoders=['trim','union']
    if seed!=42:
        selection=json.loads((ROUND/'screen/selection.json').read_text())
        if not selection['passed']:raise ValueError('No eligible seed42 method')
        head,mode,_=selection['selected'].split('_');heads=[head];decoders=[mode]
    saved=ROUND/stage;output=output_override or ROUND/(stage+'_replay' if replay else stage)
    if replay:
        parent=json.loads((saved/'manifest.json').read_text())
        for name,expected in parent['cache_sha256'].items():
            if digest(saved/name)!=expected:raise ValueError('Saved experiment artifact changed: '+name)
    output.mkdir(parents=True,exist_ok=False);(output/'plan.json').write_bytes(PLAN.read_bytes())
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl');n=len(raw);public=np.arange(11185,18685)
    features=np.load(ROUND/'inputs/physical.npy');columns=json.loads((ROUND/'inputs/manifest.json').read_text())['columns']
    if len(features)!=n or features.shape[1]!=len(columns):raise ValueError('Feature shape differs')
    camera=ROOT/'data/control_broadcast_v1/inputs';motion=ROOT/'data/control_pair_motion_v1'
    audit_manifest(camera);audit_manifest(motion/'inputs');audit_manifest(motion/'dead_trim_confirmation')
    bc=pd.read_pickle(camera/'combined_inputs.pkl').iloc[:n];info=pd.read_pickle(motion/'inputs/motion_info.pkl').iloc[:n]
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text());half_blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels!='X'),public)
    baselines={};input_hashes={};identities=[]
    for protocol in plan['protocols']:
        folds=[(s['block'],np.asarray(s['train']),np.asarray(s['validation'])) for s in splits if s['protocol']==protocol]
        blocks=[(int(v[0]),int(v[-1])+1) for _,_,v in folds]
        arrays,hashes=selected_arrays(protocol,seed,n);input_hashes.update(hashes)
        reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
        expected=pd.read_csv(motion/'dead_trim_confirmation'/protocol/'oof_predictions.csv')[f'motion_dead_trim_seed{seed}'].to_numpy()
        np.testing.assert_array_equal(reference[public],expected)
        copy=[a.copy() for a in arrays]
        np.testing.assert_array_equal(decode_motion_dead_trim(*copy,bc,info,raw,blocks),reference)
        for mode,dead in [('trim',arrays[0][public,2]),('union',arrays[3][public,0])]:
            np.testing.assert_array_equal(predict_variant(arrays,dead,bc,info,raw,blocks,mode,public),reference)
        baselines[protocol]=(folds,blocks,arrays,reference);identities.append(protocol)
    print('All unchanged pipeline and self-D identity controls pass',flush=True)
    rows=[];details=[];segments=[];states={};changes={};models=0
    for protocol,(folds,blocks,arrays,reference) in baselines.items():
        folder=output/protocol;folder.mkdir();dead={h:np.full(7500,np.nan) for h in heads}
        table=pd.DataFrame(dict(frame=public,ground_truth=labels[public]));old_name=f'current_seed{seed}'
        table[old_name]=reference[public]
        record_predictions(output,protocol,old_name,public,labels,folds,reference[public],rows,details,segments)
        add_overlap_metrics(rows,segments,table,protocol,old_name)
        for block,train,val in folds:
            job_dir=folder/f'block{block}';job_dir.mkdir()
            mean,scale=normalization(features,train);target,weight=binary_targets(labels,train,val)
            masked=np.full_like(labels,'X');masked[train]=labels[train]
            other_target,other_weight=binary_targets(masked,train,val)
            np.testing.assert_array_equal(target,other_target);np.testing.assert_array_equal(weight,other_weight)
            normalized=transform(features,mean,scale)
            np.save(job_dir/'normalized.npy',normalized)
            np.savez_compressed(job_dir/'fitting.npz',train=train,target=target,weight=weight,validation=val,mean=mean,scale=scale)
            if replay:
                with np.load(saved/protocol/f'block{block}'/'fitting.npz') as record:
                    for key,value in dict(train=train,target=target,weight=weight,validation=val,mean=mean,scale=scale).items():np.testing.assert_array_equal(record[key],value)
            job=dict(normalized=str(job_dir/'normalized.npy'),fitting=str(job_dir/'fitting.npz'),half_blocks=half_blocks,seed=seed,modes=heads)
            if replay:job['parent']=str(saved/protocol/f'block{block}'/'worker')
            write_json(job_dir/'job.json',job)
            command=[str(RUNTIME),str(ROOT/'experiments/temporal_dead_network.py'),'--job',str(job_dir/'job.json'),'--output',str(job_dir/'worker')]
            if replay:command.append('--replay')
            print(('Replay' if replay else 'Fit')+' public-only '+protocol+' block '+str(block),flush=True)
            with (job_dir/'worker.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            record=json.loads((job_dir/'worker/worker.json').read_text())
            if len({r['parameters'] for r in record['records']})!=1:raise ValueError('Control and temporal model parameter counts differ')
            for head in heads:
                value=np.load(job_dir/'worker'/(head+'_probability.npy'))
                if replay:np.testing.assert_array_equal(value,np.load(saved/protocol/f'block{block}'/'worker'/(head+'_probability.npy')))
                dead[head][val-public[0]]=value;models+=not replay
        for head in heads:
            np.save(folder/(head+'_dead_probability.npy'),dead[head])
            for mode in decoders:
                name=f'{head}_{mode}_seed{seed}'
                prediction=predict_variant(arrays,dead[head],bc,info,raw,blocks,mode,public)
                np.testing.assert_array_equal(prediction=='X',reference=='X')
                keep=np.isin(reference,['H','A'])&np.isin(prediction,['H','A'])
                np.testing.assert_array_equal(prediction[keep],reference[keep])
                table[name]=prediction[public]
                record_predictions(output,protocol,name,public,labels,folds,prediction[public],rows,details,segments)
                add_overlap_metrics(rows,segments,table,protocol,name)
                states[(protocol,name)]=(matching(labels[public],reference[public]),matching(labels[public],prediction[public]))
                changes[(protocol,name)]=int((prediction[public]!=reference[public]).sum())
        table.to_csv(folder/'oof_predictions.csv',index=False)
        if replay and (folder/'oof_predictions.csv').read_bytes()!=(saved/protocol/'oof_predictions.csv').read_bytes():raise ValueError('Shared inference replay differs')
    add_short_metrics(rows,segments);scores=pd.DataFrame(rows);diagnostics=pd.DataFrame(segments)
    scores.to_csv(output/'scores.csv',index=False);pd.DataFrame(details).to_csv(output/'fold_scores.csv',index=False)
    diagnostics.to_csv(output/'segment_diagnostics.csv',index=False)
    official=official_scores(output,scores)
    selection=assess(scores,diagnostics,official,states,changes,seed)
    write_json(output/'selection.json',selection)
    if replay:
        for name in ('scores.csv','fold_scores.csv','segment_diagnostics.csv','official_comparison.csv','selection.json'):
            if (output/name).read_bytes()!=(saved/name).read_bytes():raise ValueError('Saved score/gate replay differs: '+name)
    else:
        manifest=dict(plan_sha256=digest(PLAN),input_sha256=input_hashes,source_sha256=json.loads((ROUND/'protocol_freeze/implementation.json').read_text()),
            cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},new_models=models,
            identities=identities,manual_labels_used=False,official_public_target_only=True,all_shared_live_owners_and_unknown_exact=True)
        write_json(output/'manifest.json',manifest)
    check_inputs()
    print(scores[['protocol','model','spellF1','macroF1','false_dead_frames','dead_recall']].to_string(index=False),flush=True)
    print('Eligible methods:',selection['eligible'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    parser.add_argument('--replay',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();run(args.seed,args.replay,args.output)
