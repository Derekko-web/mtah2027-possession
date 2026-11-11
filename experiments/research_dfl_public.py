"""One externally gated paired public ownership trial."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from dfl_transfer_data import digest
from dfl_transfer_features import geometry
from dfl_transfer_teacher import fit,probability
from games import GAME,GAMES,half_video_blocks
from finalize_broadcast_closure import audit_manifest
from motion_dead_trim import decode_motion_dead_trim
from provided_speed import decode_provided_speed
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions


def main():
    root=ROOT/'data/control_dfl_transfer_v1'
    plan_path=ROOT/'experiments/DFL_TRANSFER_TEACHER_PLAN.json'
    plan=json.loads(plan_path.read_text());freeze=root/'protocol_freeze_B'
    if plan_path.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('B plan changed')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('B input changed: '+name)
    sources=json.loads((freeze/'implementation.json').read_text())
    for name,expected in sources.items():
        if digest(ROOT/name)!=expected:raise ValueError('B source changed: '+name)
    external=json.loads((root/'external_trial/manifest.json').read_text())
    for name,expected in external['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('External source changed')
    for name,expected in external['cache_sha256'].items():
        if digest(root/'external_trial'/name)!=expected:raise ValueError('External artifact changed')
    if not json.loads((root/'external_trial/summary.json').read_text())['passed']:
        raise ValueError('STOP: external performance gate failed')
    output=root/'public_trial';output.mkdir()
    source=ROOT/'data/advanced_validation'
    raw,ordered=[pd.read_pickle(source/(name+'.pkl')) for name in ('raw','ordered')]
    n=len(raw)
    parent=ROOT/'data/control_broadcast_v1/inputs'
    audit_manifest(parent)
    with np.load(parent/'tactical_observations.npz') as observations:
        positions,ball,teams,valid=[observations[k] for k in ('positions','ball','teams','valid')]
    positions=positions[:n].copy();ball=ball[:n].copy();valid=valid[:n]
    positions[~valid]=np.nan;ball[~valid]=np.nan
    pitch=np.array(GAMES[GAME]['pitch'])
    ball[~(np.isfinite(ball).all(axis=1)&(ball>=-5).all(axis=1)&(ball<=pitch+5).all(axis=1))]=np.nan
    half_blocks=[(start,end+1) for _,start,end in half_video_blocks(GAME)]
    inputs=geometry(positions,ball,teams,half_blocks)
    short=geometry(positions[:18935],ball[:18935],teams,half_blocks)
    pd.testing.assert_frame_equal(inputs.iloc[11185:18685],short.iloc[11185:18685],check_exact=True)
    public=np.arange(11185,18685)
    coverage=(inputs.h__n_tracks.iloc[public]>=4)&(inputs.a__n_tracks.iloc[public]>=4)
    if coverage.mean()<.8 or inputs.ball_seen.iloc[public].mean()<.8:
        raise ValueError('STOP before targets: native shared geometry support failed')
    teacher=XGBClassifier();teacher.load_model(root/'external_trial/teacher_seed42.ubj')
    p=probability(teacher,inputs)
    views=dict(matched_control=inputs,external_teacher=inputs.assign(h__teacher=p,a__teacher=1-p))
    for name,view in views.items():view.to_pickle(output/(name+'_inputs.pkl'))
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    splits=json.loads((source/'splits.json').read_text());(output/'splits.json').write_bytes((source/'splits.json').read_bytes())
    bc=pd.read_pickle(parent/'combined_inputs.pkl').iloc[:n]
    upstream=ROOT/'data/control_pair_motion_v1'
    audit_manifest(upstream/'inputs');audit_manifest(upstream/'dead_trim_confirmation')
    info=pd.read_pickle(upstream/'inputs/motion_info.pkl').iloc[:n]
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels!='X'),public)
    rows=[];details=[];segments=[]
    for protocol in plan['public_screen']['protocols']:
        folder=output/protocol;folder.mkdir()
        folds=[(f['block'],np.array(f['train']),np.array(f['validation'])) for f in splits if f['protocol']==protocol]
        blocks=[(int(v[0]),int(v[-1])+1) for _,_,v in folds]
        arrays,_=selected_arrays(protocol,42,n)
        reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)[public]
        np.testing.assert_array_equal(reference,pd.read_csv(upstream/'dead_trim_confirmation'/protocol/'oof_predictions.csv').motion_dead_trim_seed42)
        table=pd.DataFrame(dict(frame=public,ground_truth=labels[public],current_seed42=reference))
        record_predictions(output,protocol,'current_seed42',public,labels,folds,reference,rows,details,segments)
        add_overlap_metrics(rows,segments,table,protocol,'current_seed42')
        for head,features in views.items():
            pp=np.full((n,3),np.nan)
            for block,train,val in folds:
                if np.any((train>=val.min()-250)&(train<=val.max()+250)):
                    raise ValueError('Public training buffer violated')
                live=train[np.isin(labels[train],['H','A'])]
                target=(labels[live]=='H').astype(int)
                model,weight=fit(features.iloc[live],target,np.zeros(len(live),int))
                prefix=folder/f'{head}_seed42_block{block}'
                model.save_model(str(prefix)+'.ubj')
                np.savez_compressed(str(prefix)+'_fitting.npz',train=train,live=live,validation=val,target=target,weight=weight)
                home=probability(model,features.iloc[val]);pp[val]=np.c_[home,1-home,np.zeros(len(val))]
                print('Public paired fit',protocol,head,block,flush=True)
            np.save(folder/f'{head}_seed42_probabilities.npy',pp[public])
            prediction=decode_provided_speed(arrays,pp,bc,info,raw,blocks,'live_replace')[public]
            assert np.array_equal(prediction=='D',reference=='D')
            name=head+'_seed42';table[name]=prediction
            record_predictions(output,protocol,name,public,labels,folds,prediction,rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,name)
        add_short_metrics(rows,segments)
        table.to_csv(folder/'oof_predictions.csv',index=False)
        pd.DataFrame(rows).to_csv(output/'scores.csv',index=False)
        pd.DataFrame(details).to_csv(output/'fold_scores.csv',index=False)
        pd.DataFrame(segments).to_csv(output/'segment_diagnostics.csv',index=False)
    for name,expected in sources.items():
        if digest(ROOT/name)!=expected:raise ValueError('B source changed during trial')
    manifest=dict(plan_sha256=digest(plan_path),source_sha256=sources,input_sha256=plan['input_sha256'],
        cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        new_models=26,actual_prefix_parity=True,competition_target_source='provided public CSV only',
        matched_geometry_columns=len(inputs.columns),teacher_columns=2,external_gate_passed=True,selected_eight_reference_exact=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(rows)[['protocol','model','spellF1','macroF1','short_live_matched_iou50']].to_string(index=False))


if __name__=='__main__':main()
