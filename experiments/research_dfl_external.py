"""Frozen external ownership trial; requires Phase A acceptance and B protocol."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
# On promotion into experiments/, ROOT is the project root.
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from dfl_transfer_data import digest
from dfl_transfer_features import geometry,corrupt
from dfl_transfer_teacher import fit,probability,nearest,binary_metrics


def conditional_spells(labels,p):
    labels=np.asarray(labels)
    prediction=labels.copy();live=np.isin(labels,['H','A'])
    prediction[live]=np.where(p[live]>=.5,'H','A')
    gt=pm.label_codes(labels);pred=pm.label_codes(prediction)
    scores=pm.spell_scores(gt,pred,pm.valid_runs(gt))
    runs=pm.possession_spells(gt,pm.valid_runs(gt))
    predicted=pm.possession_spells(pred,pm.valid_runs(gt))
    overlaps=pm.spell_overlaps(runs,predicted)
    matched=pm.matched_gt_flags(overlaps,len(runs),.5)
    short=np.array([end-start<=125 for start,end,_ in runs])
    return dict(conditional_spell_f1=scores['mean_f1'],gt_spells=len(runs),pred_spells=len(predicted),
        short_gt_spells=int(short.sum()),short_matched_iou50=int((short&matched).sum()),
        reference_D_X_used=True)


def main():
    round_root=ROOT/'data/control_dfl_transfer_v1'
    plan_path=ROOT/'experiments/DFL_TRANSFER_TEACHER_PLAN.json'
    plan=json.loads(plan_path.read_text())
    freeze=round_root/'protocol_freeze_B'
    if plan_path.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Phase B plan differs')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Phase B input identity differs: '+name)
    for name,expected in json.loads((freeze/'implementation.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Phase B source identity differs: '+name)
    source=round_root/'assessment_A'
    if not json.loads((source/'summary.json').read_text())['passed']:raise ValueError('Data gate failed')
    output=round_root/'external_trial'
    output.mkdir()
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    feature_folder=output/'features';feature_folder.mkdir()
    keys={};train_x=[];train_y=[];train_groups=[];training=[]
    for match_index,game in enumerate(plan['matches']):
        for half_index,section in enumerate(('firstHalf','secondHalf')):
            key=f'{game}_{section}'
            filename=source/(key+'.npz')
            expected=json.loads((source/'manifest.json').read_text())['cache_sha256'][filename.name]
            if digest(filename)!=expected:raise ValueError('Parsed DFL cache changed')
            with np.load(filename) as data:
                positions,ball,teams=[data[name] for name in ('positions','ball','teams')]
                labels=data['labels'];n=len(labels)
            seed=270000+100*match_index+half_index
            masked=corrupt(positions,ball,seed)
            views=dict(clean=geometry(positions,ball,teams,[(0,n)]),
                masked=geometry(*masked,teams,[(0,n)]))
            keys[key]=dict(match=game,section=section,labels=labels,files={})
            for view,features in views.items():
                path=feature_folder/(key+'_'+view+'.pkl');features.to_pickle(path)
                keys[key]['files'][view]=path
                if game in plan['splits']['train']:
                    rows=np.flatnonzero(np.isin(labels,['H','A'])&(np.arange(n)%5==0))
                    train_x.append(features.iloc[rows]);train_y.append((labels[rows]=='H').astype(int))
                    train_groups.append(np.full(len(rows),match_index))
                    training.append(dict(match=game,section=section,view=view,rows=rows,target=train_y[-1]))
            print('Shared external XY features ready',game,section,flush=True)
    inputs=pd.concat(train_x,ignore_index=True);target=np.concatenate(train_y);groups=np.concatenate(train_groups)
    model,weight=fit(inputs,target,groups)
    model.save_model(output/'teacher_seed42.ubj')
    np.savez_compressed(output/'fitting.npz',target=target,group=groups,weight=weight)
    for i,record in enumerate(training):
        np.savez_compressed(output/f'training_rows_{i:02d}.npz',rows=record['rows'],target=record['target'])
    scores={}
    for game in plan['splits']['development']+plan['splits']['held_out']:
        scores[game]={}
        for view in ('clean','masked'):
            label_parts=[];teacher_parts=[];baseline_parts=[]
            for section in ('firstHalf','secondHalf'):
                item=keys[f'{game}_{section}'];features=pd.read_pickle(item['files'][view])
                p=probability(model,features);baseline=nearest(features)
                np.save(output/f'{game}_{section}_{view}_teacher.npy',p)
                np.save(output/f'{game}_{section}_{view}_baseline.npy',baseline)
                label_parts.extend([item['labels'],np.array(['X'])]);teacher_parts.extend([p,np.array([.5])]);baseline_parts.extend([baseline,np.array([.5])])
            labels=np.concatenate(label_parts);p=np.concatenate(teacher_parts);baseline=np.concatenate(baseline_parts)
            scores[game][view]={name:dict(**binary_metrics(labels,values),**conditional_spells(labels,values))
                for name,values in (('teacher',p),('nearest',baseline))}
        teacher=scores[game]['masked']['teacher'];baseline=scores[game]['masked']['nearest']
        scores[game]['gate']=dict(minimum_balanced_accuracy=teacher['balanced_accuracy']>=plan['external_gate']['minimum_masked_balanced_accuracy'],
            gain_over_nearest=teacher['balanced_accuracy']>=baseline['balanced_accuracy']+plan['external_gate']['minimum_gain_over_nearest'])
        scores[game]['passed']=all(scores[game]['gate'].values())
        print('External aggregate',game,json.dumps(dict(teacher=teacher['balanced_accuracy'],nearest=baseline['balanced_accuracy'],passed=scores[game]['passed'])),flush=True)
    result=dict(passed=all(v['passed'] for v in scores.values()),scores=scores,new_external_models=1,
        augmented_training_rows=len(target)*2,training_matches=plan['splits']['train'],competition_target_reads=0,new_public_models=0,new_competition_score_rows=0)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    artifact_paths=[p for p in sorted(output.rglob('*')) if p.is_file()]
    manifest=dict(plan_sha256=digest(plan_path),source_sha256=json.loads((freeze/'implementation.json').read_text()),
        input_sha256=plan['input_sha256'],cache_sha256={str(p.relative_to(output)):digest(p) for p in artifact_paths})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],new_external_models=1,competition_target_reads=0)),flush=True)


if __name__=='__main__':main()
