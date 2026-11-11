"""One frozen public-DFL binary live/dead teacher and whole-match assessment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from dfl_dead_features import build
from dfl_transfer_features import corrupt
from prepare_dfl_dead import validate as audit_validate
from research_temporal_dead import write_json

ROUND=ROOT/'data/control_dfl_dead_v1'
PLAN=ROOT/'experiments/DFL_DEAD_TEACHER_PLAN.json'


def validate():
    audit_validate();plan=json.loads(PLAN.read_text());freeze=ROUND/'protocol_freeze_B'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Teacher plan changed')
    for group in (plan['input_sha256'],json.loads((freeze/'implementation.json').read_text())):
        for name,expected in group.items():
            if digest(ROOT/name)!=expected:raise ValueError('External teacher dependency changed: '+name)
    if not json.loads((ROUND/'assessment_A/summary.json').read_text())['passed']:raise ValueError('External support gate failed')
    return plan


def metrics(status,p,seen):
    status=np.asarray(status);p=np.asarray(p,float);seen=np.asarray(seen,bool)
    known=np.isin(status,[0,1]);y=status[known]==0;p=p[known];seen=seen[known]
    if set(y)!={False,True} or not np.isfinite(p).all():raise ValueError('Insufficient binary assessment support')
    prediction=p>=.5;tp=int((y&prediction).sum());fp=int((~y&prediction).sum());fn=int((y&~prediction).sum())
    recall=tp/max(int(y.sum()),1);live_recall=float((~prediction[~y]).mean())
    visible=y&seen;hidden=y&~seen;clipped=np.clip(p,1e-6,1-1e-6)
    return dict(frames=len(y),dead_frames=int(y.sum()),live_frames=int((~y).sum()),
        balanced_accuracy=(recall+live_recall)/2,dead_precision=tp/max(tp+fp,1),dead_recall=recall,
        dead_f1=2*tp/max(2*tp+fp+fn,1),live_false_dead_rate=1-live_recall,
        visible_dead_frames=int(visible.sum()),visible_dead_recall=float(prediction[visible].mean()) if visible.any() else None,
        hidden_dead_frames=int(hidden.sum()),hidden_dead_recall=float(prediction[hidden].mean()) if hidden.any() else None,
        brier=float(((p-y)**2).mean()),log_loss=float(-(y*np.log(clipped)+(~y)*np.log(1-clipped)).mean()))


def gates(m,rule):
    return dict(balanced_accuracy=m['balanced_accuracy']>=rule['each_masked_validation_match_balanced_accuracy'],
        dead_precision=m['dead_precision']>=rule['dead_precision'],dead_recall=m['dead_recall']>=rule['dead_recall'],
        visible_dead_recall=m['visible_dead_recall'] is not None and m['visible_dead_recall']>=rule['visible_dead_recall'],
        live_false_dead=m['live_false_dead_rate']<=rule['live_false_dead_rate_max'])


def run(output,replay=False):
    plan=validate();saved=ROUND/'external_trial';output.mkdir(parents=True,exist_ok=False)
    if replay:
        for name,expected in json.loads((saved/'manifest.json').read_text())['cache_sha256'].items():
            if digest(saved/name)!=expected:raise ValueError('Saved external artifact changed: '+name)
    (output/'plan.json').write_bytes(PLAN.read_bytes());features_dir=output/'features';features_dir.mkdir()
    source=ROOT/'data/control_dfl_transfer_v1/assessment_A';x=[];y=[];groups=[];records=[];items={}
    for match_index,game in enumerate(plan['matches']):
        for half_index,half in enumerate(('firstHalf','secondHalf')):
            key=game+'_'+half
            with np.load(source/(key+'.npz')) as d:
                xy,ball,teams,status=[d[k] for k in ('positions','ball','teams','status')]
            n=len(status);masked=corrupt(xy,ball,270000+100*match_index+half_index)
            items[key]=dict(status=status,seen={})
            for view,positions,balls in (('clean',xy,ball),('masked',*masked)):
                features=build(positions,balls,teams,[(0,n)]);path=features_dir/(key+'_'+view+'.pkl');features.to_pickle(path)
                items[key]['seen'][view]=np.isfinite(balls).all(axis=1)
                if replay:pd.testing.assert_frame_equal(features,pd.read_pickle(saved/'features'/path.name),check_exact=True)
                if game in plan['splits']['train']:
                    rows=np.flatnonzero(np.isin(status,[0,1])&(np.arange(n)%plan['training']['stride']==0))
                    x.append(features.iloc[rows]);y.append((status[rows]==0).astype(int));groups.append(np.full(len(rows),match_index))
                    records.append(dict(match=game,half=half,view=view,rows=rows))
                print('External D features ready',key,view,'columns',len(features.columns),flush=True)
    inputs=pd.concat(x,ignore_index=True);target=np.concatenate(y);group=np.concatenate(groups)
    weight=np.zeros(len(target));unique=np.unique(group)
    for g in unique:
        for label in (0,1):
            eligible=(group==g)&(target==label)
            if not eligible.any():raise ValueError('Each training match requires both statuses')
            weight[eligible]=len(target)/(len(unique)*2*eligible.sum())
    np.savez_compressed(output/'fitting.npz',target=target,group=group,weight=weight)
    for i,r in enumerate(records):
        np.savez_compressed(output/f'rows_{i:02d}.npz',rows=r['rows'])
    write_json(output/'fitting_provenance.json',[{k:v for k,v in r.items() if k!='rows'} for r in records])
    if replay:
        model=XGBClassifier();model.load_model(saved/'teacher_seed42.ubj')
        with np.load(saved/'fitting.npz') as d:
            for k,value in dict(target=target,group=group,weight=weight).items():np.testing.assert_array_equal(d[k],value)
    else:
        model=XGBClassifier(**plan['training']['params']);model.fit(inputs,target,sample_weight=weight)
        model.save_model(output/'teacher_seed42.ubj')
    print('External D teacher ready; training rows',len(target),flush=True)
    loaded=XGBClassifier();loaded.load_model((saved if replay else output)/'teacher_seed42.ubj')
    scores={}
    for game in plan['splits']['development']+plan['splits']['held_out']:
        scores[game]={}
        for view in ('clean','masked'):
            statuses=[];probabilities=[];visible=[]
            for half in ('firstHalf','secondHalf'):
                key=game+'_'+half;features=pd.read_pickle(features_dir/(key+'_'+view+'.pkl'))
                p=model.predict_proba(features)[:,1];np.testing.assert_array_equal(p,loaded.predict_proba(features)[:,1])
                file=output/(key+'_'+view+'_dead.npy');np.save(file,p)
                if replay:np.testing.assert_array_equal(p,np.load(saved/file.name))
                statuses.append(items[key]['status']);probabilities.append(p);visible.append(items[key]['seen'][view])
            scores[game][view]=metrics(np.concatenate(statuses),np.concatenate(probabilities),np.concatenate(visible))
        scores[game]['checks']=gates(scores[game]['masked'],plan['external_gate'])
        scores[game]['passed']=all(scores[game]['checks'].values())
        print('External masked D metrics',game,json.dumps(scores[game]['masked']),'passes',scores[game]['passed'],flush=True)
    result=dict(passed=all(g['passed'] for g in scores.values()),matches=scores,new_external_models=0 if replay else 1,
        target='direct DFLBallStatus only',training_rows=len(target),manual_labels_used=False,competition_targets_loaded=False,checkpoint_reload_exact=True)
    if replay:result['new_external_models']=1 # Reproduce original artifact accounting; no refit occurs.
    write_json(output/'summary.json',result)
    if replay and (output/'summary.json').read_bytes()!=(saved/'summary.json').read_bytes():raise ValueError('External score replay differs')
    if not replay:write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),input_sha256=plan['input_sha256'],source_sha256=json.loads((ROUND/'protocol_freeze_B/implementation.json').read_text()),
        cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    validate();print('External D model passes every fixed gate:',result['passed'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replay',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();run(args.output or ROUND/('external_trial_replay' if args.replay else 'external_trial'),args.replay)
