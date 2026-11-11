"""Frozen source-aware arrival screen against the current boundary recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from source_aware_arrivals import VARIANTS,refine
from reproduce_home_boundary_precision import BASE
from reproduce_boundary_precision import PROTOCOLS,overlap_summary

ROUND=ROOT/'data/control_home_boundaries_v1'
SOURCES=('experiments/source_aware_arrivals.py','experiments/test_source_aware_arrivals.py',
         'experiments/research_home_boundaries.py','experiments/reproduce_home_boundary_precision.py',
         'experiments/possession_boundary_precision.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    inputs=['data/advanced_validation/raw.pkl','data/advanced_validation/splits.json',
        'data/control_remaining_away_v1/broadcast_extension_18935.pkl','data/control_remaining_away_v1/broadcast_extension_19060.pkl',
        'evaluation/gt/BAR-ATH_possession_gt_public.csv']
    for seed,stage in ((42,'screen'),(43,'confirmation_seed43'),(44,'confirmation_seed44')):
        for p in PROTOCOLS:
            inputs += [str((BASE/f'seed{seed}'/p/'oof_predictions.csv').relative_to(ROOT)),f'data/control_tabpfn_fixed_confirmation_v2/{stage}/{p}/owner_probability.npy']
    protected=['experiments/SELECTED_MODEL.json','experiments/SELECTED_MODEL.md','experiments/EXPERIMENT_LEDGER.csv',
        'data/candidate_tabpfn_boundary_precision_v1/candidate.csv','data/candidate_tabpfn_new_camera_v1/candidate.csv',
        'data/candidate_tabpfn_broadcast_short_v1/candidate.csv','data/candidate_tabpfn_short_control_v1/candidate.csv',
        'data/candidate_motion_dead_trim/candidate.csv']
    plan=dict(hypotheses=['Opposite contact majorities in the two cameras identify short arrivals where a tactical-conditioned posterior veto is unreliable.',
        'Incomplete broadcast coverage cannot support a contact envelope;local saved-owner likelihood can refine its arrival with existing13/25frame radii.'],
        variants=VARIANTS,seed42_first=True,no_targets_in_inference=True,no_model_fits=True,
        gates='All9SpellF1,macroF1,accuracy,meanbestIoU,fiveIoUF1 nondecreasing;allpriorGTflags atall5thresholds retained;bothhome targetIoUs nondecreasing ineachpair;mainbothtargets>=.7andstrictlyimproved;strictmainprimary;39foldprimary/macro nonregress;D/X/topology exact.',
        selection='Eligible methods:maximum minimumprimarygain rounded12,then maingain,then mean of two main targetIoUs,then smaller posteriorradius.',
        limitations='Extensively adaptive single-match public checks,not independent private validation.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES},input_sha256={p:digest(ROOT/p) for p in inputs},
        protected_sha256={p:digest(ROOT/p) for p in protected})
    with (ROUND/'plan.json').open('x') as f:f.write(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    archive=ROUND/'source_freeze';archive.mkdir()
    for name in SOURCES:(archive/Path(name).name).write_bytes((ROOT/name).read_bytes())


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for group in ('source_sha256','input_sha256','protected_sha256'):
        for name,sha in plan[group].items():
            if digest(ROOT/name)!=sha:raise ValueError('Frozen dependency changed: '+name)
    return plan


def run(seed):
    verify();public=np.arange(11185,18685)
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    stage='screen' if seed==42 else f'confirmation_seed{seed}'
    variants=list(VARIANTS) if seed==42 else [json.loads((ROUND/'selection.json').read_text())['selected']]
    rows=[]
    for variant in variants:
        for protocol in PROTOCOLS:
            base=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv');np.testing.assert_array_equal(base.frame,public)
            owner=np.load(ROOT/f'data/control_tabpfn_fixed_confirmation_v2/{stage}/{protocol}/owner_probability.npy')
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            pred,events=refine(base.prediction.to_numpy(),owner,raw,bc,blocks,**VARIANTS[variant])
            truth=pm.label_codes(base.ground_truth);before=pm.evaluate(truth,pm.label_codes(base.prediction),bootstrap_n=0)
            after=pm.evaluate(truth,pm.label_codes(pred),bootstrap_n=0)
            a,b=overlap_summary(base.ground_truth,base.prediction),overlap_summary(base.ground_truth,pred)
            checks={k:after[k]>=before[k]-1e-12 for k in ('spell_f1_mean','macro_f1','accuracy')}
            for t in pm.SPELL_IOUS:
                checks[f'IoU{t}']=after['spell_f1'][t]['f1']>=before['spell_f1'][t]['f1']-1e-12
                checks[f'prior_matches_{t}']=bool(b['flags'][t][a['flags'][t]].all())
            checks['mean_best_iou']=b['mean_best_iou']>=a['mean_best_iou']-1e-12
            targets={}
            for length in (33,87):
                index=next(i for i,(s,e,k) in enumerate(a['gt']) if e-s==length and k==pm.H)
                v0=max((v for v,j,_ in a['overlap'] if j==index),default=0.)
                v1=max((v for v,j,_ in b['overlap'] if j==index),default=0.)
                targets[str(length)]=dict(before_iou=v0,iou=v1);checks[f'home{length}']=v1>=v0-1e-12
            row=dict(seed=seed,variant=variant,protocol=protocol,spellF1=after['spell_f1_mean'],macroF1=after['macro_f1'],
                accuracy=after['accuracy'],mean_best_iou=b['mean_best_iou'],gain=after['spell_f1_mean']-before['spell_f1_mean'],
                changed_frames=int(np.count_nonzero(pred!=base.prediction)),targets=targets,checks=checks)
            folder=ROUND/'candidates'/variant/f'seed{seed}'/protocol;folder.mkdir(parents=True,exist_ok=False)
            pd.DataFrame(dict(frame=public,ground_truth=base.ground_truth,prediction=pred)).to_csv(folder/'oof_predictions.csv',index=False)
            write(folder/'events.json',events);write(folder/'metrics.json',dict(before=before,after=after,summary=row))
            rows.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('targets','checks')} for r in rows]).to_csv(ROUND/f'seed{seed}_scores.csv',index=False)
    if seed==42:
        eligible=[]
        for variant in variants:
            r=[v for v in rows if v['variant']==variant];main=r[0]
            if all(all(v['checks'].values()) for v in r) and main['gain']>1e-12 and all(t['iou']>=.7 and t['iou']>t['before_iou']+1e-12 for t in main['targets'].values()):
                eligible.append(dict(variant=variant,minimum_gain=min(v['gain'] for v in r),main_gain=main['gain'],
                    target_mean_iou=float(np.mean([t['iou'] for t in main['targets'].values()]))))
        eligible.sort(key=lambda r:(round(r['minimum_gain'],12),round(r['main_gain'],12),round(r['target_mean_iou'],12),-VARIANTS[r['variant']]['posterior_radius']),reverse=True)
        selection=dict(eligible=eligible,selected=eligible[0]['variant'] if eligible else None)
        write(ROUND/'selection.json',selection);print('SELECTION',json.dumps(selection),flush=True)
    verify()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44));a=p.parse_args()
    if a.prepare:prepare()
    elif a.seed:run(a.seed)
    else:p.error('Choose --prepare or --seed')
