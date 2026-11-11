"""Corroborated boundary qualification; all prior failed screens stay frozen."""
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
from probability_guarded_boundaries import guarded_refine,VARIANTS
from reproduce_boundary_precision import BASE,PROTOCOLS,overlap_summary

ROUND=ROOT/'data/control_guarded_boundaries_v1'
SOURCES=('experiments/probability_guarded_boundaries.py','experiments/test_probability_guarded_boundaries.py',
         'experiments/research_guarded_boundaries.py','experiments/possession_boundary_precision.py',
         'experiments/reproduce_boundary_precision.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    ROUND.mkdir(exist_ok=False)
    prior=json.loads((ROOT/'data/control_boundary_precision_v1/plan.json').read_text())
    inputs=dict(prior['input_sha256'])
    for seed,stage in ((42,'screen'),(43,'confirmation_seed43'),(44,'confirmation_seed44')):
        for protocol in PROTOCOLS:
            name=f'data/control_tabpfn_fixed_confirmation_v2/{stage}/{protocol}/owner_probability.npy';inputs[name]=digest(ROOT/name)
    for name in ('data/control_boundary_precision_v1/plan.json','data/control_boundary_precision_v1/seed42_scores.csv'):
        inputs[name]=digest(ROOT/name)
    plan=dict(hypothesis='Unsafe contact cuts can be rejected when the saved ownership posterior still favors the original owner;proposedreplacement must have meanprobability>.5.',
        variants=VARIANTS,no_probability_or_parameter_refits=True,seed42_first=True,
        gates='All9SpellF1,macroF1,accuracy,each5IoUF1,meanbestIoU nondecreasing;everyGTflag atall5IoUs retained;D/X/topology exact;strictmainprimary/overlap gain;39clippedfoldprimary/macro checks.',
        selection='Among eligible choose minimumprimarygain rounded12,then mainprimarygain,then meanIoU gain,then simpler method.',
        input_sha256=inputs,source_sha256={p:digest(ROOT/p) for p in SOURCES},protected_sha256=prior['protected_sha256'])
    with (ROUND/'plan.json').open('x') as f:f.write(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    archive=ROUND/'source_freeze';archive.mkdir()
    for name in SOURCES:(archive/Path(name).name).write_bytes((ROOT/name).read_bytes())


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for group in ('input_sha256','source_sha256','protected_sha256'):
        for name,sha in plan[group].items():
            if digest(ROOT/name)!=sha:raise ValueError('Frozen dependency changed: '+name)
    return plan


def run(seed):
    verify();public=np.arange(11185,18685)
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    old=np.load(ROOT/'data/control_broadcast_v1/inputs/strict_mapping.npz')['frames'][public]>=0
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    stage='screen' if seed==42 else f'confirmation_seed{seed}'
    variants=list(VARIANTS) if seed==42 else [json.loads((ROUND/'selection.json').read_text())['selected']]
    rows=[]
    for variant in variants:
        for protocol in PROTOCOLS:
            base=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv');np.testing.assert_array_equal(base.frame,public)
            events=json.loads((BASE/f'seed{seed}'/protocol/'events.json').read_text())
            fast=np.load(ROOT/f'data/control_boundary_v1/baseline_confirmation/{protocol}/fast_seed{seed}_probabilities.npy')
            owner=np.load(ROOT/f'data/control_tabpfn_fixed_confirmation_v2/{stage}/{protocol}/owner_probability.npy')
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            pred,changes=guarded_refine(base.prediction.to_numpy(),fast,owner,raw,bc,old,blocks,events,**VARIANTS[variant])
            gt=pm.label_codes(base.ground_truth);before=pm.evaluate(gt,pm.label_codes(base.prediction),bootstrap_n=0)
            after=pm.evaluate(gt,pm.label_codes(pred),bootstrap_n=0)
            a,b=overlap_summary(base.ground_truth,base.prediction),overlap_summary(base.ground_truth,pred)
            checks={k:after[k]>=before[k]-1e-12 for k in ('spell_f1_mean','macro_f1','accuracy')}
            for t in pm.SPELL_IOUS:
                checks[f'IoU{t}']=after['spell_f1'][t]['f1']>=before['spell_f1'][t]['f1']-1e-12
                checks[f'prior_matches_{t}']=bool(b['flags'][t][a['flags'][t]].all())
            checks['mean_best_iou']=b['mean_best_iou']>=a['mean_best_iou']-1e-12
            row=dict(seed=seed,variant=variant,protocol=protocol,spellF1=after['spell_f1_mean'],macroF1=after['macro_f1'],
                accuracy=after['accuracy'],mean_best_iou=b['mean_best_iou'],gain=after['spell_f1_mean']-before['spell_f1_mean'],
                overlap_gain=b['mean_best_iou']-a['mean_best_iou'],changed_frames=int(np.count_nonzero(pred!=base.prediction)),checks=checks)
            output=ROUND/'candidates'/variant/f'seed{seed}'/protocol;output.mkdir(parents=True,exist_ok=False)
            pd.DataFrame(dict(frame=public,ground_truth=base.ground_truth,prediction=pred)).to_csv(output/'oof_predictions.csv',index=False)
            write(output/'events.json',changes);write(output/'metrics.json',dict(before=before,after=after,summary=row))
            rows.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame([{k:v for k,v in row.items() if k!='checks'} for row in rows]).to_csv(ROUND/f'seed{seed}_scores.csv',index=False)
    if seed==42:
        eligible=[]
        for variant in variants:
            values=[r for r in rows if r['variant']==variant]
            if all(all(r['checks'].values()) for r in values) and values[0]['gain']>1e-12 and values[0]['overlap_gain']>1e-12:
                eligible.append(dict(variant=variant,minimum_gain=min(r['gain'] for r in values),main_gain=values[0]['gain'],overlap_gain=values[0]['overlap_gain']))
        eligible.sort(key=lambda r:(round(r['minimum_gain'],12),round(r['main_gain'],12),round(r['overlap_gain'],12),r['variant']=='guarded_release'),reverse=True)
        selection=dict(eligible=eligible,selected=eligible[0]['variant'] if eligible else None)
        write(ROUND/'selection.json',selection);print('SELECTION',json.dumps(selection),flush=True)
    verify()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44));a=p.parse_args()
    if a.prepare:prepare()
    elif a.seed:run(a.seed)
    else:p.error('Choose --prepare or --seed')
