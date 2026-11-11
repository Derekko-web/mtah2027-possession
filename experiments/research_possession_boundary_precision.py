"""Frozen boundary screen against the selected new-camera score floors."""
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
from possession_boundary_precision import VARIANTS,refine
from reproduce_boundary_precision import BASE,PROTOCOLS,overlap_summary

ROUND=ROOT/'data/control_boundary_precision_v1'
SOURCES=('experiments/possession_boundary_precision.py','experiments/test_possession_boundary_precision.py',
         'experiments/research_possession_boundary_precision.py','experiments/reproduce_boundary_precision.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    paths=['data/control_remaining_away_v1/broadcast_extension_18935.pkl',
        'data/control_remaining_away_v1/broadcast_extension_19060.pkl','data/advanced_validation/raw.pkl',
        'data/advanced_validation/splits.json','data/control_broadcast_v1/inputs/strict_mapping.npz',
        'evaluation/gt/BAR-ATH_possession_gt_public.csv']
    for seed in (42,43,44):
        for protocol in PROTOCOLS:
            paths += [str((BASE/f'seed{seed}'/protocol/name).relative_to(ROOT)) for name in ('oof_predictions.csv','events.json')]
            paths += [f'data/control_boundary_v1/baseline_confirmation/{protocol}/fast_seed{seed}_probabilities.npy']
    protected=['experiments/SELECTED_MODEL.json','experiments/SELECTED_MODEL.md','experiments/EXPERIMENT_LEDGER.csv',
        'data/candidate_tabpfn_new_camera_v1/candidate.csv','data/candidate_tabpfn_broadcast_short_v1/candidate.csv',
        'data/candidate_tabpfn_short_control_v1/candidate.csv','data/candidate_motion_dead_trim/candidate.csv']
    plan=dict(hypotheses=['Distinct arrival/release padding can remove late contact tails without sacrificing approach context.',
        'Proven new-camera control should use preceding bilateral3m approach evidence to localize onset retrospectively.',
        'Combining both independent proposals may sharpen multiple boundaries while preserving topology.'],
        variants=VARIANTS,seed42_first=True,no_model_fits=True,no_truth_or_protocol_in_inference=True,
        gates='All3screen then9pairs:SpellF1,macroF1,accuracy,all5per-IoU F1,meanbestIoU nondecreasing;retain every previously matched GTspell at every IoU;exactD/X and topology;strictmainprimary/meanIoU gain. Then39foldSpellF1/macro checks.',
        selection='Eligible candidate with largest minimumprimarygain rounded12,then maingain rounded12,then mainmeanIoU gain rounded12,then fewer enabled mechanisms.',
        limitations='Adaptive reuse of7500same-match public labels;no private improvement guarantee.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES},input_sha256={p:digest(ROOT/p) for p in paths},
        protected_sha256={p:digest(ROOT/p) for p in protected})
    with (ROUND/'plan.json').open('x') as f:f.write(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    archive=ROUND/'source_freeze';archive.mkdir()
    for p in SOURCES:(archive/Path(p).name).write_bytes((ROOT/p).read_bytes())


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
    old=np.load(ROOT/'data/control_broadcast_v1/inputs/strict_mapping.npz')['frames'][public]>=0
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    variants=list(VARIANTS) if seed==42 else [json.loads((ROUND/'selection.json').read_text())['selected']]
    rows=[]
    for variant in variants:
        for protocol in PROTOCOLS:
            table=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(table.frame,public)
            events=json.loads((BASE/f'seed{seed}'/protocol/'events.json').read_text())
            fast=np.load(ROOT/f'data/control_boundary_v1/baseline_confirmation/{protocol}/fast_seed{seed}_probabilities.npy')
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            pred,changes=refine(table.prediction.to_numpy(),fast,raw,bc,old,blocks,events,**VARIANTS[variant])
            gt=pm.label_codes(table.ground_truth);before=pm.evaluate(gt,pm.label_codes(table.prediction),bootstrap_n=0)
            after=pm.evaluate(gt,pm.label_codes(pred),bootstrap_n=0)
            a,b=overlap_summary(table.ground_truth,table.prediction),overlap_summary(table.ground_truth,pred)
            checks={k:after[k]>=before[k]-1e-12 for k in ('spell_f1_mean','macro_f1','accuracy')}
            for t in pm.SPELL_IOUS:
                checks[f'IoU{t}']=after['spell_f1'][t]['f1']>=before['spell_f1'][t]['f1']-1e-12
                checks[f'prior_matches_{t}']=bool(b['flags'][t][a['flags'][t]].all())
            checks['mean_best_iou']=b['mean_best_iou']>=a['mean_best_iou']-1e-12
            row=dict(seed=seed,variant=variant,protocol=protocol,spellF1=after['spell_f1_mean'],
                macroF1=after['macro_f1'],accuracy=after['accuracy'],mean_best_iou=b['mean_best_iou'],
                gain=after['spell_f1_mean']-before['spell_f1_mean'],overlap_gain=b['mean_best_iou']-a['mean_best_iou'],
                changed_frames=int(np.count_nonzero(pred!=table.prediction)),checks=checks)
            output=ROUND/'candidates'/variant/f'seed{seed}'/protocol;output.mkdir(parents=True,exist_ok=False)
            pd.DataFrame(dict(frame=public,ground_truth=table.ground_truth,prediction=pred)).to_csv(output/'oof_predictions.csv',index=False)
            write(output/'events.json',changes);write(output/'metrics.json',dict(before=before,after=after,summary=row))
            rows.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame([{k:v for k,v in row.items() if k!='checks'} for row in rows]).to_csv(ROUND/f'seed{seed}_scores.csv',index=False)
    if seed==42:
        eligible=[]
        for variant in variants:
            rr=[r for r in rows if r['variant']==variant]
            if all(all(r['checks'].values()) for r in rr) and rr[0]['gain']>1e-12 and rr[0]['overlap_gain']>1e-12:
                c=VARIANTS[variant]
                eligible.append(dict(variant=variant,minimum_gain=min(r['gain'] for r in rr),main_gain=rr[0]['gain'],
                    overlap_gain=rr[0]['overlap_gain'],mechanisms=int(c['leading'] is not None)+int(c['approach']>0)))
        eligible.sort(key=lambda r:(round(r['minimum_gain'],12),round(r['main_gain'],12),round(r['overlap_gain'],12),-r['mechanisms']),reverse=True)
        result=dict(eligible=eligible,selected=eligible[0]['variant'] if eligible else None)
        write(ROUND/'selection.json',result);print('SELECTION',json.dumps(result),flush=True)
    verify()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44));a=p.parse_args()
    if a.prepare:prepare()
    elif a.seed:run(a.seed)
    else:p.error('Choose --prepare or --seed')
