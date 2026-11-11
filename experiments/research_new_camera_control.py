"""Frozen qualification of newly observed bilateral contact recovery."""
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
from new_camera_control import recover
from reproduce_remaining_away import BASE,PROTOCOLS,evaluate

ROUND=ROOT/'data/control_remaining_away_v1'
SOURCES=('experiments/new_camera_control.py','experiments/test_new_camera_control.py',
         'experiments/research_new_camera_control.py','experiments/reproduce_remaining_away.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    inputs=['data/control_remaining_away_v1/observation_plan.json',
        'data/control_remaining_away_v1/broadcast_extension_18935.pkl','data/control_remaining_away_v1/broadcast_extension_19060.pkl',
        'data/advanced_validation/raw.pkl','data/advanced_validation/splits.json',
        'data/control_pair_motion_v1/inputs/motion_info.pkl','data/control_broadcast_v1/inputs/strict_mapping.npz',
        'evaluation/gt/BAR-ATH_possession_gt_public.csv']
    inputs += [str((BASE/f'seed{s}'/p/'oof_predictions.csv').relative_to(ROOT)) for s in (42,43,44) for p in PROTOCOLS]
    inputs += [f'data/control_boundary_v1/baseline_confirmation/{p}/fast_seed{s}_probabilities.npy' for s in (42,43,44) for p in PROTOCOLS]
    protected=['experiments/SELECTED_MODEL.json','experiments/SELECTED_MODEL.md','experiments/EXPERIMENT_LEDGER.csv',
               'data/candidate_tabpfn_broadcast_short_v1/candidate.csv','data/candidate_tabpfn_short_control_v1/candidate.csv',
               'data/candidate_motion_dead_trim/candidate.csv']
    plan=dict(hypothesis='Newly observed broadcast contacts distinguish otherwise ambiguous tactical-only touches; restrict new ownership persistence to intervals entirely absent from the old broadcast path.',
        variants=[13,25],seed42_first=True,source='Unchanged frozen Round25sub-grid extension mapping;no threshold relaxation or new alignment method',
        inference='Both cameras supply3exclusive contacts within13frames;fast majority>.5 and weak old-owner kinematic kernel<.5;oldcamera missing throughout;newcamera fully observed;oldowner release confirmed13or25frames with kernel>=.5 and fast>=.65;max125frames;longreference>250;D/Xexact.',
        gates='All3screen and then all9paired aggregate SpellF1,macroF1,accuracy and all5per-IoU F1 scores nondecreasing;allprevious IoU.5GT matches retained;71frameAway matched IoU.5 in everypair;mainSpellF1 strictly higher.',
        selection='Among complete candidates prefer minimumprimarygain rounded12,then main gain rounded12,then13confirmationframes.',
        no_fits=True,no_manual_output_patches=True,no_targets_in_inference=True,
        source_sha256={p:digest(ROOT/p) for p in SOURCES},input_sha256={p:digest(ROOT/p) for p in inputs},
        protected_sha256={p:digest(ROOT/p) for p in protected})
    with (ROUND/'plan.json').open('x') as f:f.write(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    source=ROUND/'source_freeze';source.mkdir()
    for p in SOURCES:(source/Path(p).name).write_bytes((ROOT/p).read_bytes())


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for group in ('source_sha256','input_sha256','protected_sha256'):
        for name,sha in plan[group].items():
            if digest(ROOT/name)!=sha:raise ValueError('Frozen dependency changed: '+name)
    return plan


def run(seed):
    verify();public=np.arange(11185,18685)
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    motion=pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROUND/'broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    old=np.load(ROOT/'data/control_broadcast_v1/inputs/strict_mapping.npz')['frames'][public]>=0
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    variants=[13,25] if seed==42 else [json.loads((ROUND/'selection.json').read_text())['stable_frames']]
    rows=[]
    for stable in variants:
        for protocol in PROTOCOLS:
            baseline=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(baseline.frame,public)
            fast=np.load(ROOT/f'data/control_boundary_v1/baseline_confirmation/{protocol}/fast_seed{seed}_probabilities.npy')
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            pred,events=recover(baseline.prediction.to_numpy(),fast,raw,motion,bc,old,blocks,stable)
            truth=pm.label_codes(baseline.ground_truth)
            before=pm.evaluate(truth,pm.label_codes(baseline.prediction),bootstrap_n=0)
            after=pm.evaluate(truth,pm.label_codes(pred),bootstrap_n=0)
            runs=pm.valid_runs(truth);gt=pm.possession_spells(truth,runs)
            target=next(i for i,(a,b,k) in enumerate(gt) if b-a==71 and k==pm.A)
            prior=pm.spell_overlaps(gt,pm.possession_spells(pm.label_codes(baseline.prediction),runs))
            overlap=pm.spell_overlaps(gt,pm.possession_spells(pm.label_codes(pred),runs))
            old_flags=pm.matched_gt_flags(prior,len(gt),.5);new_flags=pm.matched_gt_flags(overlap,len(gt),.5)
            checks={k:after[k]>=before[k]-1e-12 for k in ('spell_f1_mean','macro_f1','accuracy')}
            checks.update({f'IoU{t}':after['spell_f1'][t]['f1']>=before['spell_f1'][t]['f1']-1e-12 for t in pm.SPELL_IOUS})
            checks.update(target_matched=bool(new_flags[target]),previous_matches_retained=bool(new_flags[old_flags].all()))
            row=dict(seed=seed,stable_frames=stable,protocol=protocol,before_spellF1=before['spell_f1_mean'],spellF1=after['spell_f1_mean'],
                     macroF1=after['macro_f1'],accuracy=after['accuracy'],gain=after['spell_f1_mean']-before['spell_f1_mean'],
                     target_iou=max((v for v,i,_ in overlap if i==target),default=0.),events=len(events),changed_frames=int(np.count_nonzero(pred!=baseline.prediction)),checks=checks)
            output=ROUND/'candidates'/f'stable{stable}'/f'seed{seed}'/protocol;output.mkdir(parents=True,exist_ok=False)
            pd.DataFrame(dict(frame=public,ground_truth=baseline.ground_truth,prediction=pred)).to_csv(output/'oof_predictions.csv',index=False)
            write(output/'events.json',events);write(output/'metrics.json',dict(before=before,after=after,summary=row))
            rows.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame([{k:v for k,v in row.items() if k!='checks'} for row in rows]).to_csv(ROUND/f'seed{seed}_scores.csv',index=False)
    if seed==42:
        eligible=[]
        for stable in variants:
            selected=[r for r in rows if r['stable_frames']==stable]
            if all(all(r['checks'].values()) for r in selected) and selected[0]['gain']>1e-12:
                eligible.append(dict(stable_frames=stable,minimum_gain=min(r['gain'] for r in selected),main_gain=selected[0]['gain']))
        eligible.sort(key=lambda r:(round(r['minimum_gain'],12),round(r['main_gain'],12),r['stable_frames']==13),reverse=True)
        result=dict(eligible=eligible,stable_frames=eligible[0]['stable_frames'] if eligible else None)
        write(ROUND/'selection.json',result);print('SELECTION',json.dumps(result),flush=True)
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--seed',type=int,choices=(42,43,44));a=parser.parse_args()
    if a.prepare:prepare()
    elif a.seed:run(a.seed)
    else:parser.error('Choose --prepare or --seed')
