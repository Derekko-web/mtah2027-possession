"""Freeze and qualify stoppage-fragment suppression against the current recipe."""
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
from reproduce_spurious_possessions import BASE
from reproduce_boundary_precision import PROTOCOLS,overlap_summary
from spurious_stoppage_fragments import refine,VARIANTS

ROUND=ROOT/'data/control_spurious_possessions_v1'
SOURCES=('experiments/spurious_stoppage_fragments.py','experiments/test_spurious_stoppage_fragments.py',
    'experiments/research_spurious_possessions.py','experiments/reproduce_spurious_possessions.py',
    'experiments/reproduce_boundary_precision.py','experiments/possession_boundary_precision.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    inputs=['data/advanced_validation/raw.pkl','data/advanced_validation/splits.json',
        'data/control_remaining_away_v1/broadcast_extension_18935.pkl','evaluation/gt/BAR-ATH_possession_gt_public.csv']
    for seed in (42,43,44):
        for protocol in PROTOCOLS:
            inputs += [str((BASE/f'seed{seed}'/protocol/'oof_predictions.csv').relative_to(ROOT)),
                f'data/control_broadcast_v1/baseline_confirmation/{protocol}/heads_seed{seed}_probabilities.npy']
    protected=['experiments/SELECTED_MODEL.json','experiments/SELECTED_MODEL.md','experiments/EXPERIMENT_LEDGER.csv']
    protected += [f'data/{name}/candidate.csv' for name in ('candidate_tabpfn_home_arrivals_v1',
        'candidate_tabpfn_boundary_precision_v1','candidate_tabpfn_new_camera_v1','candidate_tabpfn_broadcast_short_v1',
        'candidate_tabpfn_short_control_v1','candidate_motion_dead_trim')]
    plan=dict(variants=VARIANTS,seed42_first=True,production_seed=42,
        hypotheses=['Entirely unobserved short owner flips next to D should inherit the opposite live neighbor.',
            'A short run touching existing D without broadcast observations may be a stoppage when weak dead heads favor D and observed tactical ball movement is in the minority.'],
        causal_D_policy='The three extras overlap stoppages. Existing D is retained; adding D is allowed only with no increase in measured false-D frames. The initial red D-exact diagnostic remains preserved; this policy is frozen before candidate scoring.',
        settings='1..125frame original runs;existingDneighbor;zero observed broadcast frames;invisible TC branch to opposite live neighbor;otherwise TCvisibility>=.5,moving>4.5fraction<=.5,maxmeanweakDhead>.5 toD;uniformteams;nochaining.',
        gates='All9SpellF1/macroF1/accuracy/meanbestIoU/fiveIoUF1 nonregress;everypriorGTmatch retained atall5thresholds;extras nonincreasing atall5;33/87homeIoUs nonregress;priorDretained;Xexact;noaddedfalseD;39foldSpellF1/macroF1nonregress;mainzeroextrasall5;strictmainprimarygain.',
        selection='Eligible seed42 methods:maximum minimumprotocol SpellF1gain rounded12,then main gain,then meanmacroF1gain,then variant name ascending;freeze before seeds43/44.',
        limitation='Adaptive development on extensively reused same-match public labels;private effects unknown.',
        new_model_fits=0,new_TabPFN_queries=0,
        source_sha256={p:digest(ROOT/p) for p in SOURCES},input_sha256={p:digest(ROOT/p) for p in inputs},
        protected_sha256={p:digest(ROOT/p) for p in protected})
    write(ROUND/'plan.json',plan);frozen=ROUND/'source_freeze';frozen.mkdir()
    for name in SOURCES:(frozen/Path(name).name).write_bytes((ROOT/name).read_bytes())


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for group in ('source_sha256','input_sha256','protected_sha256'):
        for name,expected in plan[group].items():
            if digest(ROOT/name)!=expected:raise ValueError('Frozen dependency changed: '+name)
    return plan


def pair_checks(table,pred):
    truth=pm.label_codes(table.ground_truth)
    before=pm.evaluate(truth,pm.label_codes(table.prediction),bootstrap_n=0)
    after=pm.evaluate(truth,pm.label_codes(pred),bootstrap_n=0)
    old,new=overlap_summary(table.ground_truth,table.prediction),overlap_summary(table.ground_truth,pred)
    checks={k:after[k]>=before[k]-1e-12 for k in ('spell_f1_mean','macro_f1','accuracy')}
    for t in pm.SPELL_IOUS:
        checks[f'IoU{t}']=after['spell_f1'][t]['f1']>=before['spell_f1'][t]['f1']-1e-12
        checks[f'prior_matches_{t}']=bool(new['flags'][t][old['flags'][t]].all())
        checks[f'extras_{t}']=after['spell_f1'][t]['n_spurious']<=before['spell_f1'][t]['n_spurious']
    for length in (33,87):
        i=next(i for i,(s,e,k) in enumerate(old['gt']) if e-s==length and k==pm.H)
        a=max((v for v,j,_ in old['overlap'] if j==i),default=0.)
        b=max((v for v,j,_ in new['overlap'] if j==i),default=0.)
        checks[f'home{length}']=b>=a-1e-12
    checks.update(mean_best_iou=new['mean_best_iou']>=old['mean_best_iou']-1e-12,
        prior_D_retained=bool(np.all(pred[table.prediction=='D']=='D')),
        X_exact=bool(np.array_equal(pred=='X',table.prediction=='X')),
        no_new_false_D=bool(np.sum((pred=='D')&(table.ground_truth!='D'))<=np.sum((table.prediction=='D')&(table.ground_truth!='D'))))
    return before,after,checks


def run(seed):
    verify();public=np.arange(11185,18685)
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    variants=list(VARIANTS) if seed==42 else [json.loads((ROUND/'selection.json').read_text())['selected']]
    if variants==[None]:raise ValueError('No eligible seed42 method')
    rows=[];fold_rows=[]
    for variant in variants:
        for protocol in PROTOCOLS:
            table=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(table.frame,public)
            heads=np.load(ROOT/f'data/control_broadcast_v1/baseline_confirmation/{protocol}/heads_seed{seed}_probabilities.npy')
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            pred,events=refine(table.prediction.to_numpy(),heads,raw,bc,blocks,**VARIANTS[variant])
            before,after,checks=pair_checks(table,pred)
            for block,(a,b) in enumerate(blocks,1):
                gt=pm.label_codes(table.ground_truth.iloc[a:b])
                v0=pm.evaluate(gt,pm.label_codes(table.prediction.iloc[a:b]),bootstrap_n=0)
                v1=pm.evaluate(gt,pm.label_codes(pred[a:b]),bootstrap_n=0)
                ok=v1['spell_f1_mean']>=v0['spell_f1_mean']-1e-12 and v1['macro_f1']>=v0['macro_f1']-1e-12
                checks[f'fold{block}']=ok
                fold_rows.append(dict(seed=seed,variant=variant,protocol=protocol,block=block,
                    before_spellF1=v0['spell_f1_mean'],spellF1=v1['spell_f1_mean'],before_macroF1=v0['macro_f1'],macroF1=v1['macro_f1'],passed=ok))
            row=dict(seed=seed,variant=variant,protocol=protocol,spellF1=after['spell_f1_mean'],
                before_spellF1=before['spell_f1_mean'],macroF1=after['macro_f1'],macro_gain=after['macro_f1']-before['macro_f1'],
                accuracy=after['accuracy'],gain=after['spell_f1_mean']-before['spell_f1_mean'],
                spurious=after['spell_f1'][.5]['n_spurious'],no_extras_all_thresholds=all(after['spell_f1'][t]['n_spurious']==0 for t in pm.SPELL_IOUS),
                changed_frames=int(np.sum(pred!=table.prediction)),checks=checks)
            folder=ROUND/'candidates'/variant/f'seed{seed}'/protocol;folder.mkdir(parents=True,exist_ok=False)
            pd.DataFrame(dict(frame=public,ground_truth=table.ground_truth,prediction=pred)).to_csv(folder/'oof_predictions.csv',index=False)
            write(folder/'events.json',events);write(folder/'metrics.json',dict(before=before,after=after,summary=row))
            rows.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='checks'} for r in rows]).to_csv(ROUND/f'seed{seed}_scores.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(ROUND/f'seed{seed}_fold_scores.csv',index=False)
    if seed==42:
        eligible=[]
        for variant in variants:
            values=[r for r in rows if r['variant']==variant]
            if all(all(r['checks'].values()) for r in values) and values[0]['gain']>1e-12 and values[0]['no_extras_all_thresholds']:
                eligible.append(dict(variant=variant,minimum_gain=min(r['gain'] for r in values),
                    main_gain=values[0]['gain'],mean_macro_gain=float(np.mean([r['macro_gain'] for r in values]))))
        eligible.sort(key=lambda r:(-round(r['minimum_gain'],12),-round(r['main_gain'],12),-round(r['mean_macro_gain'],12),r['variant']))
        write(ROUND/'selection.json',dict(eligible=eligible,selected=eligible[0]['variant'] if eligible else None))
    verify()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44))
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.seed:run(a.seed)
    else:p.error('Choose --prepare or --seed')
