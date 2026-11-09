"""Frozen physical/continuity repair screen against the current selected recipe."""
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
from reproduce_holdout_extras import BASE,pair_checks
from reproduce_boundary_precision import PROTOCOLS
from holdout_possession_repair import refine,VARIANTS
ROUND=ROOT/'data/control_holdout_extras_v1'
SOURCES=('experiments/holdout_possession_repair.py','experiments/test_holdout_possession_repair.py',
    'experiments/research_holdout_extras.py','experiments/reproduce_holdout_extras.py',
    'experiments/dual_anchor_stoppage.py','experiments/stoppage_gap_closure.py',
    'experiments/observed_gap_stoppage.py','experiments/spurious_stoppage_fragments.py',
    'experiments/possession_boundary_precision.py','experiments/reproduce_boundary_precision.py',
    'evaluation/possession_metrics.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as handle:handle.write(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    inputs=['data/advanced_validation/raw.pkl','data/advanced_validation/splits.json',
        'data/control_remaining_away_v1/broadcast_extension_18935.pkl','evaluation/gt/BAR-ATH_possession_gt_public.csv']
    for seed,stage in ((42,'screen'),(43,'confirmation_seed43'),(44,'confirmation_seed44')):
        for protocol in PROTOCOLS:
            inputs += [str((BASE/f'seed{seed}'/protocol/'oof_predictions.csv').relative_to(ROOT)),
                f'data/control_broadcast_v1/baseline_confirmation/{protocol}/heads_seed{seed}_probabilities.npy',
                f'data/control_boundary_v1/baseline_confirmation/{protocol}/fast_seed{seed}_probabilities.npy',
                f'data/control_tabpfn_fixed_confirmation_v2/{stage}/{protocol}/owner_probability.npy']
    protected=['experiments/SELECTED_MODEL.json','experiments/SELECTED_MODEL.md','experiments/EXPERIMENT_LEDGER.csv']
    protected += [f'data/{p}/candidate.csv' for p in ('candidate_tabpfn_stoppage_fragments_v1','candidate_tabpfn_home_arrivals_v1',
        'candidate_tabpfn_boundary_precision_v1','candidate_tabpfn_new_camera_v1','candidate_tabpfn_broadcast_short_v1',
        'candidate_tabpfn_short_control_v1','candidate_motion_dead_trim')]
    plan=dict(variants=VARIANTS,production_seed=42,seed42_first=True,
        hypotheses=['Three consecutive outside-pitch observations establish a stoppage;unsupported live flips before it can continue the previous owner.',
            'A short D gap between same-owner moving control is a possible false split;an unseen handoff beside D can use existing anchors with a5frame dead-edge allowance.',
            'Disconnected contact bouts and cross-source disagreement identify overlong short-possession boundaries.',
            'Broadcast-supported short control may extend into premature D or replace an unsupported opponent/trace/D tail.'],
        settings='Source-fixed rules with existing3/5/8/13/25/125/250frame contact,edge and duration constants; no truth/frame/protocol input;all proposals read original reference.',
        dead_policy='Allow D growth/shrinkage. Require D recall nonregression and no net increase of false-D count in every labelled pair;per-frame exact D would prevent fixing true possessions split by falseD.',
        gates='All9 SpellF1/macroF1/accuracy/fiveIoUF1 nonregress;priorGTmatches andall153individualGTIoUs retained;extras nonincreasingall5;Xexact;Drecallnonregress;no netnewfalseD;39foldprimary/macro nonregress;allholdoutzeroextrasall5;seed42purged/longerstrictprimarygain;main1retained.',
        selection='Eligible seed42 methods:greatest minimum gain on purged/longer,then mean macro gain,then variant name; frozen before other seeds.',
        limitation='Extensively adaptive same-match development;all seeds have informed historical work;not independent/private confirmation.',
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


def run(seed):
    verify();public=np.arange(11185,18685)
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    variants=list(VARIANTS) if seed==42 else [json.loads((ROUND/'selection.json').read_text())['selected']]
    if variants==[None]:raise ValueError('No complete eligible method')
    rows=[];folds=[];stage='screen' if seed==42 else f'confirmation_seed{seed}'
    for variant in variants:
        for protocol in PROTOCOLS:
            table=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv');np.testing.assert_array_equal(table.frame,public)
            heads=np.load(ROOT/f'data/control_broadcast_v1/baseline_confirmation/{protocol}/heads_seed{seed}_probabilities.npy')
            fast=np.load(ROOT/f'data/control_boundary_v1/baseline_confirmation/{protocol}/fast_seed{seed}_probabilities.npy')
            owner=np.load(ROOT/f'data/control_tabpfn_fixed_confirmation_v2/{stage}/{protocol}/owner_probability.npy')
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            pred,events=refine(table.prediction.to_numpy(),heads,owner,fast,raw,bc,blocks,**VARIANTS[variant])
            before,after,checks=pair_checks(table,pred)
            for block,(a,b) in enumerate(blocks,1):
                g=pm.label_codes(table.ground_truth.iloc[a:b]);v0=pm.evaluate(g,pm.label_codes(table.prediction.iloc[a:b]),bootstrap_n=0)
                v1=pm.evaluate(g,pm.label_codes(pred[a:b]),bootstrap_n=0)
                ok=v1['spell_f1_mean']>=v0['spell_f1_mean']-1e-12 and v1['macro_f1']>=v0['macro_f1']-1e-12
                checks[f'fold{block}']=ok
                folds.append(dict(seed=seed,variant=variant,protocol=protocol,block=block,before_spellF1=v0['spell_f1_mean'],
                    spellF1=v1['spell_f1_mean'],before_macroF1=v0['macro_f1'],macroF1=v1['macro_f1'],passed=ok))
            row=dict(seed=seed,variant=variant,protocol=protocol,before_spellF1=before['spell_f1_mean'],spellF1=after['spell_f1_mean'],
                macroF1=after['macro_f1'],accuracy=after['accuracy'],gain=after['spell_f1_mean']-before['spell_f1_mean'],
                macro_gain=after['macro_f1']-before['macro_f1'],spurious=after['spell_f1'][.5]['n_spurious'],
                zero_extras=all(after['spell_f1'][t]['n_spurious']==0 for t in pm.SPELL_IOUS),
                changed_frames=int(np.sum(pred!=table.prediction)),checks=checks)
            folder=ROUND/'candidates'/variant/f'seed{seed}'/protocol;folder.mkdir(parents=True,exist_ok=False)
            pd.DataFrame(dict(frame=public,ground_truth=table.ground_truth,prediction=pred)).to_csv(folder/'oof_predictions.csv',index=False)
            write(folder/'events.json',events);write(folder/'metrics.json',dict(before=before,after=after,summary=row));rows.append(row)
            print(json.dumps(row),flush=True)
    pd.DataFrame([{k:v for k,v in row.items() if k!='checks'} for row in rows]).to_csv(ROUND/f'seed{seed}_scores.csv',index=False)
    pd.DataFrame(folds).to_csv(ROUND/f'seed{seed}_fold_scores.csv',index=False)
    if seed==42:
        eligible=[]
        for variant in variants:
            values=[r for r in rows if r['variant']==variant]
            if (all(all(r['checks'].values()) and r['zero_extras'] for r in values)
                and all(r['gain']>1e-12 for r in values[1:])):
                eligible.append(dict(variant=variant,minimum_holdout_gain=min(r['gain'] for r in values[1:]),
                    mean_macro_gain=float(np.mean([r['macro_gain'] for r in values]))))
        eligible.sort(key=lambda r:(-round(r['minimum_holdout_gain'],12),-round(r['mean_macro_gain'],12),r['variant']))
        write(ROUND/'selection.json',dict(eligible=eligible,selected=eligible[0]['variant'] if eligible else None))
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--seed',type=int,choices=(42,43,44));args=parser.parse_args()
    if args.prepare:prepare()
    elif args.seed:run(args.seed)
    else:parser.error('Choose --prepare or --seed')
