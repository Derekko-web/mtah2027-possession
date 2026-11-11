"""Evaluate frozen observed-contact envelopes on current TabPFN predictions."""
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
from broadcast_short_envelope import refine
from reproduce_short_possessions import BASE,PROTOCOLS

ROUND=ROOT/'data/control_broadcast_short_envelope_v1'
SOURCES=('experiments/broadcast_short_envelope.py','experiments/test_broadcast_short_envelope.py',
         'experiments/research_broadcast_short_envelope.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    ROUND.mkdir(exist_ok=False)
    paths=[ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl',ROOT/'data/advanced_validation/splits.json',
           ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv']
    paths += [BASE/stage/p/'oof_predictions.csv' for stage in ('screen','confirmation_seed43','confirmation_seed44') for p in PROTOCOLS]
    protected=[ROOT/'experiments'/name for name in ('SELECTED_MODEL.json','SELECTED_MODEL.md','EXPERIMENT_LEDGER.csv')]
    protected += [ROOT/'data/candidate_tabpfn_short_control_v1/candidate.csv',ROOT/'data/candidate_motion_dead_trim/candidate.csv']
    plan=dict(hypothesis='Independent broadcast-ball contact can bound existing short islands where tactical ball evidence disagrees.',
        variants=[13,5],rule='Fully observed opposite-live-bracketed8..250-frame reference islands;3exclusive same-camera contacts(distance<2m,margin>.5m);intersect contact envelope plus5or13framepadding;maximum25frame inward edge movement;original inputs only.',
        score_gates='All three seed42 SpellF1,macroF1,accuracy nondecreasing;main SpellF1 and short IoU.5 matches strictly improve. Then one frozen finalist must pass all9 seed/protocol pairs. Full success would recover all6mainshort spells; partial progress recorded.',
        no_new_fits=True,targets_used_only_for_scoring=True,
        source_sha256={p:digest(ROOT/p) for p in SOURCES},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        protected_sha256={str(p.relative_to(ROOT)):digest(p) for p in protected})
    write(ROUND/'plan.json',plan)
    archive=ROUND/'source_freeze';archive.mkdir()
    for name in SOURCES:(archive/Path(name).name).write_bytes((ROOT/name).read_bytes())


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for key in ('source_sha256','input_sha256','protected_sha256'):
        for name,sha in plan[key].items():
            if digest(ROOT/name)!=sha:raise ValueError('Frozen dependency changed: '+name)
    return plan


def run(seed):
    verify();public=np.arange(11185,18685)
    broadcast=pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[public].reset_index(drop=True)
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    stage='screen' if seed==42 else f'confirmation_seed{seed}'
    variants=(13,5) if seed==42 else (json.loads((ROUND/'selection.json').read_text())['padding'],)
    rows=[]
    for padding in variants:
        for protocol in PROTOCOLS:
            table=pd.read_csv(BASE/stage/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(table.frame,public)
            ref=table[f'tabpfn_short_control_seed{seed}'].to_numpy()
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            result,events=refine(ref,broadcast,blocks,padding)
            truth=pm.label_codes(table.ground_truth);before=pm.evaluate(truth,pm.label_codes(ref),bootstrap_n=0)
            after=pm.evaluate(truth,pm.label_codes(result),bootstrap_n=0)
            runs=pm.valid_runs(truth);spells=pm.possession_spells(truth,runs)
            overlap=pm.spell_overlaps(spells,pm.possession_spells(pm.label_codes(result),runs))
            flags=pm.matched_gt_flags(overlap,len(spells),.5);short=[i for i,(a,b,_) in enumerate(spells) if b-a<=125]
            checks={key:after[key]>=before[key]-1e-12 for key in ('spell_f1_mean','macro_f1','accuracy')}
            row=dict(seed=seed,padding=padding,protocol=protocol,before_spellF1=before['spell_f1_mean'],spellF1=after['spell_f1_mean'],
                spellF1_delta=after['spell_f1_mean']-before['spell_f1_mean'],macroF1=after['macro_f1'],accuracy=after['accuracy'],
                short_matches=int(flags[short].sum()),spurious_iou50=after['spell_f1'][.5]['n_spurious'],
                changed_frames=int((ref!=result).sum()),checks=checks)
            output=ROUND/'candidates'/f'pad{padding}'/f'seed{seed}'/protocol;output.mkdir(parents=True,exist_ok=False)
            pd.DataFrame(dict(frame=public,ground_truth=table.ground_truth,prediction=result)).to_csv(output/'oof_predictions.csv',index=False)
            write(output/'events.json',events);write(output/'metrics.json',dict(before=before,after=after,summary=row))
            rows.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='checks'} for r in rows]).to_csv(ROUND/f'seed{seed}_scores.csv',index=False)
    if seed==42:
        eligible=[]
        for padding in variants:
            relevant=[r for r in rows if r['padding']==padding];main=relevant[0]
            if all(all(r['checks'].values()) for r in relevant) and main['spellF1_delta']>0 and main['short_matches']>3:
                eligible.append(dict(padding=padding,minimum_gain=min(r['spellF1_delta'] for r in relevant),main_gain=main['spellF1_delta'],short_matches=main['short_matches']))
        eligible.sort(key=lambda r:(r['minimum_gain'],r['main_gain'],r['padding']),reverse=True)
        selection=dict(eligible=eligible)
        if eligible:selection.update(eligible[0])
        write(ROUND/'selection.json',selection);print('SELECTION',json.dumps(selection),flush=True)
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--seed',type=int,choices=(42,43,44));args=parser.parse_args()
    if args.prepare:prepare()
    elif args.seed:run(args.seed)
    else:parser.error('Choose --prepare or --seed')
