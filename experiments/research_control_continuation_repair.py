"""Qualification of broadcast trimming and optional contact continuation."""
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
from control_continuation_repair import repair
from reproduce_short_possessions import BASE,PROTOCOLS

ROUND=ROOT/'data/control_short_continuation_v1'
SOURCES=('experiments/broadcast_short_envelope.py','experiments/control_continuation_repair.py',
         'experiments/test_broadcast_short_envelope.py','experiments/test_control_continuation_repair.py',
         'experiments/research_control_continuation_repair.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    ROUND.mkdir(exist_ok=False)
    paths=[ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl',ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl',
           ROOT/'data/advanced_validation/raw.pkl',ROOT/'data/advanced_validation/splits.json',
           ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',
           ROOT/'data/control_broadcast_short_envelope_v1/plan.json',ROOT/'data/control_broadcast_short_envelope_v1/seed42_scores.csv']
    paths += [BASE/stage/p/'oof_predictions.csv' for stage in ('screen','confirmation_seed43','confirmation_seed44') for p in PROTOCOLS]
    paths += [ROOT/'data/control_boundary_v1/baseline_confirmation'/p/f'fast_seed{s}_probabilities.npy' for s in (42,43,44) for p in PROTOCOLS]
    protected=[ROOT/'experiments'/name for name in ('SELECTED_MODEL.json','SELECTED_MODEL.md','EXPERIMENT_LEDGER.csv')]
    protected += [ROOT/'data/candidate_tabpfn_short_control_v1/candidate.csv',ROOT/'data/candidate_motion_dead_trim/candidate.csv']
    plan=dict(hypothesis='A supported brief foreign contact is suppressed while the old owner has no coherent ball-relative motion; wait for stable kinematic reacquisition rather than immediate nearest-player reassignment.',
        variants=['broadcast13','continuation13','continuation25'],broadcast_padding=13,
        prior_ranking_repair='Original pad5 selection was caused by1.1e-16difference in minimum SpellF1 gains. Compare rounded12decimal scores before main gain; pad13 then dominates on SpellF1 and short recovery. Original selection and source remain untouched.',
        continuation='At least3 fast>=.65 exclusive foreign contacts within13frames;reference-owner kernel<.5;onlylong>250frame homogeneous live reference runs;newowner persists until13or25consecutive old-owner kernel>=.5 and fast>=.65,within125frames;no timeouts or D/X edits.',
        gates='All9matched SpellF1,macroF1,accuracy nondecreasing;main strictly higher;short-IoU.5matchcounts not below original across9. Prefer complete6/6main,then minimum gain rounded12,then main gain rounded12,then simplerbroadcast-only.',
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        source_sha256={p:digest(ROOT/p) for p in SOURCES},
        protected_sha256={str(p.relative_to(ROOT)):digest(p) for p in protected})
    write(ROUND/'plan.json',plan);archive=ROUND/'source_freeze';archive.mkdir()
    for name in SOURCES:(archive/Path(name).name).write_bytes((ROOT/name).read_bytes())


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for key in ('source_sha256','input_sha256','protected_sha256'):
        for name,sha in plan[key].items():
            if digest(ROOT/name)!=sha:raise ValueError('Frozen dependency changed: '+name)
    return plan


def evaluate(truth,prediction):
    g=pm.label_codes(truth);r=pm.evaluate(g,pm.label_codes(prediction),bootstrap_n=0)
    runs=pm.valid_runs(g);spells=pm.possession_spells(g,runs)
    overlap=pm.spell_overlaps(spells,pm.possession_spells(pm.label_codes(prediction),runs))
    flags=pm.matched_gt_flags(overlap,len(spells),.5)
    short=[i for i,(a,b,_) in enumerate(spells) if b-a<=125]
    return r,int(flags[short].sum())


def run(seed):
    verify();public=np.arange(11185,18685)
    broadcast=pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[public].reset_index(drop=True)
    motion=pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[public].reset_index(drop=True)
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    stage='screen' if seed==42 else f'confirmation_seed{seed}'
    variants=['broadcast13','continuation13','continuation25'] if seed==42 else [json.loads((ROUND/'selection.json').read_text())['selected']]
    rows=[]
    for variant in variants:
        for protocol in PROTOCOLS:
            table=pd.read_csv(BASE/stage/protocol/'oof_predictions.csv');np.testing.assert_array_equal(table.frame,public)
            ref=table[f'tabpfn_short_control_seed{seed}'].to_numpy()
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            result,bc_events=refine(ref,broadcast,blocks,13);events=[]
            if variant!='broadcast13':
                fast=np.load(ROOT/'data/control_boundary_v1/baseline_confirmation'/protocol/f'fast_seed{seed}_probabilities.npy')
                result,events=repair(result,fast,raw,motion,blocks,int(variant.removeprefix('continuation')))
            before,before_short=evaluate(table.ground_truth,ref);after,short=evaluate(table.ground_truth,result)
            checks={key:after[key]>=before[key]-1e-12 for key in ('spell_f1_mean','macro_f1','accuracy')}
            checks['short_matches']=short>=before_short
            row=dict(seed=seed,variant=variant,protocol=protocol,before_spellF1=before['spell_f1_mean'],spellF1=after['spell_f1_mean'],
                spellF1_delta=after['spell_f1_mean']-before['spell_f1_mean'],macroF1=after['macro_f1'],accuracy=after['accuracy'],
                short_matches=short,spurious_iou50=after['spell_f1'][.5]['n_spurious'],changed_frames=int((ref!=result).sum()),checks=checks)
            output=ROUND/'candidates'/variant/f'seed{seed}'/protocol;output.mkdir(parents=True,exist_ok=False)
            pd.DataFrame(dict(frame=public,ground_truth=table.ground_truth,prediction=result)).to_csv(output/'oof_predictions.csv',index=False)
            write(output/'events.json',dict(broadcast=bc_events,continuation=events))
            write(output/'metrics.json',dict(before=before,after=after,summary=row));rows.append(row)
            print(json.dumps(row),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='checks'} for r in rows]).to_csv(ROUND/f'seed{seed}_scores.csv',index=False)
    if seed==42:
        eligible=[]
        for variant in variants:
            relevant=[r for r in rows if r['variant']==variant];main=relevant[0]
            if all(all(r['checks'].values()) for r in relevant) and main['spellF1_delta']>0 and main['short_matches']>3:
                eligible.append(dict(variant=variant,minimum_gain=min(r['spellF1_delta'] for r in relevant),
                                     main_gain=main['spellF1_delta'],short_matches=main['short_matches']))
        eligible.sort(key=lambda r:(r['short_matches']==6,round(r['minimum_gain'],12),round(r['main_gain'],12),r['variant']=='broadcast13'),reverse=True)
        selection=dict(eligible=eligible,selected=eligible[0]['variant'] if eligible else None)
        write(ROUND/'selection.json',selection);print('SELECTION',json.dumps(selection),flush=True)
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--seed',type=int,choices=(42,43,44));args=parser.parse_args()
    if args.prepare:prepare()
    elif args.seed:run(args.seed)
    else:parser.error('Choose --prepare or --seed')
