"""Frozen bounded screen for short-control repair of the selected TabPFN."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from player_representation import COLUMNS, interpolate
from short_possession_repair import HEADS, features, fit, predict, repair, training_record
from reproduce_short_possessions import PROTOCOLS, BASE

ROUND=ROOT/'data/control_short_repair_v1'
SOURCES=('experiments/short_possession_repair.py','experiments/test_short_possession_repair.py',
         'experiments/research_short_possession_repair.py','experiments/reproduce_short_possessions.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    enc=ROOT/'data/control_player_representation_v1/encodings'
    paths=[ROOT/'data/advanced_validation'/name for name in ('raw.pkl','ordered.pkl','splits.json')]
    paths += [enc/name for name in ('frames.npy','pretrained_latent.npy','eligible.npy')]
    paths += [ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv']
    protected=[ROOT/'experiments'/name for name in ('SELECTED_MODEL.json','SELECTED_MODEL.md','EXPERIMENT_LEDGER.csv')]
    protected += [ROOT/'data/candidate_tabpfn_short_control_v1/candidate.csv',ROOT/'data/candidate_motion_dead_trim/candidate.csv']
    for stage in ('screen','confirmation_seed43','confirmation_seed44'):
        for protocol in PROTOCOLS:
            paths += [BASE/stage/protocol/name for name in ('oof_predictions.csv','owner_probability.npy')]
    plan=dict(purpose='Recover remaining short possessions with no matched current-score regression',
        hypothesis='Duration-balanced conditional ownership with finer observation taps can reveal short events absent from the current probability path.',
        heads=HEADS,decoders=['insert','refine','insert_refine'],seed42_screen_first=True,
        seeds=[42,43,44],protocols=list(PROTOCOLS),fixed_parameters=True,
        inference='General label-free proposal rule,11-frame smoothing,8..125frames,confidence.65,support3,endpoint radius25; no frame or protocol exceptions',
        training='Outer-owntrain H/A only; paired team augmentation; inverse sqrt or inverse clipped contiguous-spell duration weights; no validation calibration',
        gates='Every protocol SpellF1,macroF1,accuracy nondecreasing; exactD/X; main strictly higher and all6main short matches for complete success. Partial gains reported separately.',
        limitations='Adaptive repeated same-match public development; no independent private accuracy claim.',
        source_sha256={p:digest(ROOT/p) for p in SOURCES},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        protected_sha256={str(p.relative_to(ROOT)):digest(p) for p in protected})
    path=ROUND/'plan.json'
    with path.open('x') as handle:handle.write(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    archive=ROUND/'source_freeze';archive.mkdir()
    for name in SOURCES:(archive/Path(name).name).write_bytes((ROOT/name).read_bytes())
    print('Frozen two training objectives and three bounded decoder variants before native fits.',flush=True)


def verify():
    plan=json.loads((ROUND/'plan.json').read_text())
    for identities in ('source_sha256','input_sha256','protected_sha256'):
        for name,sha in plan[identities].items():
            if digest(ROOT/name)!=sha:raise ValueError('Frozen file changed: '+name)
    return plan


def inputs():
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl')
    ordered=pd.read_pickle(ROOT/'data/advanced_validation/ordered.pkl')
    enc=ROOT/'data/control_player_representation_v1/encodings'
    latent,_=interpolate(np.load(enc/'frames.npy'),np.load(enc/'pretrained_latent.npy'),np.load(enc/'eligible.npy'),len(raw))
    base=ordered.join(pd.DataFrame(latent,columns=COLUMNS,index=ordered.index))
    return raw,features(base,raw)


def run(seed):
    verify();raw,x=inputs();n=len(raw);public=np.arange(11185,18685)
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    stage='screen' if seed==42 else f'confirmation_seed{seed}'
    if seed==42:
        methods=[(h,m) for h in HEADS for m in ('insert','refine','insert_refine')]
    else:
        selected=json.loads((ROUND/'screen_selection.json').read_text())
        if not selected['eligible']:raise ValueError('No nonregressing seed42 candidate')
        methods=[(selected['head'],selected['mode'])]
    scores=[]
    for protocol in PROTOCOLS:
        baseline=pd.read_csv(BASE/stage/protocol/'oof_predictions.csv')
        reference=np.full(n,'X',dtype='<U1');reference[public]=baseline[f'tabpfn_short_control_seed{seed}']
        owner=np.full((n,2),np.nan);owner[public]=np.load(BASE/stage/protocol/'owner_probability.npy')
        folds=[s for s in splits if s['protocol']==protocol]
        blocks=[(s['validation'][0],s['validation'][-1]+1) for s in folds]
        for head in sorted({h for h,_ in methods}):
            folder=ROUND/f'seed{seed}'/head/protocol;folder.mkdir(parents=True,exist_ok=False)
            probability=np.full((n,2),np.nan)
            for s in folds:
                train,validation=np.asarray(s['train']),np.asarray(s['validation'])
                record=training_record(labels,train,HEADS[head])
                masked=np.full_like(labels,'X');masked[train]=labels[train]
                alternate=training_record(masked,train,HEADS[head])
                for key in record:np.testing.assert_array_equal(record[key],alternate[key])
                assert not np.intersect1d(train,validation).size
                print('Fit',head,'seed',seed,protocol,'block',s['block'],flush=True)
                model,record=fit(x,labels,train,head,seed)
                path=folder/f'block{s["block"]}.cbm';model.save_model(path)
                np.savez_compressed(folder/f'block{s["block"]}_training.npz',**record,validation=validation)
                value=predict(model,x.iloc[validation]);loaded=CatBoostClassifier();loaded.load_model(path)
                np.testing.assert_array_equal(value,predict(loaded,x.iloc[validation]))
                probability[validation]=value
            np.save(folder/'probability.npy',probability[public])
            before=pm.evaluate(pm.label_codes(baseline.ground_truth),pm.label_codes(reference[public]),bootstrap_n=0)
            for _,mode in [(h,m) for h,m in methods if h==head]:
                output=ROUND/'candidates'/f'{head}_{mode}'/f'seed{seed}'/protocol
                output.mkdir(parents=True,exist_ok=False)
                prediction,events=repair(reference,probability,owner,raw,blocks,mode)
                table=pd.DataFrame(dict(frame=public,ground_truth=labels[public],prediction=prediction[public]))
                table.to_csv(output/'oof_predictions.csv',index=False);write(output/'events.json',events)
                after=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(table.prediction),bootstrap_n=0)
                gt=pm.label_codes(table.ground_truth);runs=pm.valid_runs(gt);spells=pm.possession_spells(gt,runs)
                overlap=pm.spell_overlaps(spells,pm.possession_spells(pm.label_codes(table.prediction),runs))
                flags=pm.matched_gt_flags(overlap,len(spells),.5);short=[i for i,(a,b,_) in enumerate(spells) if b-a<=125]
                checks={key:after[key]>=before[key]-1e-12 for key in ('spell_f1_mean','macro_f1','accuracy')}
                row=dict(seed=seed,protocol=protocol,head=head,mode=mode,
                    spellF1=after['spell_f1_mean'],macroF1=after['macro_f1'],accuracy=after['accuracy'],
                    baseline_spellF1=before['spell_f1_mean'],spellF1_delta=after['spell_f1_mean']-before['spell_f1_mean'],
                    short_matches=int(flags[short].sum()),changed_frames=int((prediction!=reference).sum()),
                    spurious_iou50=after['spell_f1'][.5]['n_spurious'],checks=checks)
                scores.append(row);print('RESULT',json.dumps(row),flush=True)
                write(output/'metrics.json',dict(before=before,after=after,summary=row))
        pd.DataFrame([{k:v for k,v in r.items() if k!='checks'} for r in scores]).to_csv(ROUND/f'seed{seed}_scores.csv',index=False)
    if seed==42:
        eligible=[]
        for head,mode in methods:
            relevant=[r for r in scores if r['head']==head and r['mode']==mode]
            main=next(r for r in relevant if r['protocol']=='buffered_minutes')
            if (all(all(r['checks'].values()) for r in relevant) and main['spellF1_delta']>1e-12
                    and main['short_matches']>3):
                eligible.append(dict(head=head,mode=mode,complete=main['short_matches']==6,
                    minimum_gain=min(r['spellF1_delta'] for r in relevant),main_gain=main['spellF1_delta']))
        eligible.sort(key=lambda r:(r['complete'],r['minimum_gain'],r['main_gain']),reverse=True)
        result=dict(eligible=eligible)
        if eligible:result.update(eligible[0])
        write(ROUND/'screen_selection.json',result)
        print('ELIGIBLE',json.dumps(result),flush=True)
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--seed',type=int,choices=(42,43,44))
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.seed:run(args.seed)
    else:parser.error('Choose --prepare or --seed')
