"""Separate replay, official scoring and full-export audit for short repairs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from games import GAME,half_video_blocks
from reproduce_short_possessions import BASE,PROTOCOLS
from research_control_continuation_repair import ROUND,verify


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_envelope(labels,bc,blocks):
    output=np.array(labels,copy=True)
    for left,right in blocks:
        position=left
        while position<right:
            end=position+1
            while end<right and labels[end]==labels[position]:end+=1
            team=labels[position];opponent='A' if team=='H' else 'H'
            if (team in ('H','A') and 8<=end-position<=250 and position>left and end<right
                    and labels[position-1]==opponent and labels[end]==opponent):
                view=bc.iloc[position:end]
                distances=view[[f'{team.lower()}__bc_same_distance_1',f'{opponent.lower()}__bc_same_distance_1']].to_numpy()
                valid=(view.bc_matched==1).all() and (view.bc_same_ball_seen==1).all() and np.isfinite(distances).all()
                if valid:
                    contacts=np.where((distances[:,0]<2)&(distances[:,1]>distances[:,0]+.5))[0]
                    if len(contacts)>=3:
                        clip_left=max(0,min(int(contacts[0])-13,25))
                        clip_right=max(0,min(end-position-int(contacts[-1])-1-13,25))
                        if end-position-clip_left-clip_right>=8:
                            output[position:position+clip_left]=opponent
                            if clip_right:output[end-clip_right:end]=opponent
            position=end
    return output


def audit():
    verify()
    from research_short_possession_repair import verify as verify_research,inputs
    from short_possession_repair import predict,repair as learned_repair
    from control_continuation_repair import repair as physical_repair
    verify_research()
    public=np.arange(11185,18685)
    raw,x=inputs();n=len(raw)
    bc_fullprefix=pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl')
    bc=bc_fullprefix.iloc[public].reset_index(drop=True)
    motion=pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[public].reset_index(drop=True)
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    model_reloads=0
    for head in ('sqrt_spell','equal_spell'):
        for protocol in PROTOCOLS:
            folder=ROOT/'data/control_short_repair_v1/seed42'/head/protocol
            expected=np.load(folder/'probability.npy');actual=np.full_like(expected,np.nan)
            for split in (s for s in splits if s['protocol']==protocol):
                validation=np.asarray(split['validation']);model=CatBoostClassifier()
                model.load_model(folder/f'block{split["block"]}.cbm')
                actual[validation-11185]=predict(model,x.iloc[validation]);model_reloads+=1
            np.testing.assert_array_equal(actual,expected)
    official_inputs=ROUND/'official_inputs';official_inputs.mkdir(exist_ok=False)
    scoring_records=[];replayed=0;candidate_labels={};fold_rows=[]
    roots=[('duration',ROOT/'data/control_short_repair_v1/candidates'),
           ('envelope',ROOT/'data/control_broadcast_short_envelope_v1/candidates'),
           ('continuation',ROUND/'candidates')]
    for family,root in roots:
        for path in sorted(root.glob('*/seed*/*/oof_predictions.csv')):
            variant,seed_name,protocol=path.relative_to(root).parts[:3];seed=int(seed_name[4:])
            stage='screen' if seed==42 else f'confirmation_seed{seed}'
            original=pd.read_csv(BASE/stage/protocol/'oof_predictions.csv');table=pd.read_csv(path)
            np.testing.assert_array_equal(table.frame,public);np.testing.assert_array_equal(table.ground_truth,original.ground_truth)
            reference=original[f'tabpfn_short_control_seed{seed}'].to_numpy()
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            if family=='duration':
                head,mode=next((h,variant[len(h)+1:]) for h in ('sqrt_spell','equal_spell') if variant.startswith(h+'_'))
                p=np.full((n,2),np.nan);p[public]=np.load(ROOT/'data/control_short_repair_v1/seed42'/head/protocol/'probability.npy')
                owner=np.full((n,2),np.nan);owner[public]=np.load(BASE/stage/protocol/'owner_probability.npy')
                ref=np.full(n,'X',dtype='<U1');ref[public]=reference
                fullblocks=[(a+11185,b+11185) for a,b in blocks]
                actual,_=learned_repair(ref,p,owner,raw,fullblocks,mode);actual=actual[public]
            elif family=='envelope':
                from broadcast_short_envelope import refine
                actual,_=refine(reference,bc,blocks,int(variant[3:]))
            else:
                actual=independent_envelope(reference,bc,blocks)
                if variant!='broadcast13':
                    fast=np.load(ROOT/'data/control_boundary_v1/baseline_confirmation'/protocol/f'fast_seed{seed}_probabilities.npy')
                    actual,_=physical_repair(actual,fast,raw.iloc[public].reset_index(drop=True),motion,blocks,int(variant.removeprefix('continuation')))
            np.testing.assert_array_equal(actual,table.prediction.to_numpy());replayed+=1
            metrics=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(actual),bootstrap_n=0)
            name=f'{family}_{variant}_{seed_name}_{protocol}'
            full=np.full(199319,'X',dtype='<U1');full[public]=actual
            pd.DataFrame(dict(frame=np.arange(199319),label=full)).to_csv(official_inputs/(name+'.csv'),index=False)
            scoring_records.append(dict(name=name,source=str(path.relative_to(ROOT)),metrics=metrics,family=family,variant=variant,seed=seed,protocol=protocol))
            if family=='continuation' and variant=='broadcast13':
                candidate_labels[(seed,protocol)]=actual
                baseline=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(reference),bootstrap_n=0)
                for metric in ('spell_f1_mean','macro_f1','accuracy'):
                    if metrics[metric]<baseline[metric]-1e-12:raise ValueError('Selected aggregate regression')
                for state in ('D','X'):np.testing.assert_array_equal(actual==state,reference==state)
                for i,(a,b) in enumerate(blocks,1):
                    truth=pm.label_codes(table.ground_truth.iloc[a:b]);before=pm.evaluate(truth,pm.label_codes(reference[a:b]),bootstrap_n=0)
                    after=pm.evaluate(truth,pm.label_codes(actual[a:b]),bootstrap_n=0)
                    fold_rows.append(dict(seed=seed,protocol=protocol,block=i,before_spellF1=before['spell_f1_mean'],
                                          after_spellF1=after['spell_f1_mean'],before_macroF1=before['macro_f1'],after_macroF1=after['macro_f1']))
    assert replayed==39 and len(candidate_labels)==9 and model_reloads==26
    official=ROUND/'official_scoring'
    completed=subprocess.run([sys.executable,str(ROOT/'evaluation/run_evaluation.py'),'--results',str(official_inputs),
                              '--out',str(official),'--bootstrap','0','--no-detail'],capture_output=True,text=True,cwd=ROOT)
    (ROUND/'official_scoring.log').write_text(completed.stdout+completed.stderr)
    if completed.returncode:raise ValueError('Official scorer failed')
    for record in scoring_records:
        scored=json.loads((official/'reports'/(record['name']+'.json')).read_text())
        for metric in ('spell_f1_mean','macro_f1','accuracy'):
            assert abs(scored[metric]-record['metrics'][metric])<1e-12
    pd.DataFrame(fold_rows).to_csv(ROUND/'selected_fold_scores.csv',index=False)
    first,second=ROUND/'full_export1',ROUND/'full_export2'
    for name in ('candidate.csv','manifest.json','repair_events.json'):
        assert (first/name).read_bytes()==(second/name).read_bytes()
    baseline=pd.read_csv(ROOT/'data/candidate_tabpfn_short_control_v1/candidate.csv')
    proposed=pd.read_csv(first/'candidate.csv')
    full_bc=pd.read_pickle(ROOT/'data/control_broadcast_v1/full_inputs/combined_inputs.pkl')
    expected=independent_envelope(baseline.label.to_numpy(),full_bc,[(a,b+1) for _,a,b in half_video_blocks(GAME)])
    np.testing.assert_array_equal(expected,proposed.label.to_numpy())
    assert not pm.validate_submission(proposed,199319)
    for state in ('D','X'):np.testing.assert_array_equal(proposed.label==state,baseline.label==state)
    for index,folder in enumerate((first,second),1):
        result=subprocess.run([sys.executable,str(ROOT/'evaluation/validate_submission.py'),str(folder/'candidate.csv')],capture_output=True,text=True)
        (ROUND/f'full_validator{index}.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0
    verify()
    info=dict(passed=True,official_scores_recomputed=39,candidate_replays_exact=39,rejected_head_reloads_exact=26,
        all_nine_primary_macro_accuracy_nonregressing=True,selected_per_fold_checks=len(fold_rows),
        selected_fold_spell_regressions=sum(r['after_spellF1']<r['before_spellF1']-1e-12 for r in fold_rows),
        selected_fold_macro_regressions=sum(r['after_macroF1']<r['before_macroF1']-1e-12 for r in fold_rows),
        independent_envelope_implementation_exact_all_nine_and_full=True,two_full_exports_byte_exact=True,
        official_full_validator_passed_twice=True,full_D_X_exact=True,full_changed_frames=int((proposed.label!=baseline.label).sum()),
        full_csv_sha256=digest(first/'candidate.csv'),complete_short_error_resolution=False,remaining_main_short_matches='5/6',
        model_fits=26,additional_TabPFN_fits=0,private_full_match_accuracy_unknown=True,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'experiments/build_broadcast_short_candidate.py']})
    (ROUND/'final_audit.json').write_text(json.dumps(info,indent=2,sort_keys=True)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':audit()
