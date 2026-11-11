"""Independent interval reconstruction, official scores and full export audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from reproduce_remaining_away import BASE,PROTOCOLS,evaluate
from research_new_camera_control import ROUND,verify


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def independently_recover(reference,fast,raw,motion,bc,old_mapped,blocks,confirmation):
    result=np.array(reference,copy=True)
    d=raw[['h__distance_1','a__distance_1']].to_numpy()
    bd=bc[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    kernels=motion[['h__tc_kernel','a__tc_kernel']].to_numpy()
    seen=(bc.bc_matched.to_numpy()==1)&(bc.bc_same_ball_seen.to_numpy()==1)&np.isfinite(bd).all(axis=1)
    contacts=(d<2)&(d[:,::-1]-d>.5)
    bilateral=(bd<2)&(bd[:,::-1]-bd>.5)&seen[:,None]&~old_mapped[:,None]
    for left,right in blocks:
        cursor=left
        while cursor<right:
            stop=cursor+1
            while stop<right and reference[stop]==reference[cursor]:stop+=1
            if reference[cursor] not in ('H','A') or stop-cursor<=250:cursor=stop;continue
            own=int(reference[cursor]=='A');opponent=1-own
            trigger=contacts[cursor:stop,opponent]&(fast[cursor:stop,opponent]>.5)&(kernels[cursor:stop,own]<.5)&~old_mapped[cursor:stop]&seen[cursor:stop]
            hits=np.where(trigger)[0];groups=np.split(hits,np.where(np.diff(hits)>13)[0]+1)
            for group in groups:
                candidates=[int(group[i]) for i in range(max(0,len(group)-2)) if group[i+2]-group[i]<13]
                if not candidates:continue
                begin=cursor+candidates[0]
                if begin-cursor<13:continue
                if bilateral[max(cursor,begin-13):min(stop,begin+13),opponent].sum()<3:continue
                limit=min(stop-13,begin+125)
                stable=(kernels[begin:limit,own]>=.5)&(fast[begin:limit,own]>=.65)
                if len(stable)<confirmation:continue
                counts=np.convolve(stable.astype(int),np.ones(confirmation,dtype=int),mode='valid')
                positions=np.where(counts==confirmation)[0]
                if not len(positions):continue
                end=begin+int(positions[0])+confirmation
                if not 8<=end-begin<=125 or old_mapped[begin:end].any() or not seen[begin:end].all():continue
                if bilateral[max(begin,end-confirmation):end,own].sum()<3:continue
                result[begin:end]='H' if opponent==0 else 'A'
            cursor=stop
    return result


def audit_oof():
    verify();public=np.arange(11185,18685)
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    motion=pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROUND/'broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    old=np.load(ROOT/'data/control_broadcast_v1/inputs/strict_mapping.npz')['frames'][public]>=0
    truth_full=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    output=ROUND/'official_inputs';output.mkdir(exist_ok=False);records=[];folds=[];diagnostics=[]
    for path in sorted((ROUND/'candidates').glob('*/seed*/*/oof_predictions.csv')):
        variant,seed_name,protocol=path.relative_to(ROUND/'candidates').parts[:3]
        seed=int(seed_name[4:]);stable=int(variant[6:]);table=pd.read_csv(path)
        base=pd.read_csv(BASE/seed_name/protocol/'oof_predictions.csv')
        np.testing.assert_array_equal(table.frame,public);np.testing.assert_array_equal(table.ground_truth,truth_full[public])
        fast=np.load(ROOT/f'data/control_boundary_v1/baseline_confirmation/{protocol}/fast_seed{seed}_probabilities.npy')
        blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
        expected=independently_recover(base.prediction.to_numpy(),fast,raw,motion,bc,old,blocks,stable)
        np.testing.assert_array_equal(expected,table.prediction)
        for state in ('D','X'):np.testing.assert_array_equal(table.prediction==state,base.prediction==state)
        for title,prediction in ((f'{variant}_{seed_name}_{protocol}',expected),):
            full=np.full(len(truth_full),'X',dtype='<U1');full[public]=prediction
            pd.DataFrame(dict(frame=np.arange(len(full)),label=full)).to_csv(output/(title+'.csv'),index=False)
            metric=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(prediction),bootstrap_n=0)
            records.append(dict(name=title,path=str(path.relative_to(ROOT)),metrics=metric,seed=seed,stable_frames=stable,protocol=protocol))
        baseline_title=f'baseline_{seed_name}_{protocol}'
        if not (output/(baseline_title+'.csv')).exists():
            full=np.full(len(truth_full),'X',dtype='<U1');full[public]=base.prediction
            pd.DataFrame(dict(frame=np.arange(len(full)),label=full)).to_csv(output/(baseline_title+'.csv'),index=False)
        if stable==13:
            for block,(a,b) in enumerate(blocks,1):
                gt=pm.label_codes(table.ground_truth.iloc[a:b])
                before=pm.evaluate(gt,pm.label_codes(base.prediction.iloc[a:b]),bootstrap_n=0)
                after=pm.evaluate(gt,pm.label_codes(expected[a:b]),bootstrap_n=0)
                if after['spell_f1_mean']<before['spell_f1_mean']-1e-12 or after['macro_f1']<before['macro_f1']-1e-12:raise ValueError('Fold regression')
                folds.append(dict(seed=seed,protocol=protocol,block=block,before_spellF1=before['spell_f1_mean'],spellF1=after['spell_f1_mean'],
                                  before_macroF1=before['macro_f1'],macroF1=after['macro_f1']))
            g=pm.label_codes(table.ground_truth);runs=pm.valid_runs(g);sp=pm.possession_spells(g,runs)
            ov=pm.spell_overlaps(sp,pm.possession_spells(pm.label_codes(expected),runs));flags=pm.matched_gt_flags(ov,len(sp),.5)
            short=[i for i,(a,b,k) in enumerate(sp) if b-a<=125]
            changed=base.prediction.to_numpy()!=expected
            diagnostics.append(dict(seed=seed,protocol=protocol,short_matches=int(flags[short].sum()),
                corrected_frames=int(np.sum(changed&(expected==table.ground_truth.to_numpy()))),
                damaged_frames=int(np.sum(changed&(base.prediction.to_numpy()==table.ground_truth.to_numpy())))))
    assert len(records)==12 and len(folds)==39
    scored=subprocess.run([sys.executable,str(ROOT/'evaluation/run_evaluation.py'),'--results',str(output),
        '--out',str(ROUND/'official_scoring'),'--bootstrap','0','--no-detail'],cwd=ROOT,text=True,capture_output=True)
    (ROUND/'official_scoring.log').write_text(scored.stdout+scored.stderr)
    if scored.returncode:raise ValueError('Official score execution failed')
    for row in records:
        official=json.loads((ROUND/'official_scoring/reports'/(row['name']+'.json')).read_text())
        for metric in ('spell_f1_mean','macro_f1','accuracy'):
            if abs(official[metric]-row['metrics'][metric])>1e-12:raise ValueError('Official scoring mismatch')
    proof=evaluate(ROUND/'candidates/stable13')
    assert proof['passed'] and proof['target_recovered_pairs']==9
    pd.DataFrame(folds).to_csv(ROUND/'fold_scores.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(ROUND/'diagnostics.csv',index=False)
    write(ROUND/'oof_audit.json',dict(passed=True,independent_candidate_replays=12,official_candidate_scores=12,
        official_baseline_scores=9,all_nine_current_score_and_target_gates=True,fold_checks=39,fold_regressions=0,
        main_short_matches='6/6',model_fits=0,private_accuracy_unknown=True,diagnostics=diagnostics,
        source_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}))
    verify();print('OOF audit passed:12replays,21official scores,39nonregressing folds,9/9target recoveries.',flush=True)


def audit_full():
    verify()
    from games import GAME,half_video_blocks
    from broadcast_alignment import map_anchors
    from subgrid_alignment import extension,STRICT
    first,second=ROUND/'full_export1',ROUND/'full_export2'
    for name in ('candidate.csv','manifest.json','recovery_events.json'):
        if (first/name).read_bytes()!=(second/name).read_bytes():raise ValueError('Full export mismatch: '+name)
    features=ROUND/'full_inputs';m=json.loads((features/'manifest.json').read_text())
    for name,sha in m['cache_sha256'].items():
        if digest(features/name)!=sha:raise ValueError('Full observation artifact changed')
    load=lambda name:dict(np.load(features/name,allow_pickle=False))
    tactical,broadcast=load('tactical_observations.npz'),load('broadcast_observations.npz')
    anchors=load('anchors.npz');actual=load('extension_mapping.npz')
    parent_mapping=dict(np.load(ROOT/'data/control_broadcast_v1/full_inputs/mapping.npz',allow_pickle=False))
    rebuilt=extension(parent_mapping,map_anchors(anchors,tactical,broadcast,STRICT))
    for field in rebuilt:np.testing.assert_array_equal(rebuilt[field],actual[field])
    parent=pd.read_csv(ROOT/'data/candidate_tabpfn_broadcast_short_v1/candidate.csv')
    candidate=pd.read_csv(first/'candidate.csv')
    raw=pd.read_pickle(ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl')
    motion=pd.read_pickle(ROUND/'full_control_kernel/kernels.pkl')
    bc=pd.read_pickle(features/'broadcast_extension.pkl')
    fast=np.load(ROOT/'data/candidate_tabpfn_short_control_v1/fast_probabilities.npy',allow_pickle=False)
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    expected=independently_recover(parent.label.to_numpy(),fast,raw,motion,bc,parent_mapping['frames']>=0,blocks,13)
    np.testing.assert_array_equal(expected,candidate.label)
    for state in ('D','X'):np.testing.assert_array_equal(candidate.label==state,parent.label==state)
    for index,folder in enumerate((first,second),1):
        result=subprocess.run([sys.executable,str(ROOT/'evaluation/validate_submission.py'),str(folder/'candidate.csv')],capture_output=True,text=True)
        (ROUND/f'full_validator{index}.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('Official full validator failed')
    oof=json.loads((ROUND/'oof_audit.json').read_text());assert oof['passed']
    verify()
    write(ROUND/'final_audit.json',dict(passed=True,full_export_sha256=digest(first/'candidate.csv'),
        two_full_exports_byte_exact=True,full_independent_control_reconstruction_exact=True,
        saved_full_anchor_mapping_reconstruction_exact=True,full_public_observations_exact=True,
        full_D_X_exact=True,official_full_validator_passed_twice=True,all_nine_target_and_score_gates=True,
        previous_IoU50_GT_matches_retained=True,all_five_threshold_F1s_nonregressing=True,fold_regressions=0,
        main_short_matches='6/6',full_changed_frames=int(np.sum(parent.label!=candidate.label)),
        complete_frame_exact_71_recovery=False,private_full_match_accuracy_unknown=True,new_fits=0,
        OOF_audit_sha256=digest(ROUND/'oof_audit.json'),full_feature_manifest_sha256=digest(features/'manifest.json'),
        source_sha256={p:digest(ROOT/p) for p in ('experiments/audit_new_camera_control.py','experiments/build_new_camera_candidate.py')}))
    print((ROUND/'final_audit.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--oof',action='store_true');p.add_argument('--full',action='store_true');a=p.parse_args()
    if a.oof:audit_oof()
    elif a.full:audit_full()
    else:p.error('Choose --oof or --full')
