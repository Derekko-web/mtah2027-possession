"""Independent boundary reconstruction and official score/full-file checks."""
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
from possession_boundary_precision import VARIANTS as UNGUARDED
from probability_guarded_boundaries import VARIANTS as GUARDED
from reproduce_boundary_precision import BASE,PROTOCOLS,evaluate,overlap_summary
from research_guarded_boundaries import ROUND,verify


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def intervals(values,left,right):
    result=[];cursor=left
    while cursor<right:
        end=cursor+1
        while end<right and values[end]==values[cursor]:end+=1
        result.append((cursor,end,str(values[cursor])));cursor=end
    return result


def preserve(reference,candidate,blocks):
    output=candidate.copy()
    for left,right in blocks:
        a,b=intervals(reference,left,right),intervals(output,left,right)
        if [s[2] for s in a]!=[s[2] for s in b] or any(y-x<min(v-u,8) for (x,y,_),(u,v,_) in zip(b,a)):
            output[left:right]=reference[left:right]
    return output


def independent_refine(reference,fast,owner,raw,bc,old_mapped,blocks,recoveries,leading,trailing,approach,guarded=False):
    reference=np.asarray(reference);envelope=reference.copy();onset=reference.copy()
    distance=bc[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    tactical=raw[['h__distance_1','a__distance_1']].to_numpy()
    observed=(bc.bc_matched.to_numpy()==1)&(bc.bc_same_ball_seen.to_numpy()==1)&np.isfinite(distance).all(axis=1)
    for left,right in blocks:
        spans=intervals(reference,left,right)
        if leading is not None:
            for i,(start,end,team) in enumerate(spans):
                if i==0 or i==len(spans)-1 or team not in ('H','A') or not 8<=end-start<=250:continue
                opponent='H' if team=='A' else 'A'
                if spans[i-1][2]!=opponent or spans[i+1][2]!=opponent or not observed[start:end].all():continue
                k=int(team=='A');rows=distance[start:end]
                contacts=np.flatnonzero((rows[:,k]<2)&(rows[:,1-k]-rows[:,k]>.5))
                if len(contacts)<3:continue
                lead=min(25,max(0,int(contacts[0])-leading))
                tail=min(25,max(0,end-start-1-int(contacts[-1])-trailing))
                if end-start-lead-tail<8:continue
                envelope[start:start+lead]=opponent
                if tail:envelope[end-tail:end]=opponent
        if approach:
            for event in recoveries:
                start,end,team=event['start'],event['end_exclusive'],event['team']
                if not left<start<end<right:continue
                opponent='H' if team=='A' else 'A';k=int(team=='A')
                if not np.all(reference[start:end]==team) or reference[start-1]!=opponent or reference[end]!=opponent:continue
                previous=start
                while previous>left and reference[previous-1]==opponent:previous-=1
                low=max(previous+8,start-approach)
                if low>=start or old_mapped[low:end].any() or not observed[low:end].all():continue
                d,b=tactical[low:start],distance[low:start]
                evidence=(d[:,k]<3)&(d[:,1-k]-d[:,k]>.5)&(b[:,k]<3)&(b[:,1-k]-b[:,k]>.5)&(fast[low:start,k]>.5)
                found=[i for i in range(len(evidence)-2) if evidence[i:i+3].all()]
                if found:onset[low+found[0]:start]=team
    if guarded:
        envelope=preserve(reference,envelope,blocks);result=reference.copy()
        for left,right in blocks:
            old,new=intervals(reference,left,right),intervals(envelope,left,right)
            for i in range(len(old)-1):
                a,b=old[i][1],new[i][1]
                if a==b:continue
                x,y=min(a,b),max(a,b);team=old[i][2] if b>a else old[i+1][2]
                if np.mean(owner[x:y,int(team=='A')])>.5:result[x:y]=team
        onset=preserve(reference,onset,blocks)
    else:result=envelope
    changed=onset!=reference;result[changed]=onset[changed]
    return preserve(reference,result,blocks)


def audit():
    verify();public=np.arange(11185,18685)
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    old=np.load(ROOT/'data/control_broadcast_v1/inputs/strict_mapping.npz')['frames'][public]>=0
    truth_full=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    scorer_inputs=ROUND/'official_inputs';scorer_inputs.mkdir(exist_ok=False)
    records=[];folds=[];details=[]
    roots=[('unguarded',ROOT/'data/control_boundary_precision_v1/candidates'),('guarded',ROUND/'candidates')]
    for family,root in roots:
        for path in sorted(root.glob('*/seed*/*/oof_predictions.csv')):
            variant,seed_name,protocol=path.relative_to(root).parts[:3];seed=int(seed_name[4:])
            stage='screen' if seed==42 else f'confirmation_seed{seed}'
            table=pd.read_csv(path);baseline=pd.read_csv(BASE/seed_name/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(table.frame,public);np.testing.assert_array_equal(table.ground_truth,truth_full[public])
            fast=np.load(ROOT/f'data/control_boundary_v1/baseline_confirmation/{protocol}/fast_seed{seed}_probabilities.npy')
            owner=np.load(ROOT/f'data/control_tabpfn_fixed_confirmation_v2/{stage}/{protocol}/owner_probability.npy')
            events=json.loads((BASE/seed_name/protocol/'events.json').read_text())
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            config=UNGUARDED[variant] if family=='unguarded' else dict(leading=13,trailing=5,**GUARDED[variant])
            pred=independent_refine(baseline.prediction.to_numpy(),fast,owner,raw,bc,old,blocks,events,**config,guarded=family=='guarded')
            np.testing.assert_array_equal(pred,table.prediction)
            metric=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(pred),bootstrap_n=0)
            name=f'{family}_{variant}_{seed_name}_{protocol}'
            full=np.full(len(truth_full),'X',dtype='<U1');full[public]=pred
            pd.DataFrame(dict(frame=np.arange(len(full)),label=full)).to_csv(scorer_inputs/(name+'.csv'),index=False)
            records.append(dict(name=name,source=str(path.relative_to(ROOT)),metrics=metric))
            if family=='guarded' and variant=='guarded_release_approach25':
                for state in ('D','X'):np.testing.assert_array_equal(pred==state,baseline.prediction==state)
                for block,(a,b) in enumerate(blocks,1):
                    gt=pm.label_codes(table.ground_truth.iloc[a:b])
                    before=pm.evaluate(gt,pm.label_codes(baseline.prediction.iloc[a:b]),bootstrap_n=0)
                    after=pm.evaluate(gt,pm.label_codes(pred[a:b]),bootstrap_n=0)
                    if after['spell_f1_mean']<before['spell_f1_mean']-1e-12 or after['macro_f1']<before['macro_f1']-1e-12:raise ValueError('Clipped-fold regression')
                    folds.append(dict(seed=seed,protocol=protocol,block=block,before_spellF1=before['spell_f1_mean'],spellF1=after['spell_f1_mean'],before_macroF1=before['macro_f1'],macroF1=after['macro_f1']))
                a,b=overlap_summary(table.ground_truth,baseline.prediction),overlap_summary(table.ground_truth,pred)
                for i,(start,end,label) in enumerate(a['gt']):
                    prior=max(((v,j) for v,idx,j in a['overlap'] if idx==i),default=(0,None))
                    after=max(((v,j) for v,idx,j in b['overlap'] if idx==i),default=(0,None))
                    p0=a['predicted'][prior[1]] if prior[1] is not None else (start,end,label)
                    p1=b['predicted'][after[1]] if after[1] is not None else (start,end,label)
                    details.append(dict(seed=seed,protocol=protocol,start=int(public[start]),end_exclusive=int(public[end-1])+1,
                        label=('H','A','D','X')[label],frames=end-start,before_iou=prior[0],after_iou=after[0],
                        before_start_error=p0[0]-start,before_end_error=p0[1]-end,after_start_error=p1[0]-start,after_end_error=p1[1]-end))
    assert len(records)==33 and len(folds)==39
    scored=subprocess.run([sys.executable,str(ROOT/'evaluation/run_evaluation.py'),'--results',str(scorer_inputs),
        '--out',str(ROUND/'official_scoring'),'--bootstrap','0','--no-detail'],cwd=ROOT,capture_output=True,text=True)
    (ROUND/'official_scoring.log').write_text(scored.stdout+scored.stderr)
    if scored.returncode:raise ValueError('Official scorer failed')
    for row in records:
        report=json.loads((ROUND/'official_scoring/reports'/(row['name']+'.json')).read_text())
        for key in ('spell_f1_mean','macro_f1','accuracy'):
            if abs(report[key]-row['metrics'][key])>1e-12:raise ValueError('Official score mismatch')
    qualification=evaluate(ROUND/'candidates/guarded_release_approach25');assert qualification['passed']
    pd.DataFrame(folds).to_csv(ROUND/'fold_scores.csv',index=False);pd.DataFrame(details).to_csv(ROUND/'boundary_diagnostics.csv',index=False)
    first,second=ROUND/'full_export1',ROUND/'full_export2'
    for name in ('candidate.csv','boundary_events.json','manifest.json'):
        if (first/name).read_bytes()!=(second/name).read_bytes():raise ValueError('Full exports differ: '+name)
    parent=pd.read_csv(ROOT/'data/candidate_tabpfn_new_camera_v1/candidate.csv');candidate=pd.read_csv(first/'candidate.csv')
    full_raw=pd.read_pickle(ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl')
    full_bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/full_inputs/broadcast_extension.pkl')
    full_fast=np.load(ROOT/'data/candidate_tabpfn_short_control_v1/fast_probabilities.npy',allow_pickle=False)
    full_owner=np.load(ROOT/'data/candidate_tabpfn_short_control_v1/owner_probability.npy',allow_pickle=False)
    full_old=np.load(ROOT/'data/control_broadcast_v1/full_inputs/mapping.npz')['frames']>=0
    full_events=json.loads((ROOT/'data/candidate_tabpfn_new_camera_v1/recovery_events.json').read_text())
    from games import GAME,half_video_blocks
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    expected=independent_refine(parent.label.to_numpy(),full_fast,full_owner,full_raw,full_bc,full_old,blocks,full_events,13,5,25,True)
    np.testing.assert_array_equal(expected,candidate.label)
    for state in ('D','X'):np.testing.assert_array_equal(candidate.label==state,parent.label==state)
    for index,folder in enumerate((first,second),1):
        result=subprocess.run([sys.executable,str(ROOT/'evaluation/validate_submission.py'),str(folder/'candidate.csv')],capture_output=True,text=True)
        (ROUND/f'full_validator{index}.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('Official full validator failed')
    verify()
    write(ROUND/'final_audit.json',dict(passed=True,independent_OOF_replays=33,official_scores=33,all_nine_scores_and_boundary_overlap_nonregressing=True,
        every_previous_match_retained_at_all_five_IoUs=True,clipped_fold_comparisons=39,clipped_fold_regressions=0,
        two_full_exports_byte_exact=True,independent_full_reconstruction_exact=True,official_full_validator_passed_twice=True,
        full_D_X_and_topology_exact=True,full_changed_frames=int(np.sum(parent.label!=candidate.label)),
        full_csv_sha256=digest(first/'candidate.csv'),new_model_fits=0,new_TabPFN_queries=0,private_full_match_accuracy_unknown=True,
        source_sha256={p:digest(ROOT/p) for p in ('experiments/audit_guarded_boundaries.py','experiments/build_guarded_boundary_candidate.py')},
        qualification_result_sha256=digest(ROUND/'qualification_result.json')))
    print((ROUND/'final_audit.json').read_text(),flush=True)


if __name__=='__main__':audit()
