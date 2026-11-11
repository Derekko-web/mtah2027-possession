"""Independent source-aware arrivals, official scores and full-label replay."""
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
from source_aware_arrivals import VARIANTS as INITIAL
from leading_gap_arrivals import VARIANTS as FILTERED
from reproduce_home_boundary_precision import BASE,assess
from reproduce_boundary_precision import PROTOCOLS,overlap_summary
from research_leading_gap_boundaries import ROUND,verify


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def segments(values,left,right):
    cursor=left;out=[]
    while cursor<right:
        end=cursor+1
        while end<right and values[end]==values[cursor]:end+=1
        out.append((cursor,end,str(values[cursor])));cursor=end
    return out


def independent_refine(reference,owner,raw,bc,blocks,camera,radius,leading_guard):
    ref=np.asarray(reference);out=ref.copy()
    d=raw[['h__distance_1','a__distance_1']].to_numpy();b=bc[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    seen=(bc.bc_matched.to_numpy()==1)&(bc.bc_same_ball_seen.to_numpy()==1)&np.isfinite(b).all(axis=1)
    tc=(d<2)&(d[:,::-1]-d>.5);cc=(b<2)&(b[:,::-1]-b>.5)&seen[:,None]
    for left,right in blocks:
        runs=segments(ref,left,right)
        for i,(start,end,label) in enumerate(runs):
            if i==0 or i==len(runs)-1 or label not in ('H','A') or not 8<=end-start<=250:continue
            opponent='H' if label=='A' else 'A'
            if runs[i-1][2]!=opponent:continue
            k=int(label=='A');move=0;complete=seen[start:end].all()
            if complete and camera:
                own_votes=int(cc[start:end,k].sum());other_votes=int(cc[start:end,1-k].sum())
                own_tactical=int(tc[start:end,k].sum());other_tactical=int(tc[start:end,1-k].sum())
                if own_votes>=3 and other_tactical>=3 and own_votes>other_votes and other_tactical>own_tactical:
                    first=int(np.where(cc[start:end,k])[0][0]);move=max(0,min(25,end-start-8,first-5))
            elif not complete and radius:
                capacity=min(radius,end-start-8)
                if capacity<=0:continue
                q=owner[start:start+capacity]
                relative=np.log(np.maximum(q[:,k],1e-12)/np.maximum(q[:,1-k],1e-12))
                costs=np.add.accumulate(relative);best=0.;choice=0
                for index,cost in enumerate(costs,1):
                    if cost<best:best=float(cost);choice=index
                if choice and best<-1e-12 and owner[start:start+choice,1-k].mean()>.5:
                    allow=True
                    if leading_guard:
                        visible=np.where(seen[start:end])[0]
                        gap=int(visible[0]) if len(visible) else end-start
                        allow=bool(0<gap<=radius and cc[start:end,k].sum()>=3)
                    if allow:move=choice
            if move:out[start:start+move]=opponent
        actual=segments(out,left,right)
        if [s[2] for s in actual]!=[s[2] for s in runs]:raise ValueError('Independent topology violation')
    return out


def audit():
    verify();public=np.arange(11185,18685)
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    truth_full=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    scorer=ROUND/'official_inputs';scorer.mkdir(exist_ok=False);records=[];fold_rows=[];diagnostics=[]
    roots=[('initial',ROOT/'data/control_home_boundaries_v1/candidates'),('leading',ROUND/'candidates')]
    for family,root in roots:
        variants=INITIAL if family=='initial' else FILTERED
        for path in sorted(root.glob('*/seed*/*/oof_predictions.csv')):
            variant,seed_name,protocol=path.relative_to(root).parts[:3];seed=int(seed_name[4:])
            table=pd.read_csv(path);baseline=pd.read_csv(BASE/seed_name/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(table.frame,public);np.testing.assert_array_equal(table.ground_truth,truth_full[public])
            stage='screen' if seed==42 else f'confirmation_seed{seed}'
            owner=np.load(ROOT/f'data/control_tabpfn_fixed_confirmation_v2/{stage}/{protocol}/owner_probability.npy')
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            params=variants[variant]
            pred=independent_refine(baseline.prediction.to_numpy(),owner,raw,bc,blocks,params['camera'],params['posterior_radius'],family=='leading')
            np.testing.assert_array_equal(pred,table.prediction)
            metric=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(pred),bootstrap_n=0)
            name=f'{family}_{variant}_{seed_name}_{protocol}'
            full=np.full(len(truth_full),'X',dtype='<U1');full[public]=pred
            pd.DataFrame(dict(frame=np.arange(len(full)),label=full)).to_csv(scorer/(name+'.csv'),index=False)
            records.append(dict(name=name,source=str(path.relative_to(ROOT)),metrics=metric))
            if family=='leading' and variant=='leading_camera13':
                for state in ('D','X'):np.testing.assert_array_equal(pred==state,baseline.prediction==state)
                for block,(a,b) in enumerate(blocks,1):
                    g=pm.label_codes(table.ground_truth.iloc[a:b]);before=pm.evaluate(g,pm.label_codes(baseline.prediction.iloc[a:b]),bootstrap_n=0)
                    after=pm.evaluate(g,pm.label_codes(pred[a:b]),bootstrap_n=0)
                    if after['spell_f1_mean']<before['spell_f1_mean']-1e-12 or after['macro_f1']<before['macro_f1']-1e-12:raise ValueError('Clipped-fold regression')
                    fold_rows.append(dict(seed=seed,protocol=protocol,block=block,before_spellF1=before['spell_f1_mean'],spellF1=after['spell_f1_mean'],before_macroF1=before['macro_f1'],macroF1=after['macro_f1']))
                old,new=overlap_summary(table.ground_truth,baseline.prediction),overlap_summary(table.ground_truth,pred)
                for index,(start,end,label) in enumerate(old['gt']):
                    prior=max(((v,j) for v,i,j in old['overlap'] if i==index),default=(0,None))
                    current=max(((v,j) for v,i,j in new['overlap'] if i==index),default=(0,None))
                    p0=old['predicted'][prior[1]] if prior[1] is not None else (start,end,label)
                    p1=new['predicted'][current[1]] if current[1] is not None else (start,end,label)
                    diagnostics.append(dict(seed=seed,protocol=protocol,start=int(public[start]),end_exclusive=int(public[end-1])+1,
                        label=('H','A','D','X')[label],frames=end-start,before_iou=prior[0],iou=current[0],
                        before_start_error=p0[0]-start,start_error=p1[0]-start,before_end_error=p0[1]-end,end_error=p1[1]-end))
    assert len(records)==30 and len(fold_rows)==39
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/run_evaluation.py'),'--results',str(scorer),'--out',str(ROUND/'official_scoring'),
        '--bootstrap','0','--no-detail'],cwd=ROOT,capture_output=True,text=True)
    (ROUND/'official_scoring.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise ValueError('Official scoring failed')
    for row in records:
        m=json.loads((ROUND/'official_scoring/reports'/(row['name']+'.json')).read_text())
        for key in ('spell_f1_mean','macro_f1','accuracy'):
            if abs(m[key]-row['metrics'][key])>1e-12:raise ValueError('Official score mismatch')
    qualified=assess(ROUND/'candidates/leading_camera13');assert qualified['passed']
    pd.DataFrame(fold_rows).to_csv(ROUND/'fold_scores.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(ROUND/'boundary_diagnostics.csv',index=False)
    first,second=ROUND/'full_export1',ROUND/'full_export2'
    for name in ('candidate.csv','arrival_events.json','manifest.json'):
        if (first/name).read_bytes()!=(second/name).read_bytes():raise ValueError('Full exports differ: '+name)
    parent=pd.read_csv(ROOT/'data/candidate_tabpfn_boundary_precision_v1/candidate.csv');candidate=pd.read_csv(first/'candidate.csv')
    full_raw=pd.read_pickle(ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl')
    full_bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/full_inputs/broadcast_extension.pkl')
    full_owner=np.load(ROOT/'data/candidate_tabpfn_short_control_v1/owner_probability.npy',allow_pickle=False)
    from games import GAME,half_video_blocks
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    expected=independent_refine(parent.label.to_numpy(),full_owner,full_raw,full_bc,blocks,True,13,True)
    np.testing.assert_array_equal(expected,candidate.label)
    for state in ('D','X'):np.testing.assert_array_equal(candidate.label==state,parent.label==state)
    for index,folder in enumerate((first,second),1):
        checked=subprocess.run([sys.executable,str(ROOT/'evaluation/validate_submission.py'),str(folder/'candidate.csv')],capture_output=True,text=True)
        (ROUND/f'full_validator{index}.log').write_text(checked.stdout+checked.stderr)
        if checked.returncode:raise ValueError('Official full validator failed')
    verify()
    write(ROUND/'final_audit.json',dict(passed=True,independent_OOF_replays=30,official_scores=30,
        all_nine_score_and_target_checks_pass=True,all_previous_matches_retained_at_all_five_IoUs=True,
        clipped_fold_comparisons=39,clipped_fold_regressions=0,two_full_exports_byte_exact=True,
        independent_full_reconstruction_exact=True,official_full_validator_passed_twice=True,full_D_X_and_topology_exact=True,
        full_changed_frames=int(np.sum(parent.label!=candidate.label)),full_csv_sha256=digest(first/'candidate.csv'),
        new_model_fits=0,new_TabPFN_queries=0,private_full_match_accuracy_unknown=True,
        source_sha256={p:digest(ROOT/p) for p in ('experiments/audit_home_boundaries.py','experiments/build_home_boundary_candidate.py')},
        qualification_sha256=digest(ROUND/'qualification_result.json')))
    print((ROUND/'final_audit.json').read_text(),flush=True)


if __name__=='__main__':audit()
