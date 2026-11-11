"""Independent rule replay,all official scores,and complete full-export audit."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from reproduce_holdout_extras import BASE,assess
from reproduce_boundary_precision import overlap_summary,PROTOCOLS
from research_bounded_holdout_repair import ROUND,verify
from research_holdout_extras import digest,write

FAMILIES=(('initial','control_holdout_extras_v1'),('bounded','control_bounded_holdout_repair_v1'))


def runs(values,left,right):
    cursor=left;result=[]
    while cursor<right:
        end=cursor+1
        while end<right and values[end]==values[cursor]:end+=1
        result.append((cursor,end,str(values[cursor])));cursor=end
    return result


def trim_start(probability,k,start,end):
    count=min(25,end-start-8)
    if count<=0:return start
    local=probability[start:start+count]
    costs=np.add.accumulate(np.log(np.maximum(local[:,k],1e-12)/np.maximum(local[:,1-k],1e-12)))
    best=0.;choice=0
    for j,cost in enumerate(costs,1):
        if cost<best:best=float(cost);choice=j
    return start+choice if choice and best<-1e-12 and local[:choice,1-k].mean()>.5 else start


def independent_refine(reference,heads,owner,fast,raw,bc,blocks,stages,bounded):
    ref=np.asarray(reference);out=ref.copy();proposals=[];unseen_actions=[]
    td=raw[['h__distance_1','a__distance_1']].to_numpy();bd=bc[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    observed=(bc.bc_matched.to_numpy()==1)&(bc.bc_same_ball_seen.to_numpy()==1)&np.isfinite(bd).all(axis=1)
    tc=(td<2)&(td[:,::-1]-td>.5);contacts=(bd<2)&(bd[:,::-1]-bd>.5)&observed[:,None]
    visible=raw.ball_seen.to_numpy()>0;moving=raw.ball_displacement_1s.to_numpy()>4.5
    outside=(raw.ball_goal_line_distance.to_numpy()<0)|(raw.ball_touchline_distance.to_numpy()<0)
    any_visible=visible|observed
    for left,right in blocks:
        spans=runs(ref,left,right)
        for i,(start,end,label) in enumerate(spans):
            before=spans[i-1][2] if i else None;after=spans[i+1][2] if i+1<len(spans) else None
            length=end-start
            if label=='D' and 'bridge' in stages and length<=125 and before==after and before in ('H','A'):
                k=int(before=='A');lo=max(left,start-13);hi=min(right,end+13)
                if (visible[start:end].mean()>=.5 and moving[start:end].mean()>=.5 and fast[start:end,k].mean()>=.65
                    and tc[lo:hi,k].sum()>=3 and contacts[lo:hi,1-k].sum()<3 and not outside[start:end].any()):
                    proposals.append((start,end,before))
            if label not in ('H','A'):continue
            k=int(label=='A');other='H' if label=='A' else 'A'
            if 'pitch' in stages and length<=250 and 'D' in (before,after) and not observed[start:end].any():
                first=next((j for j in range(start,end-2) if outside[j:j+3].all()),None)
                if first is not None:
                    if before==after=='D':
                        begin=first+2 if bounded and first>start else start
                        proposals.append((begin,end,'D'))
                    elif before==other and after=='D':
                        proposals.extend(((start,first+2,other),(first+2,end,'D')))
            if ('edge' in stages and length<=125 and i in (0,len(spans)-1) and not observed[start:end].any()
                and visible[start:end].mean()>=.5 and moving[start:end].mean()<=.5 and heads[start:end].mean(0).max()>.5):
                proposals.append((start,end,'D'))
            if 'late' in stages and 8<=length<=250 and before==after==other and observed[start:end].all():
                points=np.flatnonzero(contacts[start:end,k]);groups=[]
                for point in points:
                    if not groups or point-groups[-1][-1]>5:groups.append([])
                    groups[-1].append(int(point))
                groups=[g for g in groups if len(g)>=3]
                if len(groups)>=2 and groups[-1][0]-groups[-2][-1]>=25:
                    lo=start+groups[-1][0];hi=start+groups[-1][-1]+1;votes=tc[lo:hi].sum(0);begin=max(start,lo-5)
                    if votes[1-k]>=3 and votes[1-k]>votes[k] and end-begin>=8 and owner[start:begin,1-k].mean()>.5:
                        proposals.append((start,begin,other))
            edge_neighbor=spans[1][2] if i==0 and len(spans)>1 else spans[-2][2] if i==len(spans)-1 and len(spans)>1 else None
            cross_context=before==after==other or bounded and i in (0,len(spans)-1) and edge_neighbor==other
            if 'cross' in stages and 8<=length<=250 and cross_context:
                tactical=tc[start:end].sum(0);broadcast=contacts[start:end].sum(0)
                if tactical[k]>=3 and tactical[k]>tactical[1-k] and broadcast[1-k]>=3 and broadcast[1-k]>broadcast[k]:
                    begin=trim_start(owner,k,start,end);finish=end-trim_start(fast[start:end][::-1],k,0,length)
                    if finish-begin>=8:proposals.extend(((start,begin,other),(finish,end,other)))
            if 'release' in stages and 8<=length<=125 and contacts[start:end,k].sum()>=3:
                if after=='D':
                    hi=min(right,end+25,spans[i+1][1]);lo=max(start,end-13)
                    points=np.flatnonzero(contacts[lo:hi,k])
                    if len(points):
                        finish=min(hi,lo+int(points[-1])+1+13)
                        if finish>end:proposals.append((end,finish,label))
                    if length<50 and before==other:
                        lo=max(left,start-13);hi=min(right,start+13)
                        points=np.flatnonzero(tc[lo:hi,k]);recent=np.flatnonzero(tc[lo:start,k])
                        if len(points)>=3 and len(recent) and lo+int(recent[-1])>=start-5:
                            begin=max(left,start-25,lo+int(points[0])-5)
                            if begin<start and np.all(ref[begin:start]==other):proposals.append((begin,start,label))
                if i+3<len(spans):
                    ns,ne,nlabel=spans[i+1];ts,te,tlabel=spans[i+2];ds,de,dlabel=spans[i+3]
                    if (nlabel==other and tlabel==label and dlabel=='D' and 8<=ne-ns<=125 and te-ts<8
                        and contacts[ns:ne,1-k].sum()<3 and owner[ns:ne,1-k].mean()<.65):
                        last=start+int(np.flatnonzero(contacts[start:end,k])[-1])+1
                        finish=min(ns+25,last+30,te)
                        if finish>=ns and te-finish>=8:proposals.extend(((ns,finish,label),(finish,te,'D')))
            if ('unseen' in stages and 0<i<len(spans)-1 and 1<=length<=125 and 'D' in (before,after)
                and other in (before,after) and not observed[start:end].any() and not visible[start:end].any()):
                if before=='D':
                    cursor=end
                    while cursor<min(right,end+25) and not any_visible[cursor]:cursor+=1
                    dead_start=start;dead_end=min(end,max(min(end,start+25),max(start,min(end,cursor-25)))+5)
                else:
                    cursor=start
                    while cursor>max(left,start-25) and not any_visible[cursor-1]:cursor-=1
                    dead_start=max(start,min(max(start,end-25),min(end,max(start,cursor+25)))-5);dead_end=end
                unseen_actions.extend(((start,end,other),(dead_start,dead_end,'D')))
    for start,end,label in proposals+unseen_actions:
        if end>start:out[start:end]=label
    return out


def audit():
    verify();public=np.arange(11185,18685)
    sources=['experiments/audit_holdout_extras.py','experiments/test_holdout_repair_audit.py','experiments/build_holdout_repair_candidate.py',
        'evaluation/run_evaluation.py','evaluation/validate_submission.py']
    for _,folder in FAMILIES:sources+=list(json.loads((ROOT/'data'/folder/'plan.json').read_text())['source_sha256'])
    write(ROUND/'audit_plan.json',dict(source_sha256={p:digest(ROOT/p) for p in sorted(set(sources))},OOF_outputs=30,official_scores=32))
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    truth=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    scorer=ROUND/'official_inputs';scorer.mkdir();records=[];diagnostics=[]
    for family,folder in FAMILIES:
        root=ROOT/'data'/folder;plan=json.loads((root/'plan.json').read_text())
        for path in sorted((root/'candidates').glob('*/seed*/*/oof_predictions.csv')):
            variant,seed_name,protocol=path.relative_to(root/'candidates').parts[:3];seed=int(seed_name[4:]);stage='screen' if seed==42 else f'confirmation_seed{seed}'
            table=pd.read_csv(path);baseline=pd.read_csv(BASE/seed_name/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(table.frame,public);np.testing.assert_array_equal(table.ground_truth,truth[public])
            heads=np.load(ROOT/f'data/control_broadcast_v1/baseline_confirmation/{protocol}/heads_seed{seed}_probabilities.npy')
            owner=np.load(ROOT/f'data/control_tabpfn_fixed_confirmation_v2/{stage}/{protocol}/owner_probability.npy')
            fast=np.load(ROOT/f'data/control_boundary_v1/baseline_confirmation/{protocol}/fast_seed{seed}_probabilities.npy')
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            pred=independent_refine(baseline.prediction.to_numpy(),heads,owner,fast,raw,bc,blocks,plan['variants'][variant]['stages'],family=='bounded')
            np.testing.assert_array_equal(pred,table.prediction)
            metric=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(pred),bootstrap_n=0)
            name=f'{family}_{variant}_{seed_name}_{protocol}';labels=np.full(len(truth),'X',dtype='<U1');labels[public]=pred
            pd.DataFrame(dict(frame=np.arange(len(truth)),label=labels)).to_csv(scorer/(name+'.csv'),index=False)
            records.append(dict(name=name,source=str(path.relative_to(ROOT)),metrics=metric))
            if family=='bounded' and variant=='bounded_combined':
                assert metric['spell_f1_mean']==1. and all(metric['spell_f1'][t]['n_matched']==17 and metric['spell_f1'][t]['n_spurious']==0 for t in pm.SPELL_IOUS)
                old,new=overlap_summary(table.ground_truth,baseline.prediction),overlap_summary(table.ground_truth,pred)
                for i,(a,b,k) in enumerate(old['gt']):
                    v0=max((v for v,j,z in old['overlap'] if j==i),default=0.);v1=max((v for v,j,z in new['overlap'] if j==i),default=0.)
                    assert v1>=v0-1e-12
                    diagnostics.append(dict(seed=seed,protocol=protocol,start=int(public[a]),end_exclusive=int(public[b-1])+1,
                        team=('H','A','D','X')[k],frames=b-a,before_iou=v0,iou=v1))
    assert len(records)==30 and len(diagnostics)==153
    first,second=ROUND/'full_export1',ROUND/'full_export2'
    for name in ('candidate.csv','manifest.json','events.json','public_comparison.json'):
        assert (first/name).read_bytes()==(second/name).read_bytes(),name
    parent=pd.read_csv(ROOT/'data/candidate_tabpfn_stoppage_fragments_v1/candidate.csv');candidate=pd.read_csv(first/'candidate.csv')
    full_raw=pd.read_pickle(ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl')
    full_bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/full_inputs/broadcast_extension.pkl')
    heads,owner,fast=[np.load(ROOT/f'data/candidate_tabpfn_short_control_v1/{name}',allow_pickle=False) for name in
        ('heads_probabilities.npy','owner_probability.npy','fast_probabilities.npy')]
    from games import GAME,half_video_blocks
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    stages=json.loads((ROUND/'plan.json').read_text())['variants']['bounded_combined']['stages']
    expected=independent_refine(parent.label.to_numpy(),heads,owner,fast,full_raw,full_bc,blocks,stages,True)
    np.testing.assert_array_equal(expected,candidate.label);np.testing.assert_array_equal(candidate.label=='X',parent.label=='X')
    np.testing.assert_array_equal(candidate.label.iloc[public],parent.label.iloc[public])
    for name,expected_hash in json.loads((first/'manifest.json').read_text())['input_sha256'].items():assert digest(ROOT/name)==expected_hash,name
    for name,table in (('full_previous',parent),('full_candidate',candidate)):
        table.to_csv(scorer/(name+'.csv'),index=False)
        records.append(dict(name=name,metrics=pm.evaluate(pm.label_codes(truth),pm.label_codes(table.label),bootstrap_n=0)))
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/run_evaluation.py'),'--results',str(scorer),'--out',str(ROUND/'official_scoring'),
        '--bootstrap','0','--no-detail'],capture_output=True,text=True,cwd=ROOT)
    (ROUND/'official_scoring.log').write_text(result.stdout+result.stderr);assert result.returncode==0
    for record in records:
        official=json.loads((ROUND/'official_scoring/reports'/(record['name']+'.json')).read_text())
        for key in ('spell_f1_mean','macro_f1','accuracy'):assert abs(official[key]-record['metrics'][key])<=1e-12
        for t in pm.SPELL_IOUS:
            for key in ('f1','n_matched','n_spurious','n_missed'):assert official['spell_f1'][str(t)][key]==record['metrics']['spell_f1'][t][key]
    assert assess(ROUND/'candidates/bounded_combined')['passed']
    folds=pd.concat([pd.read_csv(ROUND/f'seed{seed}_fold_scores.csv').query("variant=='bounded_combined'") for seed in (42,43,44)],ignore_index=True)
    assert len(folds)==39 and folds.passed.all()
    folds.to_csv(ROUND/'fold_scores.csv',index=False);pd.DataFrame(diagnostics).to_csv(ROUND/'boundary_diagnostics.csv',index=False)
    assert json.loads((first/'public_comparison.json').read_text())['passed']
    for index,folder in enumerate((first,second),1):
        checked=subprocess.run([sys.executable,str(ROOT/'evaluation/validate_submission.py'),str(folder/'candidate.csv')],capture_output=True,text=True)
        (ROUND/f'full_validator{index}.log').write_text(checked.stdout+checked.stderr);assert checked.returncode==0
    verify()
    for name,expected_hash in json.loads((ROUND/'audit_plan.json').read_text())['source_sha256'].items():assert digest(ROOT/name)==expected_hash,name
    write(ROUND/'final_audit.json',dict(passed=True,independent_OOF_replays=30,official_scores=32,
        independent_full_reconstruction_exact=True,two_full_exports_byte_exact=True,official_full_validator_passed_twice=True,
        all_nine_OOF_spellF1=1.,all_nine_matched_true_possessions=17,all_nine_spurious=0,
        all_prior_match_flags_retained_at_all_five_IoUs=True,individual_true_overlaps_nonregressing=153,
        clipped_fold_comparisons=39,clipped_fold_regressions=0,full_public_labels_exact=True,
        full_changed_frames=int(np.sum(parent.label!=candidate.label)),full_csv_sha256=digest(first/'candidate.csv'),
        private_full_match_accuracy_unknown=True,new_model_fits=0,new_TabPFN_queries=0,
        source_sha256=json.loads((ROUND/'audit_plan.json').read_text())['source_sha256']))
    print((ROUND/'final_audit.json').read_text(),flush=True)


if __name__=='__main__':audit()
