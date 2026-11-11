"""Independent segmentation,inference and official scoring of every retained trial."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from reproduce_spurious_possessions import BASE,assess
from reproduce_boundary_precision import PROTOCOLS,overlap_summary
from research_supported_gap_stoppage import ROUND,verify
from research_spurious_possessions import digest,write

FAMILIES=(('initial','control_spurious_possessions_v1'),('predicted','control_spurious_gap_closure_v1'),
    ('visible','control_observed_gap_stoppage_v1'),('dual','control_dual_anchor_stoppage_v1'),
    ('supported','control_supported_gap_stoppage_v1'))


def segments(values,left,right):
    result=[];cursor=left
    while cursor<right:
        stop=cursor+1
        while stop<right and values[stop]==values[cursor]:stop+=1
        result.append((cursor,stop,str(values[cursor])));cursor=stop
    return result


def independent_refine(reference,heads,raw,bc,blocks,family,variant):
    reference=np.asarray(reference);out=reference.copy()
    distances=bc[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    camera=(bc.bc_matched.to_numpy()==1)&(bc.bc_same_ball_seen.to_numpy()==1)&np.isfinite(distances).all(axis=1)
    tactical=raw.ball_seen.to_numpy()>0;moving=raw.ball_displacement_1s.to_numpy()>4.5
    observed=camera|tactical
    invisible=family!='initial' or variant!='observed_stoppage'
    stationary=family!='initial' or variant!='invisible_continuation'
    allowance=13 if variant in ('gap13','source_gap13') else 25
    for left,right in blocks:
        runs=segments(reference,left,right)
        for i,(start,end,team) in enumerate(runs):
            if i==0 or i==len(runs)-1 or team not in ('H','A') or not 1<=end-start<=125:continue
            before,after=runs[i-1][2],runs[i+1][2]
            if before!='D' and after!='D' or camera[start:end].any():continue
            opponent='A' if team=='H' else 'H'
            if invisible and not tactical[start:end].any() and opponent in (before,after):
                dead_start=dead_end=start
                if family!='initial':
                    if before=='D':
                        p0,p1=start,min(end,start+allowance)
                        cursor=end
                        while cursor<min(right,end+allowance) and not observed[cursor]:cursor+=1
                        v0,v1=start,max(start,min(end,cursor-allowance))
                    else:
                        p0,p1=max(start,end-allowance),end
                        cursor=start
                        while cursor>max(left,start-allowance) and not observed[cursor-1]:cursor-=1
                        v0,v1=min(end,max(start,cursor+allowance)),end
                    if family=='predicted':dead_start,dead_end=p0,p1
                    elif family=='visible':dead_start,dead_end=v0,v1
                    else:dead_start,dead_end=min(p0,v0),max(p1,v1)
                if family=='supported' and end-start-(dead_end-dead_start)<8:continue
                out[start:end]=opponent
                if family!='initial':out[dead_start:dead_end]='D'
            elif (stationary and tactical[start:end].mean()>=.5 and moving[start:end].mean()<=.5
                  and max(heads[start:end,0].mean(),heads[start:end,1].mean())>.5):
                out[start:end]='D'
    return out


def audit():
    verify();public=np.arange(11185,18685)
    sources=['experiments/audit_stoppage_fragments.py','experiments/test_stoppage_fragment_audit.py',
        'experiments/build_stoppage_fragment_candidate.py','evaluation/possession_metrics.py','evaluation/run_evaluation.py']
    for family,folder in FAMILIES:
        sources+=list(json.loads((ROOT/'data'/folder/'plan.json').read_text())['source_sha256'])
    write(ROUND/'audit_plan.json',dict(source_sha256={name:digest(ROOT/name) for name in sorted(set(sources))},
        expected_OOF_outputs=39,expected_official_scores=41,independent_full_reconstruction=True))
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl').iloc[public].reset_index(drop=True)
    bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl').iloc[public].reset_index(drop=True)
    truth=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    scorer=ROUND/'official_inputs';scorer.mkdir();records=[];diagnostics=[]
    for family,folder in FAMILIES:
        root=ROOT/'data'/folder/'candidates'
        for path in sorted(root.glob('*/seed*/*/oof_predictions.csv')):
            variant,seed_name,protocol=path.relative_to(root).parts[:3];seed=int(seed_name[4:])
            table=pd.read_csv(path);baseline=pd.read_csv(BASE/seed_name/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(table.frame,public);np.testing.assert_array_equal(table.ground_truth,truth[public])
            heads=np.load(ROOT/f'data/control_broadcast_v1/baseline_confirmation/{protocol}/heads_seed{seed}_probabilities.npy')
            blocks=[(s['validation'][0]-11185,s['validation'][-1]+1-11185) for s in splits if s['protocol']==protocol]
            pred=independent_refine(baseline.prediction.to_numpy(),heads,raw,bc,blocks,family,variant)
            np.testing.assert_array_equal(pred,table.prediction)
            metric=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(pred),bootstrap_n=0)
            name=f'{family}_{variant}_{seed_name}_{protocol}'
            labels=np.full(len(truth),'X',dtype='<U1');labels[public]=pred
            pd.DataFrame(dict(frame=np.arange(len(truth)),label=labels)).to_csv(scorer/(name+'.csv'),index=False)
            records.append(dict(name=name,source=str(path.relative_to(ROOT)),metrics=metric))
            if family=='supported':
                if protocol=='buffered_minutes':
                    assert all(metric['spell_f1'][t]['n_matched']==17 and metric['spell_f1'][t]['n_spurious']==0 for t in pm.SPELL_IOUS)
                old,new=overlap_summary(table.ground_truth,baseline.prediction),overlap_summary(table.ground_truth,pred)
                for i,(start,end,team) in enumerate(old['gt']):
                    a=max((v for v,j,_ in old['overlap'] if j==i),default=0.)
                    b=max((v for v,j,_ in new['overlap'] if j==i),default=0.)
                    diagnostics.append(dict(seed=seed,protocol=protocol,start=int(public[start]),end_exclusive=int(public[end-1])+1,
                        team=('H','A','D','X')[team],frames=end-start,before_iou=a,iou=b))
    assert len(records)==39
    first,second=ROUND/'full_export1',ROUND/'full_export2'
    for name in ('candidate.csv','events.json','manifest.json','public_comparison.json'):
        assert (first/name).read_bytes()==(second/name).read_bytes(),name
    parent=pd.read_csv(ROOT/'data/candidate_tabpfn_home_arrivals_v1/candidate.csv');candidate=pd.read_csv(first/'candidate.csv')
    full_raw=pd.read_pickle(ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl')
    full_bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/full_inputs/broadcast_extension.pkl')
    full_heads=np.load(ROOT/'data/candidate_tabpfn_short_control_v1/heads_probabilities.npy',allow_pickle=False)
    from games import GAME,half_video_blocks
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    expected=independent_refine(parent.label.to_numpy(),full_heads,full_raw,full_bc,blocks,'supported','supported_gap25')
    np.testing.assert_array_equal(expected,candidate.label)
    np.testing.assert_array_equal(candidate.label=='X',parent.label=='X')
    assert np.all(candidate.label[parent.label=='D']=='D')
    for name,expected_hash in json.loads((first/'manifest.json').read_text())['input_sha256'].items():
        assert digest(ROOT/name)==expected_hash,name
    for name,table in (('full_previous',parent),('full_candidate',candidate)):
        table.to_csv(scorer/(name+'.csv'),index=False)
        metric=pm.evaluate(pm.label_codes(truth),pm.label_codes(table.label),bootstrap_n=0)
        records.append(dict(name=name,metrics=metric))
    response=subprocess.run([sys.executable,str(ROOT/'evaluation/run_evaluation.py'),'--results',str(scorer),
        '--out',str(ROUND/'official_scoring'),'--bootstrap','0','--no-detail'],capture_output=True,text=True,cwd=ROOT)
    (ROUND/'official_scoring.log').write_text(response.stdout+response.stderr)
    assert response.returncode==0
    for record in records:
        official=json.loads((ROUND/'official_scoring/reports'/(record['name']+'.json')).read_text())
        for key in ('spell_f1_mean','macro_f1','accuracy'):
            assert abs(record['metrics'][key]-official[key])<=1e-12,(record['name'],key)
    assert assess(ROUND/'candidates/supported_gap25')['passed']
    folds=pd.concat([pd.read_csv(ROUND/f'seed{seed}_fold_scores.csv') for seed in (42,43,44)],ignore_index=True)
    assert len(folds)==39 and folds.passed.all()
    folds.to_csv(ROUND/'fold_scores.csv',index=False);pd.DataFrame(diagnostics).to_csv(ROUND/'boundary_diagnostics.csv',index=False)
    comparison=json.loads((first/'public_comparison.json').read_text());assert comparison['passed']
    np.testing.assert_array_equal(candidate.label.iloc[public],parent.label.iloc[public])
    for index,folder in enumerate((first,second),1):
        result=subprocess.run([sys.executable,str(ROOT/'evaluation/validate_submission.py'),str(folder/'candidate.csv')],capture_output=True,text=True)
        (ROUND/f'full_validator{index}.log').write_text(result.stdout+result.stderr);assert result.returncode==0
    main=pd.read_csv(BASE/'seed42/buffered_minutes/oof_predictions.csv')
    overlap=overlap_summary(main.ground_truth,main.prediction);extras=[]
    for j,(start,end,team) in enumerate(overlap['predicted']):
        best=max((v for v,i,k in overlap['overlap'] if k==j),default=0.)
        if best<.3:extras.append(dict(start=int(main.frame.iloc[start]),end_exclusive=int(main.frame.iloc[end-1])+1,
            team=('H','A','D','X')[team],frames=end-start,best_iou=best,truth=main.ground_truth.iloc[start:end].value_counts().to_dict()))
    assert len(extras)==3;write(ROUND/'main_spurious_diagnostics.json',extras)
    verify()
    for name,expected_hash in json.loads((ROUND/'audit_plan.json').read_text())['source_sha256'].items():
        assert digest(ROOT/name)==expected_hash,name
    write(ROUND/'final_audit.json',dict(passed=True,independent_OOF_replays=39,official_scores=41,
        independent_full_reconstruction_exact=True,two_full_exports_byte_exact=True,official_full_validator_passed_twice=True,
        all_nine_score_and_target_checks_pass=True,all_prior_matches_retained_at_all_five_IoUs=True,
        clipped_fold_comparisons=39,clipped_fold_regressions=0,main_spurious=0,main_true_matches=17,
        main_spellF1=1.,all_three_main_seeds_zero_spurious=True,no_added_false_D_all_nine=True,
        full_changed_frames=int(np.sum(parent.label!=candidate.label)),full_public_labels_exact=True,
        full_csv_sha256=digest(first/'candidate.csv'),private_full_match_accuracy_unknown=True,
        new_model_fits=0,new_TabPFN_queries=0,source_sha256=json.loads((ROUND/'audit_plan.json').read_text())['source_sha256']))
    print((ROUND/'final_audit.json').read_text(),flush=True)


if __name__=='__main__':audit()
