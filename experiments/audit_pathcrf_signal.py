"""Frozen raw ownership/case audit after immutable target-free PathCRF outputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from dfl_signal_audit import matching
from research_temporal_dead import write_json

FOLDER=ROOT/'data/control_pathcrf_v1'


def overlay(reference,frames,prediction,qualified):
    result=reference.copy();i=0
    while i<len(frames):
        if not qualified[i]:i+=1;continue
        j=i+1
        while j<len(frames) and qualified[j] and prediction[j]==prediction[i] and frames[j]-frames[j-1]==5:j+=1
        if j-i>=3:
            for c in frames[i:j]:
                a=max(0,int(c)-2-11185);b=min(len(result),int(c)+3-11185)
                use=np.isin(reference[a:b],['H','A']);result[a:b][use]=prediction[i]
        i=j
    np.testing.assert_array_equal(result=='D',reference=='D');np.testing.assert_array_equal(result=='X',reference=='X')
    return result


def main(output):
    plan_path=ROOT/'experiments/PATHCRF_SIGNAL_PLAN.json';plan=json.loads(plan_path.read_text())
    if plan_path.read_bytes()!=(FOLDER/'protocol_freeze_C/plan.json').read_bytes():raise ValueError('Native signal plan changed')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Native signal input changed')
    for name,expected in json.loads((FOLDER/'protocol_freeze_C/implementation_before_outputs.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Native inference source changed')
    for name,expected in json.loads((FOLDER/'native_outputs/manifest.json').read_text())['cache_sha256'].items():
        if digest(FOLDER/'native_outputs'/name)!=expected:raise ValueError('Saved target-free output changed')
    output.mkdir(parents=True,exist_ok=False)
    # First true-target join happens only after both raw arms are immutable.
    labels=pd.read_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    public=np.arange(11185,18685);np.testing.assert_array_equal(np.flatnonzero(labels!='X'),public)
    labels=labels[public];frames=np.load(FOLDER/'native_outputs/frames.npy');home=np.load(FOLDER/'observation_support/left_team_is_home.npy')
    uniform=np.arange(11185,18685,5)-11185
    scores={};selection=[]
    for arm in ('temporal','repeat'):
        with np.load(FOLDER/'native_outputs'/(arm+'.npz')) as raw:
            source=raw['sender'];mass=raw['center_mass']
        live_mass=mass[:,:2].sum(axis=1)
        pleft=np.divide(mass[:,0],live_mass,out=np.full(len(mass),.5),where=live_mass>0)
        phome=np.where(home,pleft,1-pleft);is_home=np.where(source<11,home,~home)
        pred=np.where(source<22,np.where(is_home,'H','A'),'X')
        confidence=np.where(pred=='H',phome,1-phome)
        supported_confidence=(source<22)&(mass[:,2]<=.10)&(confidence>=.65)
        scores[arm]={};common=[];net=[]
        for protocol in ('buffered_minutes','purged_spells','buffered_100_seconds'):
            eligible=np.isin(frames,np.load(FOLDER/'observation_support'/(protocol+'_eligible_frames.npy')))
            baseline=pd.read_csv(ROOT/'data/control_dfl_dead_v1/public_trial'/protocol/'oof_predictions.csv',usecols=['frame','current_seed42'])
            np.testing.assert_array_equal(baseline.frame,public);reference=baseline.current_seed42.to_numpy()
            target=labels[frames-11185];old=reference[frames-11185]
            known=eligible&np.isin(target,['H','A']);recall={k:float((pred[known&(target==k)]==k).mean()) if (known&(target==k)).any() else None for k in ('H','A')}
            balanced=sum(recall.values())/2 if all(v is not None for v in recall.values()) else None
            qualified=eligible&supported_confidence;change=qualified&(pred!=old)
            corrected=int((change&(pred==target)).sum());damage=int((change&(old==target)).sum());neutral=int((change&(target=='D')).sum())
            precision=corrected/(corrected+damage) if corrected+damage else None
            diagnostic=overlay(reference,frames,pred,qualified)
            before=matching(labels,reference);after=matching(labels,diagnostic)
            np.testing.assert_array_equal(before['truth'],after['truth'])
            truth=before['truth'];short=np.array([b-a<=125 for a,b,_ in truth])
            newly=np.flatnonzero(~before['gt_flags']&after['gt_flags']&short).tolist()
            retained=bool(np.all(after['gt_flags'][before['gt_flags']]))
            spurious0=int((~before['pred_flags']).sum());spurious1=int((~after['pred_flags']).sum())
            checks=dict(balanced=balanced is not None and balanced>=.70,corrections=corrected>=25,
                precision=precision is not None and precision>=.55,net=corrected-damage>=25,
                prior_matches=retained,no_extra_spurious=spurious1<=spurious0)
            cases=[]
            for case_id,(a,b,label) in enumerate(truth):
                group=(frames-11185>=a)&(frames-11185<b)
                cases.append(dict(gt_id=case_id,frames=b-a,label=label,short=bool(short[case_id]),
                    sampled_centers=int(((uniform>=a)&(uniform<b)).sum()),supported_centers=int((group&eligible).sum()),confident_centers=int((group&qualified).sum()),
                    correct_centers=int((group&eligible&(pred==labels[a])).sum()),reference_matched=bool(before['gt_flags'][case_id]),diagnostic_matched=bool(after['gt_flags'][case_id])))
            false_cases=[]
            for j,(a,b,label) in enumerate(before['predicted']):
                if before['pred_flags'][j]:continue
                group=(frames-11185>=a)&(frames-11185<b)
                false_cases.append(dict(frames=b-a,label=label,sampled_centers=int(((uniform>=a)&(uniform<b)).sum()),supported_centers=int((group&eligible).sum()),confident_centers=int((group&qualified).sum())))
            if sum(c['sampled_centers'] for c in cases)!=int(np.isin(labels[uniform],['H','A']).sum()):raise ValueError('Unsupported cases disappeared from sampling denominator')
            scores[arm][protocol]=dict(supported_true_live_centers=int(known.sum()),home_recall=recall['H'],away_recall=recall['A'],balanced_accuracy=balanced,
                confident_centers=int(qualified.sum()),corrected=corrected,damage=damage,neutral_D_changes=neutral,correction_precision=precision,net_corrected=corrected-damage,
                new_short_ids=newly,retained_all_prior_matches=retained,spurious_before=spurious0,spurious_after=spurious1,checks=checks,
                all_truth_cases=cases,all_spurious_cases=false_cases)
            pd.DataFrame(dict(frame=public,reference=reference,diagnostic=diagnostic)).to_csv(output/(arm+'_'+protocol+'_diagnostic.csv'),index=False)
            common.append(set(newly));net.append(corrected-damage)
        shared=sorted(set.intersection(*common));passed=all(all(v['checks'].values()) for v in scores[arm].values()) and bool(shared)
        selection.append(dict(arm=arm,passed=passed,common_new_short=shared,min_net_corrected=min(net)))
    passing=sorted([x for x in selection if x['passed']],key=lambda x:(-len(x['common_new_short']),-x['min_net_corrected'],0 if x['arm']=='temporal' else 1))
    result=dict(passed=bool(passing),selected=passing[0]['arm'] if passing else None,arms=selection,protocols=scores,
        diagnostic_only=True,new_fits=0,new_possession_score_rows=0,manual_labels_used=False,private_targets_used=False)
    write_json(output/'summary.json',result)
    write_json(output/'manifest.json',dict(plan_sha256=digest(plan_path),source_sha256=digest(Path(__file__)),
        input_sha256=plan['input_sha256'],native_output_manifest_sha256=digest(FOLDER/'native_outputs/manifest.json'),
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    compact={arm:{p:{k:v for k,v in row.items() if k not in ('all_truth_cases','all_spurious_cases')} for p,row in values.items()} for arm,values in scores.items()}
    print(json.dumps(dict(passed=result['passed'],selected=result['selected'],arms=selection,protocols=compact),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=FOLDER/'signal_audit')
    main(p.parse_args().output)
