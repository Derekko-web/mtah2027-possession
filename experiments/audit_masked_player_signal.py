"""Native scalar-ownership opportunity; all cases retained, selected D/X fixed."""
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
from audit_pathcrf_signal import overlay
from research_temporal_dead import write_json

FOLDER=ROOT/'data/control_masked_player_v1'
PLAN=ROOT/'experiments/MASKED_PLAYER_SIGNAL_PLAN.json'


def main(output):
    plan=json.loads(PLAN.read_text())
    if PLAN.read_bytes()!=(FOLDER/'protocol_freeze_D/plan.json').read_bytes():raise ValueError('Native scalar audit plan changed')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Native scalar audit input changed')
    active=FOLDER/'protocol_freeze_C/implementation_execution_revision1.json'
    for name,expected in json.loads(active.read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Active native source changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    full=pd.read_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy();public=np.arange(11185,18685)
    np.testing.assert_array_equal(np.flatnonzero(full!='X'),public);labels=full[public]
    frames=np.load(FOLDER/'native_outputs/frames.npy');p=np.load(FOLDER/'native_outputs/probability.npy');supported=np.load(FOLDER/'native_inputs/eligible.npy')
    np.testing.assert_array_equal(frames,np.arange(11185,18685,5));relative=frames-11185
    if p.shape!=frames.shape or not np.isfinite(p).all():raise ValueError('Invalid native scalar outputs')
    prediction=np.where(p>=.5,'H','A');confidence=np.maximum(p,1-p)
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text());protocols={};new_cases=[]
    for protocol in ('buffered_minutes','purged_spells','buffered_100_seconds'):
        baseline=pd.read_csv(ROOT/'data/control_dfl_dead_v1/public_trial'/protocol/'oof_predictions.csv',usecols=['frame','current_seed42'])
        np.testing.assert_array_equal(baseline.frame,public);reference=baseline.current_seed42.to_numpy();old=reference[relative];target=labels[relative]
        editable=supported&np.isin(old,['H','A']);known=editable&np.isin(target,['H','A'])
        recalls={k:float((prediction[known&(target==k)]==k).mean()) if (known&(target==k)).any() else None for k in ('H','A')}
        balanced=sum(recalls.values())/2 if all(v is not None for v in recalls.values()) else None
        qualified=editable&(confidence>=.65);changes=qualified&(prediction!=old)
        corrected=int((changes&(prediction==target)).sum());damage=int((changes&(old==target)).sum());neutral=int((changes&(target=='D')).sum())
        precision=corrected/(corrected+damage) if corrected+damage else None
        diagnostic=reference.copy()
        for s in splits:
            if s['protocol']!=protocol:continue
            a,b=min(s['validation']),max(s['validation'])+1
            q=qualified&(frames>=a)&(frames<b)
            partial=overlay(reference,frames,prediction,q);where=(public>=a)&(public<b);diagnostic[where]=partial[where]
        before=matching(labels,reference);after=matching(labels,diagnostic);np.testing.assert_array_equal(before['truth'],after['truth'])
        short=np.array([b-a<=125 for a,b,_ in before['truth']]);new=np.flatnonzero(~before['gt_flags']&after['gt_flags']&short).tolist();new_cases.append(set(new))
        retained=bool(np.all(after['gt_flags'][before['gt_flags']]));spurious0=int((~before['pred_flags']).sum());spurious1=int((~after['pred_flags']).sum())
        checks=dict(balanced=balanced is not None and balanced>=.70,corrections=corrected>=25,precision=precision is not None and precision>=.55,
            net=corrected-damage>=25,prior_matches=retained,no_extra_spurious=spurious1<=spurious0)
        cases=[]
        for i,(a,b,label) in enumerate(before['truth']):
            group=(relative>=a)&(relative<b)
            cases.append(dict(gt_id=i,frames=b-a,label=label,short=bool(short[i]),sampled_centers=int(group.sum()),physical_support=int((group&supported).sum()),
                editable_centers=int((group&editable).sum()),confident_centers=int((group&qualified).sum()),correct_centers=int((group&editable&(prediction==labels[a])).sum()),
                reference_matched=bool(before['gt_flags'][i]),diagnostic_matched=bool(after['gt_flags'][i])))
        if sum(c['sampled_centers'] for c in cases)!=int(np.isin(target,['H','A']).sum()):raise ValueError('Case denominator omitted unsupported samples')
        spurious=[]
        for i,(a,b,label) in enumerate(before['predicted']):
            if before['pred_flags'][i]:continue
            group=(relative>=a)&(relative<b)
            spurious.append(dict(frames=b-a,label=label,sampled_centers=int(group.sum()),physical_support=int((group&supported).sum()),confident_centers=int((group&qualified).sum())))
        protocols[protocol]=dict(supported_editable_true_live=int(known.sum()),home_recall=recalls['H'],away_recall=recalls['A'],balanced_accuracy=balanced,
            confident_centers=int(qualified.sum()),corrected=corrected,damage=damage,neutral_D_changes=neutral,correction_precision=precision,net_corrected=corrected-damage,
            new_short_ids=new,spurious_before=spurious0,spurious_after=spurious1,retained_all_prior_matches=retained,checks=checks,all_truth_cases=cases,all_spurious_cases=spurious)
        pd.DataFrame(dict(frame=public,reference=reference,diagnostic=diagnostic)).to_csv(output/(protocol+'_diagnostic.csv'),index=False)
    common=sorted(set.intersection(*new_cases));result=dict(passed=all(all(v['checks'].values()) for v in protocols.values()) and bool(common),common_new_short_ids=common,
        selected_external_arm=plan['selected'],protocols=protocols,diagnostic_only=True,new_fits=0,new_possession_score_rows=0,manual_annotations_used=False)
    write_json(output/'summary.json',result)
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),source_sha256=digest(Path(__file__)),input_sha256=plan['input_sha256'],
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    compact={k:{a:b for a,b in v.items() if a not in ('all_truth_cases','all_spurious_cases')} for k,v in protocols.items()}
    print(json.dumps(dict(passed=result['passed'],common_new_short_ids=common,protocols=compact),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=FOLDER/'signal_audit')
    main(p.parse_args().output)
