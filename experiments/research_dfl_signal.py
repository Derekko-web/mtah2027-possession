"""Frozen cached-teacher signal and target-aware pipeline capacity diagnostic."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from dfl_transfer_data import digest
from dfl_transfer_features import geometry
from dfl_transfer_teacher import probability
from dfl_signal_audit import (same,matching,best_iou,interval_iou,teacher_diagnostic,
    conditional_probability,oracle_probability,keep_reference_dead_unknown,
    confidence_summary,oracle_effect,continuation,array_digest,native_prefix_lengths)
from broadcast_consensus import decode_broadcast_consensus
from broadcast_intersection import intersect_broadcast
from motion_dead_trim import decode_motion_dead_trim
from pair_motion_inference import head_observed
from research_ownership_graph import selected_arrays
from stoppage_consensus import decode_stoppage_consensus
from finalize_broadcast_closure import audit_manifest
from games import GAME,GAMES,half_video_blocks

ROUND=ROOT/'data/control_dfl_signal_v1'
PLAN=ROOT/'experiments/DFL_SIGNAL_PLAN.json'


def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=json_default)+'\n')


def validate():
    plan=json.loads(PLAN.read_text());freeze=ROUND/'protocol_freeze'
    same(np.frombuffer(PLAN.read_bytes(),np.uint8),np.frombuffer((freeze/'plan.json').read_bytes(),np.uint8))
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen diagnostic input changed: '+name)
    for name,expected in json.loads((freeze/'implementation.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen diagnostic source changed: '+name)
    for name,expected in json.loads((freeze/'protected_artifacts.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Protected artifact changed: '+name)
    if (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes()!=(freeze/'ledger_before.csv').read_bytes():
        raise ValueError('Diagnostic changed existing ledger')
    for name,target in json.loads((freeze/'symlinks.json').read_text()).items():
        p=ROOT/name
        if not p.is_symlink() or str(p.readlink())!=target:raise ValueError('Source symlink changed')
    return plan


def verify_parent(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for key,base in (('source_sha256',ROOT),('input_sha256',ROOT),('cache_sha256',folder)):
        for name,expected in manifest.get(key,{}).items():
            if digest(base/name)!=expected:raise ValueError('Parent identity changed: '+name)
    return manifest


def trace_closure(arrays,bc,raw,blocks,branch):
    current,previous,fast,heads,joint=arrays
    base=decode_stoppage_consensus(current,previous,fast,heads,raw,blocks)
    learned=decode_broadcast_consensus(current,previous,fast,heads,joint,bc,raw,blocks,True,'replace_head')
    output=base.copy();records=[]
    for left,right in blocks:
        r=pm.label_codes(base[left:right]);p=pm.label_codes(learned[left:right])
        result,details=intersect_broadcast(r,p,bc.iloc[left:right],False,True)
        output[left:right]=np.array(['H','A','D'])[result]
        for row in details:
            records.append(dict(branch=branch,block_left=left,start=left+row['start'],
                end_exclusive=left+row['end_exclusive'],team=row['team'],accepted=row['accepted'],first_reason=row['reason']))
    return output,records


def trace_selected(arrays,bc,info,raw,blocks):
    reference,records=trace_closure(arrays[:5],bc,raw,blocks,'reference7')
    alternative_arrays=list(arrays[:5])
    observed=head_observed(bc,info,'broadcast_motion')
    alternative_arrays[0]=np.where(observed[:,None],arrays[5],arrays[0])
    alternative,other=trace_closure(alternative_arrays,bc,raw,blocks,'motion_alternative')
    output=reference.copy();trim=(reference=='D')&np.isin(alternative,['H','A'])
    output[trim]=alternative[trim]
    actual=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
    same(output,actual)
    return output,records+other,dict(reference7=array_digest(reference),motion_alternative=array_digest(alternative),
        full_selected=array_digest(output),motion_trim_frames=int(trim.sum()))


def main(output):
    plan=validate();output.mkdir()
    (output/'plan.json').write_bytes(PLAN.read_bytes())
    parent=ROOT/'data/control_dfl_transfer_v1'
    verify_parent(parent/'external_trial');verify_parent(parent/'public_trial')
    for name,expected in json.loads((parent/'public_dispatch.json').read_text())['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Public parent dispatch changed: '+name)
    upstream=ROOT/'data/control_pair_motion_v1';camera=ROOT/'data/control_broadcast_v1/inputs'
    audit_manifest(camera);audit_manifest(upstream/'inputs');audit_manifest(upstream/'dead_trim_confirmation')
    source=ROOT/'data/advanced_validation'
    raw=pd.read_pickle(source/'raw.pkl');n=len(raw)
    bc=pd.read_pickle(camera/'combined_inputs.pkl').iloc[:n]
    info=pd.read_pickle(upstream/'inputs/motion_info.pkl').iloc[:n]
    cached=pd.read_pickle(parent/'public_trial/matched_control_inputs.pkl')
    teacher_cached=pd.read_pickle(parent/'public_trial/external_teacher_inputs.pkl')
    with np.load(camera/'tactical_observations.npz') as data:
        positions,ball,teams,valid=[data[k] for k in ('positions','ball','teams','valid')]
    lengths=native_prefix_lengths(n,len(ball));long_rows=lengths['long_rows'];short_rows=lengths['short_rows']
    positions=positions[:long_rows].copy();ball=ball[:long_rows].copy();valid=valid[:long_rows]
    positions[~valid]=np.nan;ball[~valid]=np.nan;pitch=np.array(GAMES[GAME]['pitch'])
    ball[~(np.isfinite(ball).all(axis=1)&(ball>=-5).all(axis=1)&(ball<=pitch+5).all(axis=1))]=np.nan
    halves=[(start,end+1) for _,start,end in half_video_blocks(GAME)]
    full=geometry(positions,ball,teams,halves)
    short=geometry(positions[:short_rows],ball[:short_rows],teams,halves)
    pd.testing.assert_frame_equal(short,cached,check_exact=True)
    pd.testing.assert_frame_equal(full.iloc[11185:18685],short.iloc[11185:18685],check_exact=True)
    model=XGBClassifier();model.load_model(parent/'external_trial/teacher_seed42.ubj')
    long_p=probability(model,full);teacher=probability(model,short)
    same(teacher,teacher_cached.h__teacher.to_numpy());same(long_p[11185:18685],teacher[11185:18685])
    short.to_pickle(output/'native_features.pkl');full.to_pickle(output/'native_features_long.pkl')
    np.save(output/'native_teacher_probability.npy',teacher);np.save(output/'native_teacher_long_probability.npy',long_p)
    print('Frozen native features, teacher checkpoint and actual-prefix identity verified',flush=True)
    public=np.arange(11185,18685)
    splits=json.loads((source/'splits.json').read_text());(output/'splits.json').write_bytes((source/'splits.json').read_bytes())
    baselines={};fingerprints={};identity_controls=0
    for protocol in plan['protocols']:
        folds=[f for f in splits if f['protocol']==protocol]
        blocks=[(min(f['validation']),max(f['validation'])+1) for f in folds]
        arrays,identities=selected_arrays(protocol,42,n)
        before_hashes=[array_digest(a) for a in arrays];fingerprints.update(identities)
        reference,trace,branches=trace_selected(arrays,bc,info,raw,blocks)
        old=pd.read_csv(upstream/'dead_trim_confirmation'/protocol/'oof_predictions.csv',usecols=['motion_dead_trim_seed42'])
        same(reference[public],old.motion_dead_trim_seed42.to_numpy())
        same(reference[public],pd.read_csv(parent/'public_trial'/protocol/'oof_predictions.csv',usecols=['current_seed42']).current_seed42.to_numpy())
        for slot in (0,4):
            modified=list(arrays);modified[slot]=arrays[slot].copy()
            identity=decode_motion_dead_trim(*modified,bc,info,raw,blocks)
            same(keep_reference_dead_unknown(reference,identity),reference);identity_controls+=1
            modified[slot]=conditional_probability(arrays[slot],arrays[slot])
            identity=decode_motion_dead_trim(*modified,bc,info,raw,blocks)
            same(keep_reference_dead_unknown(reference,identity),reference);identity_controls+=1
        baselines[protocol]=(arrays,before_hashes,blocks,reference,trace,branches)
    print('All twelve full selected8 identity controls passed before target diagnostics',flush=True)
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    same(np.flatnonzero(labels!='X'),public)
    protocols={};all_traces=[];oracle_records=[];oracle_count=0;truth_identity=None
    for protocol in plan['protocols']:
        folder=output/protocol;folder.mkdir()
        arrays,before_hashes,blocks,reference,trace,branches=baselines[protocol]
        for row in trace:all_traces.append(dict(protocol=protocol,probe='baseline',slot=None,**row))
        state=matching(labels[public],reference[public]);confidence=confidence_summary(labels[public],reference[public],teacher[public])
        if len(state['truth'])!=17:raise ValueError('Known public truth spell universe differs')
        if truth_identity is None:truth_identity=state['truth']
        elif state['truth']!=truth_identity:raise ValueError('Global case identity changed across protocols')
        diagnostic=teacher_diagnostic(reference,teacher,blocks);teacher_state=matching(labels[public],diagnostic[public])
        adapted={head:np.load(parent/'public_trial'/protocol/f'{head}_seed42_probabilities.npy') for head in ('matched_control','external_teacher')}
        cases=[];spurious=[]
        for gt_id,(left,right,owner) in enumerate(state['truth']):
            frames=public[left:right];probability_correct=teacher[frames] if owner==0 else 1-teacher[frames]
            case=dict(gt_id=gt_id,start=int(frames[0]),end_exclusive=int(frames[-1]+1),frames=right-left,owner=owner,
                reference_matched=bool(state['gt_flags'][gt_id]),reference_best_iou=best_iou(state,gt_id),
                teacher_matched=bool(teacher_state['gt_flags'][gt_id]),teacher_best_iou=best_iou(teacher_state,gt_id),
                teacher_correct_probability=float(probability_correct.mean()),teacher_raw_correct_fraction=float(((teacher[frames]>=.5)==(owner==0)).mean()),
                ball_seen_fraction=float(full.ball_seen.iloc[frames].mean()),
                adapter_raw_correct_fraction={head:float((p[left:right].argmax(axis=1)==owner).mean()) for head,p in adapted.items()},oracle={})
            cases.append(case)
            if case['reference_matched']:continue
            for slot in (0,4):
                changed=list(arrays);changed[slot]=oracle_probability(arrays[slot],frames,np.full(len(frames),owner))
                alternate,new_trace,new_branches=trace_selected(changed,bc,info,raw,blocks)
                guarded=keep_reference_dead_unknown(reference,alternate)
                after=matching(labels[public],guarded[public]);effect=oracle_effect(state,after,gt_id)
                effect.update(best_iou=best_iou(after,gt_id),changed_live_frames=int((guarded[public]!=reference[public]).sum()),
                    matched_pairs=after['pairs'],output_sha256=array_digest(guarded),branch_hashes=new_branches)
                case['oracle'][str(slot)]=effect;oracle_count+=1
                for row in new_trace:all_traces.append(dict(protocol=protocol,probe=f'gt_{gt_id}',slot=slot,**row))
                oracle_records.append(dict(protocol=protocol,kind='missed_gt',case_id=gt_id,slot=slot,**effect))
        for pred_id,(left,right,owner) in enumerate(state['predicted']):
            if state['pred_flags'][pred_id]:continue
            frames=public[left:right];live=np.isin(labels[frames],['H','A'])
            eligible=frames[live];target=np.where(labels[eligible]=='H',0,1)
            wrong=eligible[labels[eligible]!=np.array(['H','A'])[owner]]
            case=dict(pred_id=pred_id,start=int(frames[0]),end_exclusive=int(frames[-1]+1),frames=right-left,owner=owner,
                true_live_frames=len(eligible),wrong_live_frames=len(wrong),true_dead_unknown_frames=int((~live).sum()),
                teacher_correct_fraction_on_wrong_live=float((np.where(teacher[wrong]>=.5,'H','A')==labels[wrong]).mean()) if len(wrong) else None,
                teacher_disagreement_fraction=float((np.where(teacher[frames]>=.5,0,1)!=owner).mean()),oracle={})
            for slot in (0,4):
                changed=list(arrays);changed[slot]=oracle_probability(arrays[slot],eligible,target)
                alternate,new_trace,new_branches=trace_selected(changed,bc,info,raw,blocks)
                guarded=keep_reference_dead_unknown(reference,alternate);after=matching(labels[public],guarded[public])
                effect=oracle_effect(state,after)
                remains=[span for j,span in enumerate(after['predicted']) if not after['pred_flags'][j] and span[2]==owner]
                effect.update(same_owner_spurious_overlap=max((interval_iou((left,right),span) for span in remains),default=0.),
                    changed_live_frames=int((guarded[public]!=reference[public]).sum()),matched_pairs=after['pairs'],
                    output_sha256=array_digest(guarded),branch_hashes=new_branches)
                case['oracle'][str(slot)]=effect;oracle_count+=1
                for row in new_trace:all_traces.append(dict(protocol=protocol,probe=f'spurious_{pred_id}',slot=slot,**row))
                oracle_records.append(dict(protocol=protocol,kind='spurious_pred',case_id=pred_id,slot=slot,**effect))
            spurious.append(case)
        same(np.array([array_digest(a) for a in arrays]),np.array(before_hashes))
        metrics=dict(confidence=confidence,truth_cases=cases,spurious_cases=spurious,
            baseline_gt_spells=len(state['truth']),baseline_pred_spells=len(state['predicted']),
            baseline_matched_gt=int(state['gt_flags'].sum()),baseline_missed_gt=int((~state['gt_flags']).sum()),
            baseline_spurious_pred=int((~state['pred_flags']).sum()),baseline_matched_pairs=state['pairs'],
            teacher_diagnostic_matched_gt=int(teacher_state['gt_flags'].sum()),
            teacher_diagnostic_matched_short=sum(c['teacher_matched'] and c['frames']<=125 for c in cases),
            baseline_branches=branches,identity_controls_exact=True)
        protocols[protocol]=metrics;write_json(folder/'diagnostics.json',metrics)
        print('Diagnostic aggregate',protocol,json.dumps(dict(confidence=confidence,missed_gt=metrics['baseline_missed_gt'],
            spurious_pred=metrics['baseline_spurious_pred'],teacher_matched_short=metrics['teacher_diagnostic_matched_short'])),flush=True)
    decision=continuation(protocols,plan['continuation_gate'])
    trace_summary={protocol:dict(Counter(row['first_reason'] or 'accepted' for row in all_traces if row['protocol']==protocol and row['probe']=='baseline')) for protocol in plan['protocols']}
    write_json(output/'oracle_diagnostics.json',oracle_records);write_json(output/'gate_traces.json',all_traces)
    result=dict(round=28,result='accepted_teacher_opportunity' if decision['passed'] else 'rejected_teacher_opportunity',
        continuation=decision,protocols=protocols,baseline_gate_first_reasons=trace_summary,
        teacher_checkpoint_reloaded=True,actual_native_prefix_parity=True,selected_reference_identity_controls=identity_controls,
        actual_native_prefix_lengths=lengths,
        oracle_interventions=oracle_count,full_both_branch_trace_agreement=True,pristine_parent_arrays=True,
        new_fits=0,new_possession_score_rows=0,new_candidate_csvs=0,selected_unchanged=True)
    write_json(output/'summary.json',result)
    validate()
    for name,expected in fingerprints.items():
        if digest(ROOT/name)!=expected:raise ValueError('Selected input changed during audit')
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),source_sha256=json.loads((ROUND/'protocol_freeze/implementation.json').read_text()),
        input_sha256={**plan['input_sha256'],**fingerprints},
        cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},new_fits=0,new_score_rows=0))
    print(json.dumps(dict(result=result['result'],continuation=decision,identity_controls=identity_controls,oracle_interventions=oracle_count,new_fits=0),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROUND/'analysis_A')
    main(parser.parse_args().output.resolve())
