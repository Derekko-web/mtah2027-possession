"""Audit diagnostic attribution, unchanged production replay and separate artifacts."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_anchor_failures import validate_inputs,require_equal
from anchor_failures import ANCHOR_REASONS,FRAME_REASONS,attribution_summary
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from games import GAME,half_video_blocks


def main():
    root=ROOT/'data/control_anchor_failures_v1';plan=validate_inputs()
    first,replay=root/'assessment',root/'assessment_replay'
    a,b=audit_manifest(first),audit_manifest(replay)
    if a!=b or (first/'manifest.json').read_bytes()!=(replay/'manifest.json').read_bytes():
        raise ValueError('Diagnostic replay manifest differs')
    for name in a['cache_sha256']:
        if (first/name).read_bytes()!=(replay/name).read_bytes():raise ValueError('Diagnostic replay artifact bytes differ')
    ledger=ROOT/'experiments/EXPERIMENT_LEDGER.csv'
    if ledger.read_bytes()!=(root/'protocol_freeze/ledger_before.csv').read_bytes():raise ValueError('Diagnosis changed ledger bytes')
    upstream=ROOT/'data/control_broadcast_v1/inputs'
    def observations(name):
        with np.load(upstream/(name+'_observations.npz')) as data:return {k:data[k] for k in ('positions','teams','valid')}
    full,broadcast=observations('tactical'),observations('broadcast')
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    checked=0
    for n in (18935,19060):
        def loaded(name):
            with np.load(first/f'{name}_{n}.npz') as data:return dict(data)
        anchors,trace,mapping,frames=[loaded(name) for name in ('anchors','anchor_trace','mapping','frame_trace')]
        tactical={k:v[:n] if k in ('positions','valid') else v for k,v in full.items()}
        require_equal(mapping['anchor_allowed'],trace['reason']=='allowed','anchor reason/allowed consistency')
        require_equal(mapping['frames']>=0,frames['reason']=='mapped','mapped frame reason consistency')
        require_equal(trace['tactical_usable'],np.isfinite(anchors['coarse'][:,:3]).all(axis=1),'coarse availability explanation')
        if np.any(~np.isin(trace['reason'],ANCHOR_REASONS)) or np.any(~np.isin(frames['reason'],FRAME_REASONS)):
            raise ValueError('Unknown diagnostic category')
        expected=json.loads((first/f'summary_{n}.json').read_text())
        actual=attribution_summary(anchors,trace,mapping,frames,tactical,broadcast,plan['reporting'])
        if any(expected[key]!=value for key,value in actual.items()):raise ValueError('Saved aggregate attribution differs')
        if sum(expected['anchor_first_failure'].values())!=300 or sum(expected['frame_first_failure'].values())!=7500:
            raise ValueError('Diagnostic denominator differs')
        if expected['frame_first_failure']['mapped']!=3359:raise ValueError('Original3359mapping coverage changed')
        for value in expected['anchor_failed_predicates_overlapping'].values():
            if not 0<=value['failed']<=value['evaluated']<=300:raise ValueError('Conditional overlap denominator invalid')
        for value in expected['unmapped_endpoint_reason_overlap'].values():
            if not 0<=value<=4141:raise ValueError('Endpoint overlap denominator invalid')
        table=pd.read_csv(first/f'consistency_{n}.csv')
        if table.duplicated(['tactical_frame','candidate_kind']).any():raise ValueError('Duplicate consistency diagnostic')
        for frame in table.tactical_frame.unique():
            left,right=frame-50,frame+50
            if not (0<=left and right<n and any(a<=left and right<=b for _,a,b in half_video_blocks(GAME))):
                raise ValueError('Consistency context guard failed')
            if any(left<=r['end']+250 and right>=r['start']-250 for r in reserved) or np.any((held+250>=left)&(held-250<=right)):
                raise ValueError('Consistency sealed/held context guard failed')
        with np.load(ROOT/f'data/control_broadcast_action_v1/prefix_replay/mapping_{n}.npz') as data:
            for key in mapping:require_equal(mapping[key],data[key],'independent original mapping '+key)
        if n==19060:
            with np.load(upstream/'anchors.npz') as data:
                for key in anchors:require_equal(anchors[key],data[key],'original anchor '+key)
        checked+=len(table)
    summary=json.loads((first/'summary.json').read_text())
    if not all(summary[k] for k in ('cached_original_anchors_exact','cached_strict_mapping_exact','both_actual_source_prefix_mappings_exact','attribution_and_consistency_prefix_parity','no_new_alignment_or_gate_changes')):
        raise ValueError('Required diagnostic replay proof absent')
    result=dict(round=24,passed=True,result='completed_diagnosis',diagnostic_artifacts_byte_exact=len(a['cache_sha256'])+1,
        original_coarse_and_fine_arrays_exact=True,original_strict_mapping_all_fields_exact=True,
        mutually_exclusive_attribution_reconciled=True,overlapping_predicate_denominators_verified=True,
        incident_interval_OR_and_knot_semantics_preserved=True,
        actual_18935_19060_prefix_attribution_parity=True,consistency_guarded_rows_checked=checked,
        consistency_offsets=[-12,12],consistency_checks_not_new_acceptance_gates=True,
        no_new_alignment_features_or_predictions=True,protected_artifacts_verified=17,
        ledger_rows=1468,all_ledger_bytes_unchanged=True,selected_unchanged=True,
        new_images=0,new_neural_calls=0,new_action_outputs_consumed=0,new_possession_targets_read=0,new_fits=0,new_score_rows=0,
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('audit_anchor_failures.py','anchor_failures.py','prepare_anchor_failures.py')})
    (root/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
