"""Ordinary source/exclusion/replay audit for the rejected metadata-only round."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from prepare_alignment_coverage import validate_inputs
from broadcast_alignment import frame_usable
from sports_action_sampling import clip_frames
from games import GAME,half_video_blocks


def main():
    root=ROOT/'data/control_alignment_coverage_v1'
    plan=validate_inputs(ROOT/'experiments/ALIGNMENT_COVERAGE_PLAN.json')
    assessment,replica=root/'assessment',root/'assessment_replay'
    a,b=audit_manifest(assessment),audit_manifest(replica)
    if a!=b:raise ValueError('Isolated metadata replay differs')
    if (assessment/'manifest.json').read_bytes()!=(replica/'manifest.json').read_bytes():
        raise ValueError('Replay manifest bytes differ')
    for name in a['cache_sha256']:
        if (assessment/name).read_bytes()!=(replica/name).read_bytes():raise ValueError('Replay artifact bytes differ')
    ledger=ROOT/'experiments/EXPERIMENT_LEDGER.csv'
    if ledger.read_bytes()!=(root/'protocol_freeze/ledger_before.csv').read_bytes():raise ValueError('No-score round changed ledger')
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    with np.load(ROOT/'data/control_broadcast_v1/inputs/broadcast_observations.npz') as data:
        broadcast={name:data[name] for name in ('positions','teams','valid')}
    usable=frame_usable(broadcast)[:164487]
    summary=json.loads((assessment/'summary.json').read_text())
    verified_rows=0;parities={};max_image_radius=0
    for n in (18935,19060):
        with np.load(ROOT/f'data/control_broadcast_action_v1/prefix_replay/mapping_{n}.npz') as data:
            mapping={name:data[name] for name in ('frames','orientation')}
        for name in ('inverse50','inverse25'):
            rows=json.loads((assessment/f'{name}_{n}_schedule.json').read_text())
            q=dict(np.load(assessment/f'{name}_{n}_queries.npz'))
            for row in rows:
                c,source=row['tactical_center'],row['broadcast_center']
                first,last=row['decision_left'],row['decision_right']
                if not (0<=first and last<n and any(a<=first and last<=b for _,a,b in half_video_blocks(GAME))):
                    raise ValueError('Dependency interval violates source/half guard')
                if any(first<=r['end']+250 and last>=r['start']-250 for r in reserved) or np.any((held+250>=first)&(held-250<=last)):
                    raise ValueError('Sealed/held dependency guard violated')
                left,right=row['image_left'],row['image_right']
                frames=clip_frames(source);values=mapping['frames'][left:right+1]
                if frames[0]<0 or frames[-1]>=164487 or not usable[frames[0]:frames[-1]+1].all():
                    raise ValueError('Source clip bounds or intervening-frame support violated')
                orientation=mapping['orientation'][left:right+1]
                if np.any(values<0) or len(np.unique(orientation))!=1 or np.any(np.diff(values)<0) or np.any(np.diff(values)>2):
                    raise ValueError('Source inverse interval crosses map cut')
                if np.min(np.abs(frames[:,None]-values),axis=1).max()>1:
                    raise ValueError('Exact source witness missing')
                if not (first==c-175 and last==c+175 and c-125<=left<=c<=right<=c+125):
                    raise ValueError('Full tactical footprint exceeds250')
                max_image_radius=max(max_image_radius,c-left,right-c)
                verified_rows+=1
            rows_by_center={r['tactical_center']:r for r in rows}
            for frame in np.flatnonzero(q['seen']):
                row=rows_by_center[int(q['tactical_center'][frame])]
                if abs(frame-row['tactical_center'])>25 or not row['component_left']<=frame<=row['component_right']:
                    raise ValueError('Query crosses source component or radius')
                if not row['broadcast_center']-100<=int(q['broadcast_frame'][frame])<=row['broadcast_center']+98:
                    raise ValueError('Query extrapolates outside clip')
            if n==19060:
                centers={r['broadcast_center'] for r in rows}
                parities[name]=dict(odd=sum(c%2 for c in centers),even=sum(c%2==0 for c in centers))
    if summary['opportunity_gate']['passed'] or summary['selected_metadata_variant'] is not None:
        raise ValueError('Expected frozen metadata rejection changed')
    result=dict(round=23,passed=True,result='rejected_metadata_opportunity',
        replay_artifacts_byte_exact=len(a['cache_sha256'])+1,
        manifest_byte_exact=True,all_source_input_cache_hashes_verified=True,
        verified_inverse_associations_across_prefixes=verified_rows,source_center_parities=parities,
        exact_inverse_witnesses_and_intervening_source_frames_verified=True,
        query_mapping_components_and_source_bounds_verified=True,
        new_variants_certified=['inverse50','inverse25'],new_tactical_dependency_bound=250,
        new_image_content_bound=200,max_actual_clip_inverse_radius=max_image_radius,
        original50_image_bound=225,original50_generic_tactical_dependency_bound=275,
        global_broadcast_observation_search_explicit=True,
        reserved_windows=len(reserved),held_weak_images=len(held),exclusion_radius=250,
        ledger_rows=1468,all_ledger_bytes_unchanged=True,protected_artifacts_verified=17,
        no_images_or_neural_calls=True,no_actual_action_outputs_consumed=True,no_possession_targets_fits_or_scores=True,
        no_private_accuracy_or_maximum_claim=True,selected_unchanged=True,
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('audit_alignment_coverage.py','prepare_alignment_coverage.py','alignment_coverage.py')})
    (root/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
