"""Frozen target-blind coverage trial; no images, action outputs or possession labels."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from broadcast_alignment import frame_usable
from broadcast_action_sampling import schedule
from alignment_coverage import (inverse_schedule,query_plan,reference_query_plan,
                                synthetic_feature_view,coverage_diagnostics,opportunity_gate)
from games import GAME,half_video_blocks


def validate_inputs(plan_path):
    plan=json.loads(plan_path.read_text())
    root=ROOT/'data/control_alignment_coverage_v1'
    if plan_path.read_bytes()!=(root/'protocol_freeze/plan.json').read_bytes():
        raise ValueError('Frozen coverage protocol changed')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen coverage input changed: '+name)
    # Reuse independently recomputed mappings, verifying the source that built them.
    prefix_root=ROOT/'data/control_broadcast_action_v1/prefix_replay'
    record=json.loads((prefix_root/'summary.json').read_text())
    if not record['independently_recomputed_mappings'] or record['actual_source_lengths']!=[18935,19060]:
        raise ValueError('Independent prefix reconstruction not established')
    for name,expected in record['source_sha256'].items():
        if digest(ROOT/'experiments'/name)!=expected:raise ValueError('Inherited prefix reconstruction source changed')
    for n in (18935,19060):
        name=f'mapping_{n}.npz'
        if digest(prefix_root/name)!=record['cache_sha256'][name]:raise ValueError('Independent prefix mapping changed')
    for name,expected in json.loads((root/'protocol_freeze/protected_artifacts.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Protected selected/reference artifact changed')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/control_alignment_coverage_v1/assessment')
    args=parser.parse_args()
    plan_path=ROOT/'experiments/ALIGNMENT_COVERAGE_PLAN.json'
    plan=validate_inputs(plan_path)
    output=args.output.resolve();output.mkdir(parents=True)
    source_names=('prepare_alignment_coverage.py','alignment_coverage.py','broadcast_alignment.py',
                  'broadcast_action_sampling.py','broadcast_action_features.py','sports_action_sampling.py',
                  'sports_action_features.py','provided_speed.py','../games.py')
    sources={n:digest(ROOT/'experiments'/n) for n in source_names}
    (output/'source_at_dispatch.json').write_text(json.dumps(sources,indent=2)+'\n')
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    # Only these observation fields are needed; no ball or speed arrays consumed.
    with np.load(ROOT/'data/control_broadcast_v1/inputs/broadcast_observations.npz') as data:
        broadcast={name:data[name] for name in ('positions','teams','valid')}
    usable=frame_usable(broadcast)[:164487]
    del broadcast
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    blocks=half_video_blocks(GAME)
    variants=plan['variants']
    queries,features,rows_by_prefix,counts_by_prefix,mappings={},{},{},{},{}
    for n in (18935,19060):
        with np.load(ROOT/f'data/control_broadcast_action_v1/prefix_replay/mapping_{n}.npz') as data:
            mapping={name:data[name] for name in ('frames','orientation')}
        if len(mapping['frames'])!=n:raise ValueError('Source prefix length differs')
        mappings[n]=mapping
        for variant in variants:
            name=variant['name']
            if name=='original50':
                rows=schedule(n,mapping,blocks,reserved,held,broadcast_n=164487)
                q=reference_query_plan(rows,mapping,n,blocks);counts={}
                if n==19060 and rows!=json.loads((ROOT/'data/control_broadcast_action_v1/feasibility/schedule.json').read_text()):
                    raise ValueError('Retained baseline schedule differs')
            else:
                rows,counts=inverse_schedule(n,mapping,blocks,reserved,held,usable,variant['grid'])
                q=query_plan(rows,mapping,n)
            prefix=f'{name}_{n}'
            (output/(prefix+'_schedule.json')).write_text(json.dumps(rows,indent=2)+'\n')
            np.savez_compressed(output/(prefix+'_queries.npz'),**q)
            view=synthetic_feature_view(q,mapping,n,blocks)
            view.to_pickle(output/(prefix+'_synthetic_features.pkl'))
            queries[name,n],features[name,n]=q,view
            rows_by_prefix[name,n],counts_by_prefix[name,n]=rows,counts
    parity={}
    for variant in variants:
        name=variant['name']
        for key in queries[name,18935]:
            np.testing.assert_array_equal(queries[name,18935][key][11135:18735],queries[name,19060][key][11135:18735])
        pd.testing.assert_frame_equal(features[name,18935].iloc[11185:18685],features[name,19060].iloc[11185:18685],check_exact=True)
        if features[name,19060].shape[1]!=121:raise ValueError('Synthetic feature schema differs')
        parity[name]=True
    original=queries['original50',19060]
    recorded=json.loads((ROOT/'data/control_broadcast_action_v1/inputs/support.json').read_text())
    if int(original['seen'][11185:18685].sum())!=recorded['observed_public_frames']:
        raise ValueError('Original observed coverage differs from retained support')
    if np.any(queries['inverse50',19060]['seen'] & ~queries['inverse25',19060]['seen']):
        raise ValueError('Finer grid unexpectedly lost query coverage')
    diagnostics={v['name']:coverage_diagnostics(queries[v['name'],19060],original,
                 rows_by_prefix[v['name'],19060],counts_by_prefix[v['name'],19060]) for v in variants}
    gate=opportunity_gate(diagnostics['inverse25'],plan['source_support_gate'],all(parity.values()))
    # Validate the derived local tactical bounds; the broadcast formation search is global.
    for name in ('inverse50','inverse25'):
        for n in (18935,19060):
            for row in rows_by_prefix[name,n]:
                c=row['tactical_center']
                if row['decision_left']!=c-175 or row['decision_right']!=c+175:raise ValueError('Dependency footprint differs')
                if row['image_left']<c-125 or row['image_right']>c+125:raise ValueError('Image footprint differs')
            q=queries[name,n];f=np.flatnonzero(q['seen']);c=q['tactical_center'][f]
            if len(f) and np.abs(f-c).max()>25:raise ValueError('Query footprint differs')
    result=dict(round=23,plan_sha256=digest(plan_path),variants=diagnostics,opportunity_gate=gate,
        selected_metadata_variant='inverse25' if gate['passed'] else None,
        strict_mapping_coverage_public=int((mappings[19060]['frames'][11185:18685]>=0).sum()),
        inherited_independent_prefix_reconstruction_verified=True,query_parity_with_50frame_context=parity,
        full121_synthetic_feature_prefix_parity=parity,public_rows=7500,
        actual_new_action_features_built=False,image_content_radius_bound=200,
        tactical_observation_dependency_radius_bound=250,broadcast_formation_search_remains_global=True,
        no_images_decoded=True,no_action_outputs_consumed=True,no_possession_targets_read=True,
        new_neural_calls=0,new_fits=0,new_score_rows=0,selected_unchanged=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if sources!={n:digest(ROOT/'experiments'/n) for n in source_names}:raise ValueError('Coverage source changed during assessment')
    validate_inputs(plan_path)
    manifest=dict(plan_sha256=digest(plan_path),source_sha256=sources,input_sha256=plan['input_sha256'],
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
