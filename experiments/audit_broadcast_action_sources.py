"""Verify frozen privacy/source/buffer contracts and save only aggregate diagnostics."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from broadcast_action_sampling import schedule,source_centers
from sports_action_sampling import clip_frames
from games import GAME,half_video_blocks
from prepare_broadcast_action_features import construct
from broadcast_action_features import support_summary


def main():
    root=ROOT/'data/control_broadcast_action_v1'
    plan_path=ROOT/'experiments/BROADCAST_ACTION_PLAN.json'
    if plan_path.read_bytes()!=(root/'protocol_freeze/plan.json').read_bytes():
        raise ValueError('Frozen broadcast protocol changed')
    plan=json.loads(plan_path.read_text())
    obs=audit_manifest(root/'observations');cache=audit_manifest(root/'inputs')
    for manifest in (obs,cache):
        if manifest['plan_sha256']!=digest(plan_path):raise ValueError('Stage plan identity differs')
    mapping=dict(np.load(ROOT/'data/control_broadcast_v1/inputs/strict_mapping.npz'))
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    video=json.loads((root/'observations/video_identity.json').read_text())
    rows=schedule(19060,mapping,half_video_blocks(GAME),reserved,held,broadcast_n=int(video['metadata']['nb_frames']))
    if rows!=json.loads((root/'feasibility/schedule.json').read_text()):raise ValueError('Frozen schedule differs')
    records=json.loads((root/'observations/progress.json').read_text())
    if sorted(r['center'] for r in records)!=source_centers(rows):raise ValueError('Source inference identities differ')
    checks=[]
    for p in sorted((root/'observations').glob('decoder_checks_*.json')):checks.extend(json.loads(p.read_text()))
    if len(checks)!=3 or not all(r['exact'] for r in checks):raise ValueError('Accurate-seek pixel replay differs')
    dense=pd.read_pickle(root/'inputs/dense_actions.pkl')
    features=pd.read_pickle(root/'inputs/events_only_inputs.pkl')
    rebuilt_dense,rebuilt,rebuilt_sources=construct(root/'observations',records,mapping,19060,half_video_blocks(GAME),reserved,held)
    pd.testing.assert_frame_equal(dense,rebuilt_dense,check_exact=True)
    pd.testing.assert_frame_equal(features,rebuilt,check_exact=True)
    rebuilt_support=support_summary(rebuilt_dense,rebuilt_sources,plan['support'])
    if rebuilt_support!=json.loads((root/'inputs/support.json').read_text()):raise ValueError('Frozen support replay differs')
    # The feasibility stage predates score append: audit its ledger identity against
    # the immutable pre-round snapshot, never rewrite the current ledger to match it.
    feasibility=json.loads((root/'feasibility/manifest.json').read_text())
    for name,expected in feasibility['input_sha256'].items():
        path=root/'protocol_freeze/ledger_before.csv' if name=='experiments/EXPERIMENT_LEDGER.csv' else ROOT/name
        if digest(path)!=expected:raise ValueError('Historical feasibility input changed')
    unseen=dense.sa_seen.to_numpy()==0
    if not features.loc[unseen].drop(columns='sa_seen').isna().all().all():raise ValueError('Missing alignment gained summary data')
    if len(features.columns)!=121 or features.columns.duplicated().any():raise ValueError('Feature schema differs')
    # All image-bearing values for f use a center <=25 away; summaries add <=50.
    # The validated source-frame coverage of each clip is inside center +/-150.
    centers=np.array([r['tactical_center'] for r in rows]);frames=np.flatnonzero(~unseen)
    nearest=np.min(np.abs(frames[:,None]-centers),axis=1)
    if nearest.max()>25:raise ValueError('Closest-center footprint differs')
    max_footprint=150+int(nearest.max())+50
    if max_footprint>250:raise ValueError('Image feature footprint exceeds training buffer')
    for name,expected in json.loads((root/'protocol_freeze/protected_artifacts.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Protected selected/reference artifact changed')
    before=(root/'protocol_freeze/ledger_before.csv').read_bytes()
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith(before):raise ValueError('Preceding ledger bytes changed')
    result=dict(plan_sha256=digest(plan_path),safe_tactical_associations=len(rows),unique_broadcast_clips=len(records),
        source_parities=dict(odd=sum(r['center']%2 for r in records),even=sum(r['center']%2==0 for r in records)),
        independent_accurate_seek_pixel_checks=len(checks),source_bounds_and_mapping_guard_passed=True,
        reserved_windows=len(reserved),held_weak_images=len(held),exclusion_radius=250,
        maximum_image_feature_reach=max_footprint,target_buffer=250,missing_rows_and_summaries_preserved=True,feature_replay_exact=True,support_replay_exact=True,historical_feasibility_inputs_verified=True,
        protected_artifact_hashes_verified=17,preceding_ledger_bytes_preserved=True,
        video_frames=int(video['metadata']['nb_frames']),broadcast_tracking_frames=164488,
        source_input_cache_hashes_verified=True,targets_not_read_by_this_audit=True,
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('audit_broadcast_action_sources.py','broadcast_action_sampling.py','broadcast_action_features.py')})
    (root/'source_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2))


if __name__=='__main__':main()
