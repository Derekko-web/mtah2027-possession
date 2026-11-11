"""Audit actual local probe schedule, weak reference eligibility and preservation."""
import json
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from games import GAME, half_video_blocks
from local_ownership import allowed_frame, parse_ownership


def main():
    root = ROOT/'data/control_local_ownership_v1'
    plan = ROOT/'experiments/LOCAL_OWNERSHIP_MONTAGE_PLAN.json'
    inputs = audit_manifest(root/'inputs_montage')
    inference = audit_manifest(root/'inference_montage')
    settings = json.loads(plan.read_text())
    if inputs['plan_sha256'] != digest(plan) or inference['plan_sha256'] != digest(plan):
        raise AssertionError('Ownership probe plan differs')
    reserved = json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held = np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    samples = json.loads((root/'inputs_montage/samples.json').read_text())
    centers = [row['center'] for row in samples]
    if len(set(centers)) != len(centers):
        raise AssertionError('Duplicate ownership samples')
    for row in samples:
        center = row['center']
        frames = [r['frame'] for r in row['source']]
        if center not in range(11310, 18685, 250) or frames != [center-12, center, center+12]:
            raise AssertionError('Ownership schedule changed')
        if not any(start <= frames[0] and frames[-1] <= end for _, start, end in half_video_blocks(GAME)):
            raise AssertionError('Ownership sequence crosses half')
        if not all(allowed_frame(frame, reserved, held) for frame in frames):
            raise AssertionError('Reserved/held images used')
        if any(abs(r['ball_source_frame']-r['frame']) > 25 for r in row['source']):
            raise AssertionError('Ball center exceeds bounded source window')
    cache = np.load(ROOT/'data/control_jersey_v1/feasibility/weak_color_cache.npz')
    model = joblib.load(ROOT/'data/control_jersey_v1/feasibility/profile0_logistic.joblib')
    refs = json.loads((root/'inputs/references.json').read_text())
    for team in (0, 1):
        own = [r for r in refs if r['team'] == team]
        if len(own) != 3 or len({r['frame'] for r in own}) != 3:
            raise AssertionError('Reference team coverage differs')
        for row in own:
            index = np.flatnonzero(cache['frames'][:240] == row['frame'])
            if len(index) != 1 or not allowed_frame(row['frame'], reserved, held, exclude_public=True):
                raise AssertionError('Reference frame eligibility failed')
            i, slot = int(index[0]), row['slot']
            if not cache['weak_mask'][i, slot] or cache['teams'][slot] != team:
                raise AssertionError('Reference weak team differs')
            feature = cache['features'][0, i, slot]
            if not np.isfinite(feature).all() or model.predict_proba(feature[None])[0, team] < .99:
                raise AssertionError('Reference weak confidence fails')
            if row['bounds'][3]-row['bounds'][1] < 8:
                raise AssertionError('Reference crop support fails')
    answers = json.loads((root/'inference_montage/answers.json').read_text())
    if sorted(row['center'] for row in answers) != sorted(centers):
        raise AssertionError('Incomplete local inference coverage')
    for row in answers:
        path = root/'inference_montage'/f'answer_{row["center"]}.json'
        raw = json.loads(path.read_text())
        if raw.get('model') != 'qwen3-vl:4b-instruct' or not raw.get('done'):
            raise AssertionError('Incomplete/wrong local model response')
        if digest(path) != row['raw_answer_sha256'] or parse_ownership(raw.get('response', '')) != row['prediction']:
            raise AssertionError('Local answer replay differs')
    baseline = json.loads((root/'baseline_manifest.json').read_text())
    if not all(digest(ROOT/name) == expected for name, expected in baseline.items()):
        raise AssertionError('Selected candidate changed')
    replay = json.loads((root/'montage_input_parity.json').read_text())
    if not replay['all_inputs_byte_identical'] or not replay['public_possession_targets_absent']:
        raise AssertionError('Target-blind source replay differs')
    assets = json.loads((root/'local_model_assets_instruct.json').read_text())
    if not assets['all_local_blob_hashes_verified']:
        raise AssertionError('Local model assets unverified')
    ignored = subprocess.run(['git', 'check-ignore', '--quiet', str((root/'inputs_montage/samples.json').relative_to(ROOT))], cwd=ROOT).returncode == 0
    if not ignored:
        raise AssertionError('Private probe artifacts are not ignored')
    record = dict(samples=len(samples), references=len(refs), source_schedule_and_exclusions_verified=True,
        automatic_weak_reference_conditions_verified=True, local_model_and_answers_verified=True,
        source_montage_rebuild_byte_exact=True, source_replay_without_targets=True, selected_artifacts_unchanged=True,
        private_artifacts_ignored=True, new_possession_fits=0, new_possession_score_rows=0,
        source_sha256=digest(Path(__file__)))
    (root/'audit_montage.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
