"""Audit private local observations without displaying detections or images."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_appearance import decode_local_frame, image_geometry
from build_temporal_candidate import digest
from detector_roi import crop_centers
from finalize_ball_appearance import audit_manifest as audit_appearance
from finalize_broadcast_closure import audit_manifest
from games import GAME, half_video_blocks
from modern_ball import letterbox_crop
from native_detector import canonical_reply, native_batches
from prepare_modern_models import verify_assets
from starter.load_homography import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-history', action='store_true')
    args = parser.parse_args()
    root = ROOT/'data/control_detector_features_v1'
    folder = root/'observations'
    plan_path = ROOT/'experiments/DETECTOR_FEATURES_PLAN.json'
    plan = json.loads(plan_path.read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['plan_sha256'] != digest(plan_path) or (folder/'plan.json').read_bytes() != plan_path.read_bytes():
        raise ValueError('Detector plan changed')
    for group in ('source_sha256', 'input_sha256'):
        for name, expected in manifest[group].items():
            if digest(ROOT/name) != expected:
                raise ValueError('Detector source/input changed')
    for name, expected in manifest['cache_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError('Private detector artifact changed')
    audit_appearance(ROOT/'data/control_ball_appearance_v1/feasibility')
    audit_manifest(ROOT/'data/control_broadcast_v1/inputs')
    verify_assets(plan, json.loads((ROOT/'data/control_modern_ball_v1/model_assets/asset_manifest.json').read_text()))
    source = dict(np.load(ROOT/'data/control_broadcast_v1/inputs/tactical_observations.npz'))
    blocks = [(start, end+1) for _, start, end in half_video_blocks(GAME)]
    centers, interpolated = crop_centers(source['ball'], blocks)
    matrices, error, valid = load()
    windows = json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held = np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    expected_records = []
    for frame in range(10935, len(source['ball']), 5):
        if not valid[frame] or not np.isfinite(error[frame]) or error[frame] > 1 or not np.isfinite(centers[frame]).all():
            continue
        if any(w['start']-250 <= frame <= w['end']+250 for w in windows) or (np.abs(held-frame) <= 250).any():
            continue
        geometry = image_geometry(matrices[frame], centers[frame])
        if geometry is None:
            continue
        x, y, diameter = geometry
        if x-128 < 0 or x+128 > 1920 or y-128 < 0 or y+128 > 1080:
            continue
        expected_records.append(dict(id=f'sample{frame:06d}', path=str((folder/'crops'/f'sample{frame:06d}.png').relative_to(ROOT)),
            frame=frame, x=x, y=y, diameter=diameter, interpolated=bool(interpolated[frame])))
    records = json.loads((folder/'crop_metadata.json').read_text())
    if records != expected_records or len(records) != manifest['crops']:
        raise AssertionError('Private crop schedule differs from frozen rules')
    indices = sorted(set((0, len(records)//2, len(records)-1)))
    for index in indices:
        record = records[index]
        image = Image.fromarray(decode_local_frame(record['frame']))
        crop = letterbox_crop(image, record['x'], record['y'], 256)
        stream = BytesIO()
        crop.save(stream, format='PNG')
        if stream.getvalue() != (ROOT/record['path']).read_bytes():
            raise AssertionError('Fresh private crop differs')
    replay_root = root/'observation_audit_replays'
    replay_root.mkdir(exist_ok=True)
    number = 1
    while (replay_root/f'run{number:03d}').exists():
        number += 1
    replay_root = replay_root/f'run{number:03d}'
    replay_root.mkdir()
    for model in plan['models']:
        condition = folder/model['name']
        predictions = json.loads((condition/'predictions.json').read_text())
        raw = []
        requested = []
        for request_path in sorted((condition/'requests').glob('*.json')):
            request = json.loads(request_path.read_text())
            if request['compiled_model'] != model['compiled_model']:
                raise AssertionError('Native model identity differs')
            requested.extend(request['images'])
            response = json.loads((condition/'native_outputs'/request_path.name).read_text())
            raw.extend(canonical_reply(response, [item['id'] for item in request['images']]))
        if requested != [dict(id=r['id'], path=r['path']) for r in records] or raw != predictions:
            raise AssertionError('Raw native observation replay differs')
        images = [dict(id=records[i]['id'], path=records[i]['path']) for i in indices]
        repeat = native_batches(ROOT/model['compiled_model'], images, replay_root/model['name'])
        if repeat != [predictions[i] for i in indices]:
            raise AssertionError('Native detector sample reload differs')
    if args.audit_history:
        baseline = json.loads((root/'baseline_manifest.json').read_text())
        for name, expected in baseline.items():
            if name.startswith('experiments/'):
                continue
            if digest(ROOT/name) != expected:
                raise AssertionError('Protected selected8/reference artifact changed')
    result = dict(crops=len(records), model_image_inferences=2*len(records),
        all_raw_results_reparsed_exactly=True, roi_schedule_reconstructed_exactly=True,
        sampled_fresh_crops_exact=True, both_model_sample_reload_exact=True,
        reserved_and_held_confirmation_buffers_verified=True, no_possession_targets_read=True,
        history_audited=args.audit_history, auditor_sha256=digest(ROOT/'experiments/finalize_detector_observations.py'))
    (root/'observation_integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
