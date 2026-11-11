"""Prepare frozen local detector observations without possession target access."""
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
from finalize_ball_appearance import audit_manifest
from games import GAME, half_video_blocks
from modern_ball import letterbox_crop
from native_detector import native_batches
from prepare_modern_models import verify_assets
from starter.load_homography import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = ROOT/'data/control_detector_features_v1'
    plan_path = ROOT/'experiments/DETECTOR_FEATURES_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen detector input changed')
    audit_manifest(ROOT/'data/control_ball_appearance_v1/feasibility')
    assets = json.loads((ROOT/'data/control_modern_ball_v1/model_assets/asset_manifest.json').read_text())
    verify_assets(plan, assets)
    output = root/'observations'
    output.mkdir(exist_ok=args.resume)
    names = ('experiments/prepare_detector_observations.py', 'experiments/detector_roi.py',
             'experiments/modern_ball.py', 'experiments/native_detector.py',
             'experiments/native_detector_predict.js', 'experiments/ball_appearance.py',
             'starter/load_homography.py', 'games.py')
    sources = {name:digest(ROOT/name) for name in names}
    source_path = output/'source_at_dispatch.json'
    if source_path.exists() and json.loads(source_path.read_text()) != sources:
        raise ValueError('Resumed detector source changed')
    source_path.write_text(json.dumps(sources, indent=2)+'\n')
    saved_plan = output/'plan.json'
    if saved_plan.exists() and saved_plan.read_bytes() != plan_path.read_bytes():
        raise ValueError('Resumed detector plan changed')
    saved_plan.write_bytes(plan_path.read_bytes())
    source = np.load(ROOT/'data/control_broadcast_v1/inputs/tactical_observations.npz')
    ball = source['ball']
    blocks = [(start, end+1) for _, start, end in half_video_blocks(GAME)]
    centers, interpolation = crop_centers(ball, blocks)
    matrices, error, valid = load()
    windows = json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held = np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    crops = output/'crops'
    crops.mkdir(exist_ok=args.resume)
    records = []
    for frame in range(10935, len(ball), 5):
        if not valid[frame] or error[frame] > 1 or not np.isfinite(centers[frame]).all():
            continue
        if any(w['start']-250 <= frame <= w['end']+250 for w in windows) or (np.abs(held-frame) <= 250).any():
            continue
        geometry = image_geometry(matrices[frame], centers[frame])
        if geometry is None:
            continue
        x, y, diameter = geometry
        if x-128 < 0 or x+128 > 1920 or y-128 < 0 or y+128 > 1080:
            continue
        path = crops/f'sample{frame:06d}.png'
        if not (args.resume and path.exists()):
            image = Image.fromarray(decode_local_frame(frame))
            crop = letterbox_crop(image, x, y, 256)
            stream = BytesIO()
            crop.save(stream, format='PNG')
            path.write_bytes(stream.getvalue())
        records.append(dict(id=f'sample{frame:06d}', path=str(path.relative_to(ROOT)), frame=frame,
                            x=x, y=y, diameter=diameter, interpolated=bool(interpolation[frame])))
        if len(records) % 100 == 0:
            print('Local detector crops prepared', len(records), flush=True)
    (output/'crop_metadata.json').write_text(json.dumps(records, indent=2)+'\n')
    images = [dict(id=r['id'], path=r['path']) for r in records]
    for model in plan['models']:
        print('Local detector inference', model['name'], len(images), flush=True)
        native_batches(ROOT/model['compiled_model'], images, output/model['name'], args.resume)
    if sources != {name:digest(ROOT/name) for name in names}:
        raise ValueError('Detector producer changed during inference')
    paths = [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.png', '.json')]
    (output/'manifest.json').write_text(json.dumps(dict(plan_sha256=digest(plan_path),
        source_sha256=sources, input_sha256=plan['input_sha256'],
        cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(paths)},
        crops=len(records), targets_not_read=True, all_processing_local=True), indent=2)+'\n')
    print('Local observations complete', len(records), flush=True)


if __name__ == '__main__':
    main()
