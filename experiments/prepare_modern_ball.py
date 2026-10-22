"""Frozen modern Core ML detector screen on local, excluded weak proxies."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_appearance import decode_local_frame, image_geometry, negative_positions
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from finalize_ball_appearance import audit_manifest
from modern_ball import letterbox_crop, modern_scores
from native_detector import detection_metrics, native_batches
from prepare_modern_models import verify_assets
from starter.load_homography import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['screen', 'confirmation'], default='screen')
    parser.add_argument('--resume', action='store_true', help='Only after the previous process is confirmed stopped')
    args = parser.parse_args()
    root = ROOT/'data/control_modern_ball_v1'
    plan_path = ROOT/'experiments/MODERN_BALL_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Modern frozen input changed')
    audit_manifest(ROOT/'data/control_ball_appearance_v1/feasibility')
    asset_path = root/'model_assets/asset_manifest.json'
    assets = json.loads(asset_path.read_text())
    verify_assets(plan, assets)
    if args.stage == 'screen':
        frames = np.load(ROOT/'data/control_ball_appearance_v1/feasibility/weak_ball_cache.npz')['frames'][160:]
        variants = [(model, side) for model in plan['models'] for side in plan['crop_sides']]
    else:
        previous = json.loads((root/'screen/manifest.json').read_text())
        if previous['plan_sha256'] != digest(plan_path):
            raise ValueError('Modern screen plan changed')
        selected = previous['selected_variant']
        if selected is None:
            raise ValueError('No modern screen variant passed; confirmation remains unused')
        frames = np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
        model = next(value for value in plan['models'] if value['name'] == selected['model'])
        variants = [(model, selected['side'])]
    if len(frames) != 60:
        raise ValueError('Expected60 frozen weak evaluation centers')
    output = root/args.stage
    output.mkdir(exist_ok=args.resume)
    saved_plan = output/'plan.json'
    if saved_plan.exists() and saved_plan.read_bytes() != plan_path.read_bytes():
        raise ValueError('Resumed modern plan differs')
    saved_plan.write_bytes(plan_path.read_bytes())
    names = ('experiments/prepare_modern_ball.py', 'experiments/modern_ball.py', 'experiments/prepare_modern_models.py',
             'experiments/native_detector.py', 'experiments/native_detector_predict.js', 'experiments/ball_appearance.py',
             'experiments/broadcast_alignment.py', 'starter/load_homography.py', 'games.py',
             'experiments/finalize_ball_appearance.py', 'experiments/build_temporal_candidate.py')
    sources = {name:digest(ROOT/name) for name in names}
    source_record = output/'source_at_dispatch.json'
    if source_record.exists() and json.loads(source_record.read_text()) != sources:
        raise ValueError('Resumed modern source differs')
    source_record.write_text(json.dumps(sources, indent=2)+'\n')
    np.save(output/'frames.npy', frames)
    crops = output/'crops'
    crops.mkdir(exist_ok=args.resume)
    source = load_source('tactical')
    matrices, _, valid = load()
    sides = sorted(set(side for _, side in variants))
    records = []
    for index, frame in enumerate(frames):
        frame = int(frame)
        if not valid[frame]:
            continue
        image = Image.fromarray(decode_local_frame(frame))
        points = [source['ball'][frame]]+negative_positions(source['ball'][frame], source['positions'][frame])
        for point_index, point in enumerate(points):
            geometry = image_geometry(matrices[frame], point)
            if geometry is None:
                continue
            x, y, diameter = geometry
            for side in sides:
                value = letterbox_crop(image, x, y, side, plan['input_width'], plan['input_height'])
                if value is None:
                    continue
                identifier = f'roi{side}_sample{index:03d}_point{point_index}'
                path = crops/(identifier+'.png')
                data = BytesIO()
                value.save(data, format='PNG')
                payload = data.getvalue()
                if path.exists() and path.read_bytes() != payload:
                    raise ValueError('Preserved modern crop differs')
                if not path.exists():
                    path.write_bytes(payload)
                records.append(dict(id=identifier, path=str(path.relative_to(ROOT)), frame=frame, sample=index,
                    point=point_index, side=side, label=int(point_index == 0), diameter=diameter, x=x, y=y))
        if (index+1) % 10 == 0:
            print('Local modern sample centers prepared', index+1, '/', len(frames), flush=True)
    record_path = output/'crop_metadata.json'
    metadata = json.dumps(records, indent=2, sort_keys=True)+'\n'
    if record_path.exists() and record_path.read_text() != metadata:
        raise ValueError('Resumed modern schedule differs')
    record_path.write_text(metadata)
    rows, selected = [], None
    for model, side in variants:
        subset = [value for value in records if value['side'] == side]
        images = [dict(id=value['id'], path=value['path']) for value in subset]
        variant = f"{model['name']}_roi{side}"
        print('Running local modern detector', variant, flush=True)
        folder = output/variant
        predictions = native_batches(ROOT/model['compiled_model'], images, folder, args.resume)
        near, anywhere = zip(*(modern_scores(value['objects'], side, record['diameter'], plan['input_width'], plan['input_height'])
                              for value, record in zip(predictions, subset)))
        target = np.array([value['label'] for value in subset])
        row = detection_metrics(near, target, len(frames))
        row.update(model=model['name'], side=side, variant=variant, crops=len(subset),
            unlocalized_positive_recall=float((np.array(anywhere)[target == 1] >= .25).mean()))
        np.savez_compressed(folder/'score_arrays.npz', target=target, near=np.array(near), any_score=np.array(anywhere))
        rows.append(row)
        if selected is None and row['feasible']:
            selected = dict(model=model['name'], side=side, variant=variant)
        pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Source changed during modern inference')
    deterministic = [output/'plan.json', output/'source_at_dispatch.json', output/'frames.npy', record_path, output/'scores.csv']
    deterministic += sorted(crops.glob('*.png'))
    deterministic += sorted(output.glob('*/predictions.json'))+sorted(output.glob('*/score_arrays.npz'))
    manifest = dict(plan_sha256=digest(plan_path), source_sha256=sources,
        input_sha256={**plan['input_sha256'], str(asset_path.relative_to(ROOT)):digest(asset_path)},
        model_archive_sha256={value['archive']:value['archive_sha256'] for value in plan['models']},
        cache_sha256={str(path.relative_to(output)):digest(path) for path in deterministic},
        selected_variant=selected, proxy_evaluations=len(rows), frames=60, stage=args.stage,
        no_model_fitting=True, public_possession_targets_not_read=True, no_media_uploaded_or_displayed=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(pd.DataFrame(rows).round(5).to_string(index=False), flush=True)
    print('Selected modern weak variant:', selected, flush=True)


if __name__ == '__main__':
    main()
