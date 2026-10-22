"""Frozen local pretrained-detector weak screen; never fits possession targets."""

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
from native_detector import ball_scores, detection_metrics, native_batches, square_crop, verify_native_assets
from starter.load_homography import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['screen', 'confirmation'], default='screen')
    parser.add_argument('--resume', action='store_true', help='Resume only after the previous process is confirmed stopped')
    args = parser.parse_args()
    root = ROOT/'data/control_native_ball_v1'
    plan_path = ROOT/'experiments/NATIVE_BALL_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Native detector frozen input changed')
    audit_manifest(ROOT/'data/control_ball_appearance_v1/feasibility')
    asset_path = root/'model_assets/asset_manifest.json'
    assets = json.loads(asset_path.read_text())
    verify_native_assets(plan, assets)
    if args.stage == 'screen':
        frames = np.load(ROOT/'data/control_ball_appearance_v1/feasibility/weak_ball_cache.npz')['frames'][160:]
        variants = [(model, side) for model in plan['models'] for side in plan['crop_sides']]
    else:
        previous = json.loads((root/'screen/manifest.json').read_text())
        selected = previous['selected_variant']
        if selected is None:
            raise ValueError('No native screen method passed; confirmation must remain unused')
        frames = np.load(root/'confirmation_frames.npy')
        model = next(value for value in plan['models'] if value['name'] == selected['model'])
        variants = [(model, selected['side'])]
    if len(frames) != 60:
        raise ValueError('Expected60 frozen weak evaluation centers')
    output = root/args.stage
    output.mkdir(exist_ok=args.resume)
    saved_plan = output/'plan.json'
    if saved_plan.exists() and saved_plan.read_bytes() != plan_path.read_bytes():
        raise ValueError('Resumed native plan differs')
    saved_plan.write_bytes(plan_path.read_bytes())
    names = ('experiments/prepare_native_ball.py', 'experiments/native_detector.py', 'experiments/native_detector_predict.js',
             'experiments/ball_appearance.py', 'experiments/broadcast_alignment.py', 'starter/load_homography.py', 'games.py',
             'experiments/finalize_ball_appearance.py', 'experiments/build_temporal_candidate.py')
    sources = {name:digest(ROOT/name) for name in names}
    source_record = output/'source_at_dispatch.json'
    if source_record.exists() and json.loads(source_record.read_text()) != sources:
        raise ValueError('Resumed native source differs')
    source_record.write_text(json.dumps(sources, indent=2)+'\n')
    np.save(output/'frames.npy', frames)
    crops = output/'crops'
    crops.mkdir(exist_ok=args.resume)
    source = load_source('tactical')
    matrices, _, valid = load()
    sides = sorted(set(side for _, side in variants))
    records = []
    print('Preparing local native crops;no possession labels', flush=True)
    for index, frame in enumerate(frames):
        frame = int(frame)
        if not valid[frame]:
            continue
        image = Image.fromarray(decode_local_frame(frame))
        ball = source['ball'][frame]
        points = [ball]+negative_positions(ball, source['positions'][frame])
        for point_index, point in enumerate(points):
            geometry = image_geometry(matrices[frame], point)
            if geometry is None:
                continue
            x, y, diameter = geometry
            for side in sides:
                value = square_crop(image, x, y, side)
                if value is None:
                    continue
                identifier = f'roi{side}_sample{index:03d}_point{point_index}'
                path = crops/(identifier+'.png')
                data = BytesIO()
                value.save(data, format='PNG')
                payload = data.getvalue()
                if path.exists() and path.read_bytes() != payload:
                    raise ValueError('Preserved native crop differs')
                if not path.exists():
                    path.write_bytes(payload)
                records.append(dict(id=identifier, path=str(path.relative_to(ROOT)), frame=frame, sample=index,
                    point=point_index, side=side, label=int(point_index == 0), diameter=diameter, x=x, y=y))
        if (index+1) % 10 == 0:
            print('Local native sample centers prepared', index+1, '/', len(frames), flush=True)
    metadata = json.dumps(records, indent=2, sort_keys=True)+'\n'
    record_path = output/'crop_metadata.json'
    if record_path.exists() and record_path.read_text() != metadata:
        raise ValueError('Resumed native crop schedule differs')
    record_path.write_text(metadata)
    rows, selected = [], None
    for model, side in variants:
        subset = [value for value in records if value['side'] == side]
        images = [dict(id=value['id'], path=value['path']) for value in subset]
        variant = f"{model['name']}_roi{side}"
        print('Running local native detector', variant, flush=True)
        folder = output/variant
        predictions = native_batches(ROOT/model['compiled_model'], images, folder, args.resume)
        near, any_score = [], []
        for value, prediction in zip(subset, predictions):
            a, b = ball_scores(prediction['objects'], side, value['diameter'])
            near.append(a)
            any_score.append(b)
        target = np.array([value['label'] for value in subset])
        row = detection_metrics(near, target, len(frames))
        row.update(model=model['name'], side=side, variant=variant, crops=len(subset),
            unlocalized_positive_recall=float((np.array(any_score)[target == 1] >= .25).mean()))
        np.savez_compressed(folder/'score_arrays.npz', target=target, near=np.array(near), any_score=np.array(any_score))
        rows.append(row)
        if selected is None and row['feasible']:
            selected = dict(model=model['name'], side=side, variant=variant)
        pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Source changed during native inference')
    deterministic = [output/'plan.json', output/'source_at_dispatch.json', output/'frames.npy', record_path, output/'scores.csv']
    deterministic += sorted(crops.glob('*.png'))
    deterministic += sorted(output.glob('*/predictions.json'))+sorted(output.glob('*/score_arrays.npz'))
    result = dict(plan_sha256=digest(plan_path), source_sha256=sources,
        input_sha256={**plan['input_sha256'], str(asset_path.relative_to(ROOT)):digest(asset_path)},
        model_sha256={value['source_model']:value['sha256'] for value in plan['models']},
        compiled_sha256={str(Path(model['compiled_model'])/name):value for model in plan['models']
            for name, value in next(row for row in assets['models'] if row['model'] == model['name'])['compiled_sha256'].items()},
        cache_sha256={str(path.relative_to(output)):digest(path) for path in deterministic},
        selected_variant=selected, proxy_evaluations=len(rows), frames=60, no_model_fitting=True,
        public_possession_targets_not_read=True, no_media_uploaded_or_displayed=True, stage=args.stage,
        deterministic_predictions_exclude_runtime_timings=True)
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(pd.DataFrame(rows).round(5).to_string(index=False), flush=True)
    print('Selected native weak variant:', selected, flush=True)


if __name__ == '__main__':
    main()
