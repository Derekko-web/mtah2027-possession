"""Reparse modern detector evidence, verify exclusions and protected artifacts."""

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
from native_detector import canonical_reply, detection_metrics, native_batches
from prepare_modern_models import verify_assets
from starter.load_homography import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-history', action='store_true')
    parser.add_argument('--compiled-assets', type=Path, help='Independently compiled assets for reload checks')
    args = parser.parse_args()
    root = ROOT/'data/control_modern_ball_v1'
    folder = root/'screen'
    manifest = json.loads((folder/'manifest.json').read_text())
    plan_path = ROOT/'experiments/MODERN_BALL_PLAN.json'
    plan = json.loads(plan_path.read_text())
    if digest(plan_path) != manifest['plan_sha256'] or plan_path.read_bytes() != (folder/'plan.json').read_bytes():
        raise ValueError('Modern plan changed')
    for group in ('source_sha256', 'input_sha256', 'model_archive_sha256'):
        for name, expected in manifest[group].items():
            if digest(ROOT/name) != expected:
                raise ValueError('Modern source/input/model changed: '+name)
    for name, expected in manifest['cache_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError('Modern deterministic artifact changed: '+name)
    asset_root = args.compiled_assets.resolve() if args.compiled_assets else root/'model_assets'
    assets = json.loads((asset_root/'asset_manifest.json').read_text())
    verify_assets(plan, assets)
    audit_manifest(ROOT/'data/control_ball_appearance_v1/feasibility')
    original = np.load(ROOT/'data/control_ball_appearance_v1/feasibility/weak_ball_cache.npz')['frames']
    frames = np.load(folder/'frames.npy')
    np.testing.assert_array_equal(frames, original[160:])
    windows = json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    if ((frames >= 10935) & (frames <= 18934)).any():
        raise AssertionError('Modern screen touches public buffer')
    for window in windows:
        if ((frames >= window['start']-250) & (frames <= window['end']+250)).any():
            raise AssertionError('Modern screen touches reserved buffer')
    confirmation = np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    if len(confirmation) != 60 or (np.abs(confirmation[:, None]-original[None]) <= 250).any():
        raise AssertionError('Confirmation separation changed')
    source = load_source('tactical')
    matrices, _, valid = load()
    records = json.loads((folder/'crop_metadata.json').read_text())
    expected_records = []
    for index, frame in enumerate(frames):
        frame = int(frame)
        if not valid[frame]:
            continue
        points = [source['ball'][frame]]+negative_positions(source['ball'][frame], source['positions'][frame])
        for point_index, point in enumerate(points):
            geometry = image_geometry(matrices[frame], point)
            if geometry is None:
                continue
            x, y, diameter = geometry
            for side in plan['crop_sides']:
                half = side/2
                if x-half < 0 or y-half < 0 or x+half > 1920 or y+half > 1080:
                    continue
                identifier = f'roi{side}_sample{index:03d}_point{point_index}'
                expected_records.append(dict(id=identifier, path=str((folder/'crops'/(identifier+'.png')).relative_to(ROOT)),
                    frame=frame, sample=index, point=point_index, side=side, label=int(point_index == 0),
                    diameter=diameter, x=x, y=y))
    if records != expected_records:
        raise AssertionError('Modern geometry schedule differs')
    for index in (0, 29, 59):
        image = Image.fromarray(decode_local_frame(frames[index]))
        for record in records:
            if record['sample'] != index:
                continue
            value = letterbox_crop(image, record['x'], record['y'], record['side'], plan['input_width'], plan['input_height'])
            data = BytesIO()
            value.save(data, format='PNG')
            if data.getvalue() != (ROOT/record['path']).read_bytes():
                raise AssertionError('Fresh modern crop differs')
    scores = pd.read_csv(folder/'scores.csv')
    if scores.variant.tolist() != plan['selection_order'] or manifest['proxy_evaluations'] != len(scores):
        raise AssertionError('Modern inventory incomplete')
    passing = scores[scores.feasible]
    expected_selected = None if passing.empty else dict(model=passing.iloc[0].model, side=int(passing.iloc[0].side), variant=passing.iloc[0].variant)
    if manifest['selected_variant'] != expected_selected:
        raise AssertionError('Modern first-pass selection differs')
    if expected_selected is None and (root/'confirmation').exists():
        raise AssertionError('Failed modern screen consumed confirmation')
    replay_root = root/'audit_replays'
    replay_root.mkdir(exist_ok=True)
    run = 1
    while (replay_root/f'run{run:03d}').exists():
        run += 1
    replay = replay_root/f'run{run:03d}'
    replay.mkdir()
    for row in scores.itertuples():
        subset = [record for record in records if record['side'] == row.side]
        condition = folder/row.variant
        predictions = json.loads((condition/'predictions.json').read_text())
        if [p['id'] for p in predictions] != [r['id'] for r in subset]:
            raise AssertionError('Modern image identity differs')
        raw_rows = []
        for request_path in sorted((condition/'requests').glob('*.json')):
            request = json.loads(request_path.read_text())
            reply = json.loads((condition/'native_outputs'/request_path.name).read_text())
            raw_rows.extend(canonical_reply(reply, [item['id'] for item in request['images']]))
        if raw_rows != predictions:
            raise AssertionError('Modern canonical results differ from raw evidence')
        near, anywhere = zip(*(modern_scores(prediction['objects'], row.side, record['diameter'], plan['input_width'], plan['input_height'])
                              for prediction, record in zip(predictions, subset)))
        target = np.array([record['label'] for record in subset])
        arrays = np.load(condition/'score_arrays.npz')
        for name, values in [('target', target), ('near', np.array(near)), ('any_score', np.array(anywhere))]:
            np.testing.assert_array_equal(arrays[name], values)
        metrics = detection_metrics(near, target, 60)
        metrics['unlocalized_positive_recall'] = float((np.array(anywhere)[target == 1] >= .25).mean())
        metrics['crops'] = len(subset)
        for name, expected in metrics.items():
            actual = getattr(row, name)
            if abs(float(actual)-float(expected)) > 1e-12:
                raise AssertionError('Modern weak metric differs')
        indices = sorted(set([0, len(subset)-1, int(np.argmax(anywhere))]))
        images = [dict(id=subset[i]['id'], path=subset[i]['path']) for i in indices]
        result = native_batches(asset_root/(row.model+'.mlmodelc'), images, replay/row.variant)
        if result != [predictions[i] for i in indices]:
            raise AssertionError('Modern reload prediction differs')
    if args.audit_history:
        baseline = json.loads((root/'baseline_manifest.json').read_text())
        for name, expected in baseline.items():
            if name.startswith('experiments/') and name != 'experiments/EXPERIMENT_LEDGER.csv':
                continue
            if digest(ROOT/name) != expected:
                raise AssertionError('Protected selected artifact or ledger changed')
        for path in (root/'baseline_candidate').rglob('*'):
            if path.is_file() and digest(path) != digest(ROOT/'data/candidate_motion_dead_trim'/path.relative_to(root/'baseline_candidate')):
                raise AssertionError('Protected selected artifact differs from before copy')
    integrity = dict(proxy_evaluations=len(scores), selected_variant=expected_selected,
        all_raw_outputs_reparsed_exact=True, all_score_arrays_regenerated_exact=True,
        all_metrics_recomputed=True, sample_model_reload_exact=True,
        sample_crops_redecoded_exact=True, public_and_reserved_exclusions_pass=True,
        no_public_possession_fitting=True, no_possession_ledger_additions=True,
        confirmation_unused=not (root/'confirmation').exists(), history_audited=args.audit_history,
        independent_compilation_checked=args.compiled_assets is not None,
        asset_manifest_sha256=digest(asset_root/'asset_manifest.json'))
    filename = 'integrity_independent_compilation.json' if args.compiled_assets else 'integrity.json'
    (root/filename).write_text(json.dumps(integrity, indent=2)+'\n')
    print(json.dumps(integrity, indent=2))


if __name__ == '__main__':
    main()
