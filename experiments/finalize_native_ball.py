"""Audit the six rejected native-detector proxies and exact local replay."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_appearance import decode_local_frame, image_geometry, negative_positions, weak_eligibility
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from finalize_ball_appearance import audit_manifest
from games import GAME, half_video_blocks
from native_detector import ball_scores, canonical_reply, detection_metrics, native_batches, square_crop, verify_native_assets
from starter.load_homography import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-history', action='store_true')
    args = parser.parse_args()
    root = ROOT/'data/control_native_ball_v1'
    folder = root/'screen'
    manifest = json.loads((folder/'manifest.json').read_text())
    plan_path = ROOT/'experiments/NATIVE_BALL_PLAN.json'
    plan = json.loads(plan_path.read_text())
    if manifest['plan_sha256'] != digest(plan_path) or (folder/'plan.json').read_bytes() != plan_path.read_bytes():
        raise ValueError('Native detector plan changed')
    for group in ('source_sha256', 'input_sha256', 'model_sha256', 'compiled_sha256'):
        for name, expected in manifest[group].items():
            if digest(ROOT/name) != expected:
                raise ValueError('Native source/input/model changed: '+name)
    for name, expected in manifest['cache_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError('Native deterministic cache changed: '+name)
    assets = json.loads((root/'model_assets/asset_manifest.json').read_text())
    verify_native_assets(plan, assets)
    audit_manifest(ROOT/'data/control_ball_appearance_v1/feasibility')
    original_frames = np.load(ROOT/'data/control_ball_appearance_v1/feasibility/weak_ball_cache.npz')['frames']
    frames = np.load(folder/'frames.npy')
    np.testing.assert_array_equal(frames, original_frames[160:])
    frozen = ROOT/'data/continuation_20261021/annotation_freeze/windows.json'
    windows = json.loads(frozen.read_text())['windows']
    if digest(frozen) != '5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c':
        raise ValueError('Reserved metadata changed')
    if ((frames >= 10935) & (frames <= 18934)).any():
        raise AssertionError('Native screen touches excluded public frames')
    for window in windows:
        if ((frames >= window['start']-250) & (frames <= window['end']+250)).any():
            raise AssertionError('Native screen touches reserved frames')
    if (root/'confirmation').exists():
        raise AssertionError('All six proxies failed; confirmation must remain unused')
    confirmation = np.load(root/'confirmation_frames.npy')
    if len(confirmation) != 60 or (np.abs(confirmation[:, None]-original_frames[None]) <= 250).any():
        raise AssertionError('Fresh confirmation separation violated')
    source = load_source('tactical')
    matrices, error, valid = load()
    eligible = weak_eligibility(source, error, valid)
    inside = np.zeros(len(eligible), bool)
    for _, start, end in half_video_blocks(GAME):
        inside[start:end+1] = True
    eligible &= inside & (np.arange(len(eligible)) % 25 == 0)
    eligible[10935:18935] = False
    for window in windows:
        eligible[max(0, window['start']-250):min(len(eligible), window['end']+251)] = False
    for frame in original_frames:
        eligible[max(0, frame-250):min(len(eligible), frame+251)] = False
    candidates = np.flatnonzero(eligible)
    expected_confirmation = np.sort(np.random.default_rng(43).choice(candidates, 60, replace=False))
    np.testing.assert_array_equal(confirmation, expected_confirmation)
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
                    frame=frame, sample=index, point=point_index, side=side, label=int(point_index == 0), diameter=diameter, x=x, y=y))
    if records != expected_records:
        raise AssertionError('Native crop geometry/schedule changed')
    for index in (0, 29, 59):
        image = Image.fromarray(decode_local_frame(frames[index]))
        for record in records:
            if record['sample'] != index:
                continue
            value = square_crop(image, record['x'], record['y'], record['side'])
            data = BytesIO()
            value.save(data, format='PNG')
            if data.getvalue() != (ROOT/record['path']).read_bytes():
                raise AssertionError('Freshly decoded native crop differs')
    scores = pd.read_csv(folder/'scores.csv')
    if len(scores) != 6 or scores.variant.tolist() != plan['selection_order'] or scores.feasible.any() or manifest['selected_variant'] is not None:
        raise ValueError('Incomplete rejected native-proxy inventory')
    replay = root/'audit_replays'
    replay.mkdir(exist_ok=True)
    run_number = 1
    while (replay/f'run{run_number:03d}').exists():
        run_number += 1
    replay = replay/f'run{run_number:03d}'
    replay.mkdir()
    for row in scores.itertuples():
        model = next(value for value in plan['models'] if value['name'] == row.model)
        subset = [value for value in records if value['side'] == row.side]
        condition = folder/row.variant
        predictions = json.loads((condition/'predictions.json').read_text())
        if [p['id'] for p in predictions] != [r['id'] for r in subset]:
            raise AssertionError('Native prediction alignment changed')
        raw_rows = []
        for request_path in sorted((condition/'requests').glob('*.json')):
            request = json.loads(request_path.read_text())
            response = json.loads((condition/'native_outputs'/request_path.name).read_text())
            raw_rows.extend(canonical_reply(response, [value['id'] for value in request['images']]))
        if raw_rows != predictions:
            raise AssertionError('Canonical native outputs differ from original inference')
        near, any_score = zip(*(ball_scores(value['objects'], row.side, record['diameter']) for value, record in zip(predictions, subset)))
        target = np.array([value['label'] for value in subset])
        arrays = np.load(condition/'score_arrays.npz')
        for key, value in [('target', target), ('near', np.array(near)), ('any_score', np.array(any_score))]:
            np.testing.assert_array_equal(arrays[key], value)
        metrics = detection_metrics(near, target, 60)
        for key, value in metrics.items():
            expected = getattr(row, key)
            if key == 'feasible':
                assert not bool(expected) and not value
            elif abs(expected-value) > 1e-12:
                raise AssertionError('Native weak metric changed')
        samples = sorted(set([0, len(subset)-1, int(np.argmax(any_score))]))
        images = [dict(id=subset[i]['id'], path=subset[i]['path']) for i in samples]
        rerun = native_batches(ROOT/model['compiled_model'], images, replay/row.variant)
        if rerun != [predictions[i] for i in samples]:
            raise AssertionError('Native model reload prediction differs')
    if args.audit_history:
        baseline = json.loads((root/'baseline_manifest.json').read_text())
        for name, expected in baseline.items():
            if name != 'experiments/SELECTED_MODEL.md' and digest(ROOT/name) != expected:
                raise AssertionError('Protected selected candidate/ledger changed')
        selected = ROOT/'data/candidate_motion_dead_trim'
        selected_manifest = json.loads((selected/'manifest.json').read_text())
        for group in ('model_sha256', 'probability_sha256'):
            for name, expected in selected_manifest[group].items():
                if digest(selected/name) != expected or digest(root/'baseline_candidate'/name) != expected:
                    raise AssertionError('Selected8 model/probability artifact changed')
        with tarfile.open(root/'experiments_before.tar.gz') as archive:
            for name in ('experiments/SELECTED_MODEL.md', 'experiments/EXPERIMENT_LEDGER.csv'):
                if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != baseline[name]:
                    raise AssertionError('Protected initial source archive changed')
        before = root/'provenance_before/screen'
        old_plan = json.loads((before/'plan.json').read_text())
        if old_plan != {name:value for name, value in plan.items() if name != 'provenance_revision'}:
            raise AssertionError('Native scientific configuration changed during provenance repair')
        for path in before.glob('*/predictions.json'):
            if path.read_bytes() != (folder/path.relative_to(before)).read_bytes():
                raise AssertionError('Original native predictions changed')
        if (before/'scores.csv').read_bytes() != (folder/'scores.csv').read_bytes():
            raise AssertionError('Original native scores changed')
        parity = json.loads((root/'provenance_parity.json').read_text())
        if not parity['six_prediction_files_byte_identical'] or not parity['all_crops_byte_identical']:
            raise AssertionError('Native independent compilation has not reproduced results')
    result = dict(proxy_evaluations=6, public_models_fitted=0, new_possession_rows=0, no_pretrained_model_refitting=True,
        all_raw_native_results_reparsed_exactly=True, all_proxy_score_arrays_regenerated_exactly=True,
        all_model_reload_samples_exact=True,
        crop_geometry_schedule_verified=True, fresh_decode_sample_centers=3, source_input_model_cache_hashes_verified=True,
        deterministic_outputs_omit_timing=True, held_confirmation_not_decoded=True,
        confirmation_schedule_regenerated_exactly=True, original_science_gates_unchanged=True,
        historical_snapshot_audited=args.audit_history, selected_candidate_unchanged=True if args.audit_history else None,
        ledger_rows=len(pd.read_csv(ROOT/'experiments/EXPERIMENT_LEDGER.csv')), ledger_unchanged=True if args.audit_history else None,
        selected_native_variant=None, maximum_score_verified=False, private_accuracy_verified=False)
    (root/'integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
