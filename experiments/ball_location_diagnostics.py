"""Local-only ball-coordinate diagnostics; aggregate output, no model selection."""

from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
from fractions import Fraction

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_appearance import decode_local_frame
from broadcast_alignment import load_source, PITCH
from build_temporal_candidate import digest
from games import require
from modern_ball import unletterbox_objects
from native_detector import image_box
from starter.load_homography import load, project


def detection_boxes(prediction, record):
    result = []
    for obj in unletterbox_objects(prediction['objects']):
        confidence = max((v['confidence'] for v in obj['labels'] if v['label'] == 'sports ball'), default=0.)
        if confidence >= .25:
            box = image_box(obj['box'], record['side'], (record['x'], record['y']))
            result.append((confidence, box))
    return sorted(result, key=lambda pair:(-pair[0], tuple(pair[1])))


def lag_diagnostics(frames, detected, matrices, ball, grid, radii):
    expected = np.full((len(frames), len(grid), 2), np.nan)
    for i, frame in enumerate(frames):
        for j, shift in enumerate(grid):
            index = int(frame+shift)
            if index < 0 or index >= len(ball) or not np.isfinite(ball[index]).all():
                continue
            expected[i, j] = project(matrices[frame], *(ball[index]/PITCH))
    complete = np.isfinite(detected).all(axis=1) & np.isfinite(expected).all(axis=(1, 2))
    distance = np.linalg.norm(expected-detected[:, None], axis=2)
    medians, coverage = [], []
    for j in range(len(grid)):
        values = distance[complete, j]
        medians.append(float(np.median(values)) if len(values) else None)
        coverage.append(float((values <= radii[complete]).mean()) if len(values) else None)
    best = [] if not complete.any() else [int(grid[j]) for j, value in enumerate(medians) if value == min(medians)]
    return dict(complete_cases=int(complete.sum()), sufficient_for_lag_interpretation=bool(complete.sum() >= 24),
                grid=grid, median_pixel_residual=medians, location_radius_coverage=coverage,
                best_diagnostic_lags=best, no_alignment_change_applied=True), distance, complete


def context_diagnostics(a, b, maximum_distance=8.):
    both, consistent, distances = 0, 0, []
    for first, second in zip(a, b):
        if not len(first) or not len(second):
            continue
        both += 1
        distance = np.linalg.norm(first[:, None]-second[None], axis=2)
        ab, ba = np.argmin(distance, axis=1), np.argmin(distance, axis=0)
        pairs = [distance[i, j] for i, j in enumerate(ab) if ba[j] == i and distance[i, j] <= maximum_distance]
        if pairs:
            consistent += 1
            distances.extend(pairs)
    return dict(both_detected_centers=both, consistent_centers=consistent,
                consistent_fraction=consistent/both if both else None,
                median_matched_center_distance=float(np.median(distances)) if distances else None)


def decode_exact_pts(frame, video, timebase, start_pts):
    target = Fraction(int(frame), 25)/timebase+start_pts
    if target.denominator != 1:
        raise ValueError('Frame timestamp is not exactly representable')
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-copyts',
               '-ss', f'{max(0., int(frame)/25-2):.8f}', '-i', video,
               '-vf', f'select=eq(pts\\,{int(target)})', '-filter_threads', '1',
               '-frames:v', '1', '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24',
               '-f', 'rawvideo', 'pipe:1']
    result = subprocess.run(command, capture_output=True)
    if result.returncode or len(result.stdout) != 1920*1080*3:
        raise ValueError('Exact original-timestamp decode failed')
    return np.frombuffer(result.stdout, np.uint8).reshape(1080, 1920, 3)


def main():
    root = ROOT/'data/control_ball_location_v1'
    output = root/'diagnostics'
    output.mkdir()
    plan_path = ROOT/'experiments/BALL_LOCATION_DIAGNOSTIC_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Location diagnostic input changed')
    modern = ROOT/'data/control_modern_ball_v1/screen'
    manifest = json.loads((modern/'manifest.json').read_text())
    for name, expected in manifest['cache_sha256'].items():
        if digest(modern/name) != expected:
            raise ValueError('Existing modern predictions or crops changed')
    for name, expected in manifest['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Existing modern producer source changed')
    frames = np.load(modern/'frames.npy')
    windows = json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    if ((frames >= 10935) & (frames <= 18934)).any():
        raise AssertionError('Location diagnostic public buffer violation')
    for window in windows:
        if ((frames >= window['start']-250) & (frames <= window['end']+250)).any():
            raise AssertionError('Location diagnostic reserved buffer violation')
    source = load_source('tactical')
    matrices, _, _ = load()
    records = json.loads((modern/'crop_metadata.json').read_text())
    results, private, all_centers = [], {}, {}
    for path in sorted(modern.glob('*/predictions.json')):
        variant = path.parent.name
        side = int(variant.split('roi')[1])
        subset = [record for record in records if record['side'] == side]
        detected = np.full((len(frames), 2), np.nan)
        boxes = np.full((len(frames), 4), np.nan)
        radii = np.full(len(frames), np.nan)
        centers = [np.empty((0, 2)) for _ in frames]
        expected_point = np.full((len(frames), 2), np.nan)
        for record, prediction in zip(subset, json.loads(path.read_text())):
            if not record['label']:
                continue
            index = record['sample']
            expected_point[index] = [record['x'], record['y']]
            radii[index] = max(4., min(12., 2*record['diameter']))
            values = detection_boxes(prediction, record)
            if values:
                boxes[index] = values[0][1]
                detected[index] = boxes[index, :2]+boxes[index, 2:]/2
                centers[index] = np.array([box[:2]+box[2:]/2 for _, box in values])
        lag, distance, complete = lag_diagnostics(frames, detected, matrices, source['ball'], plan['lag_grid'], radii)
        conventions = {}
        present = np.isfinite(boxes).all(axis=1)
        for name, coefficient in [('center', .5), ('bottom_center', 1.), ('top_center', 0.)]:
            point = boxes[:, :2]+boxes[:, 2:]*[.5, coefficient]
            residual = np.linalg.norm(point-expected_point, axis=1)
            conventions[name] = dict(count=int(present.sum()),
                median_pixel_residual=float(np.median(residual[present])) if present.any() else None,
                radius_coverage=float((residual[present] <= radii[present]).mean()) if present.any() else None)
        results.append(dict(variant=variant, lag=lag, conventions=conventions))
        private[variant+'_boxes'] = boxes
        private[variant+'_lag_distances'] = distance
        private[variant+'_complete_cases'] = complete
        all_centers[variant] = centers
    context = []
    for a, b in combinations(sorted(all_centers), 2):
        model_a, side_a = a.split('_roi')
        model_b, side_b = b.split('_roi')
        if model_a != model_b and side_a != side_b:
            continue
        context.append(dict(first=a, second=b, **context_diagnostics(all_centers[a], all_centers[b])))
    video = require(['video'])[0]
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,r_frame_rate,avg_frame_rate,time_base,start_time,start_pts,nb_frames',
        '-of', 'json', video], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)['streams'][0]
    if (stream['width'], stream['height']) != (1920, 1080) or Fraction(stream['avg_frame_rate']) != 25 or Fraction(stream['r_frame_rate']) != 25 or float(stream['start_time']) != 0:
        raise ValueError('Video contract differs from exact-index control')
    import hashlib
    decode_controls = []
    for index in (0, 29, 59):
        first = decode_local_frame(frames[index])
        second = decode_exact_pts(frames[index], video, Fraction(stream['time_base']), int(stream['start_pts']))
        equal = np.array_equal(first, second)
        decode_controls.append(dict(sample_index=index, same_rgb_bytes=equal,
            ordinary_sha256=hashlib.sha256(first.tobytes()).hexdigest(),
            exact_pts_sha256=hashlib.sha256(second.tobytes()).hexdigest()))
        if not equal:
            raise AssertionError('Original decode does not select requested frame')
    np.savez_compressed(output/'private_diagnostics.npz', **private)
    result = dict(diagnostic_only=True, no_candidates_or_alignment_modified=True,
        plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/name) for name in (
            'experiments/ball_location_diagnostics.py', 'experiments/modern_ball.py',
            'experiments/native_detector.py', 'experiments/ball_appearance.py',
            'experiments/broadcast_alignment.py', 'starter/load_homography.py')},
        input_sha256=plan['input_sha256'], variants=results, context_comparisons=context,
        video_contract=stream, decode_controls=decode_controls,
        public_and_reserved_exclusions_pass=True, no_fresh_confirmation_images=True,
        no_new_possession_scores=True, not_corrected_ball_ground_truth=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(dict(plan_sha256=digest(plan_path),
        output_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}), indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
