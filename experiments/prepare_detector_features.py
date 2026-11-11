"""Build frozen detector feature views and exact bounded-prefix comparisons."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from detector_features import sample_features, feature_views
from detector_roi import crop_centers
from ball_appearance import image_geometry
from finalize_broadcast_closure import audit_manifest
from games import GAME, half_video_blocks
from starter.load_homography import load


def main():
    root = ROOT/'data/control_detector_features_v1'
    observations = root/'observations'
    manifest = json.loads((observations/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Observation producer changed')
    for name, expected in manifest['cache_sha256'].items():
        if digest(observations/name) != expected:
            raise ValueError('Observation artifact changed')
    plan_path = ROOT/'experiments/DETECTOR_FEATURES_PLAN.json'
    if manifest['plan_sha256'] != digest(plan_path):
        raise ValueError('Detector observation plan changed')
    upstream = ROOT/'data/control_broadcast_v1/inputs'
    audit_manifest(upstream)
    source = dict(np.load(upstream/'tactical_observations.npz'))
    matrices, error, _ = load()
    records = json.loads((observations/'crop_metadata.json').read_text())
    predictions = [json.loads((observations/name/'predictions.json').read_text()) for name in ('yolo11s', 'yolo11m')]
    if any([p['id'] for p in values] != [r['id'] for r in records] for values in predictions):
        raise ValueError('Detector observation identity mismatch')
    blocks = [(start, end+1) for _, start, end in half_video_blocks(GAME)]
    rows = [sample_features(r, a, b, source['positions'][r['frame']], source['teams'], matrices[r['frame']], error[r['frame']])
            for r, a, b in zip(records, *predictions)]
    samples = pd.DataFrame(rows, index=[r['frame'] for r in records])
    views, dense = feature_views(samples, len(source['ball']), blocks)
    short_views, _ = feature_views(samples, 18935, blocks)
    centers, interpolation = crop_centers(source['ball'][:18935], blocks)
    for r in records:
        if r['frame'] > 18815:
            continue
        geometry = image_geometry(matrices[r['frame']], centers[r['frame']])
        if not np.array_equal(geometry, [r['x'], r['y'], r['diameter']]):
            raise AssertionError('Crop-center prefix geometry differs')
        assert bool(interpolation[r['frame']]) == r['interpolated']
    for name in views:
        try:
            pd.testing.assert_frame_equal(views[name].iloc[11185:18685], short_views[name].iloc[11185:18685], check_exact=True)
        except AssertionError:
            raise AssertionError('Detector feature padded-prefix parity failed') from None
    public_samples = samples.loc[(samples.index >= 11185) & (samples.index <= 18684)]
    counts = dict(public_crop_samples=len(public_samples), public_consensus_samples=int((public_samples.vd_pair_seen == 1).sum()),
                  feature_columns={name:len(view.columns) for name, view in views.items()})
    support = counts['public_crop_samples'] >= 250 and counts['public_consensus_samples'] >= 100
    output = root/'inputs'
    output.mkdir()
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    samples.to_pickle(output/'sample_features.pkl')
    dense.to_pickle(output/'dense_features.pkl')
    for name, view in views.items():
        view.to_pickle(output/(name+'_inputs.pkl'))
    names = ('prepare_detector_features.py', 'detector_features.py', 'detector_roi.py', 'provided_speed.py')
    result = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256={str((observations/'manifest.json').relative_to(ROOT)):digest(observations/'manifest.json'),
                      str((upstream/'manifest.json').relative_to(ROOT)):digest(upstream/'manifest.json')},
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        actual_padded_prefix_parity=True, crop_center_prefix_parity=True, targets_not_read=True,
        support_passed=bool(support), diagnostics=counts)
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(support_passed=bool(support), **counts), indent=2))


if __name__ == '__main__':
    main()
