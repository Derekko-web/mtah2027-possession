"""Fuse cached detector points and same-frame player colors, without targets."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from color_contact import contact_scores
from detector_features import candidates, consensus_pairs, PITCH
from finalize_broadcast_closure import audit_manifest
from games import require
from starter.load_homography import load, unproject


def build_contacts(source, color, records, predictions, matrices, variants):
    lookup = {int(frame):i for i, frame in enumerate(color['frames'])}
    arrays = {variant:np.full((len(source['ball']), 2), np.nan) for variant in variants}
    metadata = np.eye(2)[source['teams']]
    used = 0
    for record, first, second in zip(records, *predictions):
        frame = record['frame']
        if frame >= len(source['ball']):
            continue
        if frame not in lookup:
            continue
        a, pa = candidates(first, record)
        b, pb = candidates(second, record)
        pairs = consensus_pairs(a, b)
        if not pairs or not (pa or pb):
            continue
        point = pairs[0]['point']
        if np.linalg.norm(np.array(pa+pb)-point, axis=1).min() > 2*record['diameter']:
            continue
        try:
            nominal = np.array(unproject(matrices[frame], *point))*PITCH
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(nominal).all() or (nominal < -5).any() or (nominal > PITCH+5).any():
            continue
        probability = color['probability'][lookup[frame]]
        xy = source['positions'][frame]
        arrays['detected_metadata'][frame] = contact_scores(nominal, xy, metadata)
        arrays['detected_color'][frame] = contact_scores(nominal, xy, probability)
        raw = source['ball'][frame]
        if np.isfinite(raw).all() and (raw >= -5).all() and (raw <= PITCH+5).all():
            arrays['tracked_color'][frame] = contact_scores(raw, xy, probability)
        used += 1
    return arrays, used


def main():
    root = ROOT/'data/control_color_contact_v1'
    plan_path = ROOT/'experiments/COLOR_CONTACT_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen fusion input changed')
    audit_manifest(ROOT/'data/control_jersey_v1/inputs')
    feasibility = json.loads((ROOT/'data/control_jersey_v1/feasibility/manifest.json').read_text())
    for name, expected in feasibility['source_sha256'].items():
        if digest(ROOT/'experiments'/name) != expected:
            raise ValueError('Weak color source changed')
    for identity, expected in feasibility['input_sha256'].items():
        source_name, asset = identity.split('/')
        if digest(require([asset], source=source_name)[0]) != expected:
            raise ValueError('Color source identity changed')
    for name, expected in feasibility['cache_sha256'].items():
        if digest(ROOT/'data/control_jersey_v1/feasibility'/name) != expected:
            raise ValueError('Weak color artifact changed')
    audit_manifest(ROOT/'data/control_broadcast_v1/inputs')
    observations = ROOT/'data/control_detector_features_v1/observations'
    manifest = json.loads((observations/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Detector source changed')
    for name, expected in manifest['cache_sha256'].items():
        if digest(observations/name) != expected:
            raise ValueError('Detector input changed')
    source = dict(np.load(ROOT/'data/control_broadcast_v1/inputs/tactical_observations.npz'))
    color = np.load(ROOT/'data/control_jersey_v1/inputs/sampled_team_probability.npz')
    records = json.loads((observations/'crop_metadata.json').read_text())
    predictions = [json.loads((observations/model/'predictions.json').read_text()) for model in ('yolo11s', 'yolo11m')]
    if any([p['id'] for p in values] != [r['id'] for r in records] for values in predictions):
        raise ValueError('Fusion detector identities differ')
    matrices, _, _ = load()
    arrays, used = build_contacts(source, color, records, predictions, matrices, plan['variants'])
    short_source = {k:(v[:18935] if k in ('positions','speeds','ball','valid') else v) for k,v in source.items()}
    short, _ = build_contacts(short_source, color, records, predictions, matrices, plan['variants'])
    for name in arrays:
        if not np.array_equal(arrays[name][11185:18685], short[name][11185:18685], equal_nan=True):
            raise AssertionError('Color-contact actual source prefix differs')
    public = arrays['detected_color'][11185:18685]
    finite = np.isfinite(public).all(axis=1)
    strong = finite & (np.nan_to_num(public, nan=0).max(axis=1) >= .5) & (np.abs(public[:, 0]-public[:, 1]) >= .2)
    winner = np.nan_to_num(public, nan=0).argmax(axis=1)
    counts = [int((strong & (winner == team)).sum()) for team in (0, 1)]
    support = sum(counts) >= 100 and min(counts) >= 10
    output = root/'inputs'
    output.mkdir()
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    for variant, values in arrays.items():
        np.save(output/(variant+'.npy'), values)
    names = ('prepare_color_contact.py', 'color_contact.py', 'detector_features.py')
    result = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256=plan['input_sha256'], cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        targets_not_read=True, no_new_media_or_fitting=True, actual_padded_prefix_parity=True, support_passed=bool(support),
        diagnostics=dict(optical_contact_samples=used, fused_strong_by_team=counts, fused_strong_total=sum(counts)))
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(support_passed=bool(support), **result['diagnostics']), indent=2))


if __name__ == '__main__':
    main()
