"""Audit all12 rejected local appearance proxies without public possession fitting."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_appearance import decode_local_frame, frame_features, weak_eligibility, weak_split
from ball_motion_appearance import motion_frame_features
from ball_pooled import pooled_features
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from games import GAME, half_video_blocks, require
from research_ball_pooled import weak_metrics
from starter.load_homography import load


def audit_manifest(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if digest(ROOT/'experiments'/name) != expected:
            raise ValueError('Ball appearance source changed: '+name)
    for name, expected in manifest['input_sha256'].items():
        if name.startswith('tactical/'):
            path = Path(require([name.split('/')[1]])[0])
        else:
            path = ROOT/name
        if digest(path) != expected:
            raise ValueError('Ball appearance input changed: '+name)
    for name, expected in manifest['cache_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError('Ball appearance cache changed: '+name)
    return manifest


def check_metrics(folder, scores, profile, x, y, vx, vy):
    for kind in ('logistic', 'forest'):
        name = f'profile{profile}_{kind}'
        model = joblib.load(folder/(name+'.joblib'))
        row, probability = weak_metrics(model, vx, vy, 60)
        np.testing.assert_array_equal(probability, joblib.load(folder/(name+'.joblib')).predict_proba(vx))
        expected = scores.set_index('model').loc[name]
        for key, value in row.items():
            if key == 'feasible':
                if bool(expected[key]) != value:
                    raise AssertionError('Proxy feasibility decision changed')
            elif abs(float(expected[key])-float(value)) > 1e-12:
                raise AssertionError('Saved weak proxy metric differs')
        if kind == 'logistic':
            scaler = model.named_steps['standardscaler']
            np.testing.assert_allclose(scaler.mean_, x.astype(float).mean(axis=0), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(scaler.var_, x.astype(float).var(axis=0), atol=1e-10, rtol=1e-10)
        if set(model.classes_) != {0, 1} or set(y) != {0, 1}:
            raise AssertionError('Ball proxy classes differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-history', action='store_true', help='Also verify the original session snapshot and protected candidate/ledger')
    args = parser.parse_args()
    root = ROOT/'data/control_ball_appearance_v1'
    folders = {name:root/dirname for name, dirname in [('centered', 'feasibility'), ('pooled', 'pooled_feasibility'), ('motion', 'motion_feasibility')]}
    manifests = {name:audit_manifest(folder) for name, folder in folders.items()}
    plans = dict(centered='BALL_APPEARANCE_PLAN.json', pooled='BALL_POOLED_PLAN.json', motion='BALL_MOTION_APPEARANCE_PLAN.json')
    for name, filename in plans.items():
        path = ROOT/'experiments'/filename
        if manifests[name]['plan_sha256'] != digest(path) or (folders[name]/'plan.json').read_bytes() != path.read_bytes():
            raise ValueError('Frozen appearance plan changed')
        if manifests[name]['selected_model'] is not None:
            raise ValueError('This closeout expects all appearance proxies rejected')
    source = load_source('tactical')
    matrices, error, valid = load()
    frozen = ROOT/'data/continuation_20261021/annotation_freeze/windows.json'
    if digest(frozen) != '5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c':
        raise ValueError('Reserved metadata changed')
    windows = json.loads(frozen.read_text())['windows']
    train, validation, _, _ = weak_split(weak_eligibility(source, error, valid), half_video_blocks(GAME), windows)
    original = np.load(folders['centered']/'weak_ball_cache.npz')
    np.testing.assert_array_equal(original['frames'], np.r_[train, validation])
    if not original['done'].all():
        raise AssertionError('Incomplete original appearance extraction')
    frames = original['frames']
    neighbors = (frames[:, None]+np.array([-5, 0, 5])).ravel()
    if ((neighbors >= 10935) & (neighbors <= 18934)).any():
        raise AssertionError('Temporal pixels touch the excluded public buffer')
    for window in windows:
        if ((neighbors >= window['start']-250) & (neighbors <= window['end']+250)).any():
            raise AssertionError('Temporal pixels touch reserved buffers')
    if not valid[neighbors].all() or not np.isfinite(matrices[neighbors]).all() or (np.linalg.det(matrices[neighbors]) == 0).any():
        raise AssertionError('Unsupported sampled temporal homography')
    motion = np.load(folders['motion']/'motion_ball_cache.npz')
    np.testing.assert_array_equal(motion['frames'], frames)
    if not motion['done'].all():
        raise AssertionError('Incomplete temporal appearance extraction')
    scores = {name:pd.read_csv(folder/'scores.csv') for name, folder in folders.items()}
    training = np.arange(len(frames))[:, None] < 160
    for profile in range(2):
        values, target = original['features'][profile], original['target']
        observed = np.isfinite(values).all(axis=2) & (target >= 0)
        fit, test = observed & training, observed & ~training
        x, y, vx, vy = values[fit], target[fit], values[test], target[test]
        check_metrics(folders['centered'], scores['centered'], profile, x, y, vx, vy)
        regenerated, regenerated_v = pooled_features(x), pooled_features(vx)
        saved = np.load(folders['pooled']/f'profile{profile}_features.npz')
        np.testing.assert_array_equal(regenerated, saved['fitting'])
        np.testing.assert_array_equal(regenerated_v, saved['validation'])
        np.testing.assert_array_equal(y, saved['fitting_target'])
        np.testing.assert_array_equal(vy, saved['validation_target'])
        check_metrics(folders['pooled'], scores['pooled'], profile, regenerated, y, regenerated_v, vy)
        values, target = motion['features'][profile], motion['target']
        observed = np.isfinite(values).all(axis=2) & (target >= 0)
        fit, test = observed & training, observed & ~training
        check_metrics(folders['motion'], scores['motion'], profile, values[fit], target[fit], values[test], target[test])
    for index in (0, 159, 219):
        frame = int(frames[index])
        images = [decode_local_frame(frame+delta) for delta in (-5, 0, 5)]
        values, target = frame_features(images[1], matrices[frame], source['ball'][frame], source['positions'][frame], [1., 2.], index < 160)
        np.testing.assert_array_equal(values, original['features'][:, index])
        np.testing.assert_array_equal(target, original['target'][index])
        values, target = motion_frame_features(images, matrices[[frame-5, frame, frame+5]], source['ball'][frame], source['positions'][frame], [1., 2.], index < 160)
        np.testing.assert_array_equal(values, motion['features'][:, index])
        np.testing.assert_array_equal(target, motion['target'][index])
    inventory = []
    for family, table in scores.items():
        if len(table) != 4 or table.feasible.any():
            raise AssertionError('Appearance inventory differs or has an unhandled finalist')
        for row in table.to_dict('records'):
            row['family'] = family
            row['physical_profile'] = float(row['profile'])+(0 if family == 'centered' else 1)
            inventory.append(row)
    pd.DataFrame(inventory).to_csv(root/'proxy_inventory.csv', index=False)
    if args.audit_history:
        baseline = json.loads((root/'baseline_manifest.json').read_text())
        for name, expected in baseline.items():
            if name == 'experiments/SELECTED_MODEL.md':
                continue
            if digest(ROOT/name) != expected:
                raise AssertionError('Protected selected artifact or1291-row ledger changed')
        selected = ROOT/'data/candidate_motion_dead_trim'
        candidate_manifest = json.loads((selected/'manifest.json').read_text())
        for group in ('model_sha256', 'probability_sha256'):
            for name, expected in candidate_manifest[group].items():
                if digest(selected/name) != expected or digest(root/'baseline_candidate'/name) != expected:
                    raise AssertionError('Protected selected checkpoint or probability changed')
        with tarfile.open(root/'experiments_before.tar.gz') as archive:
            for name in ('experiments/EXPERIMENT_LEDGER.csv', 'experiments/SELECTED_MODEL.md'):
                if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != baseline[name]:
                    raise AssertionError('Protected source archive lost its entry or ledger')
            for group in ('source_sha256', 'reference_source_sha256'):
                for name, expected in candidate_manifest[group].items():
                    current = ROOT/name
                    if not current.is_file():
                        current = ROOT/'experiments'/name
                    path = str(current.relative_to(ROOT))
                    if path.startswith('experiments/'):
                        content = archive.extractfile(path).read()
                    else:
                        content = subprocess.check_output(['git', 'show', '3de6b60aeabd2475d95879a651d38b72bc4fba00:'+path], cwd=ROOT)
                    if hashlib.sha256(content).hexdigest() != expected or digest(current) != expected:
                        raise AssertionError('Protected selected source differs from its archive or fixed Git reference')
    result = dict(proxy_evaluations=12, original_models=4, pooled_models=4, temporal_models=4,
        all_proxy_metrics_regenerated=True, all_models_reload_exactly=True, fitted_scalers_training_only=True,
        pooled_features_regenerated_exactly=True, centered_and_temporal_sample_recomputations_each=3,
        weak_train_frames=160, weak_validation_frames=60, sample_centers=220, distinct_temporal_frames=len(np.unique(neighbors)),
        public_and_reserved_pixels_excluded=True, all_temporal_homographies_invertible=True,
        actual_input_source_cache_hashes_verified=True, ledger_rows=len(pd.read_csv(ROOT/'experiments/EXPERIMENT_LEDGER.csv')),
        historical_snapshot_audited=args.audit_history, ledger_unchanged_exactly=True if args.audit_history else None,
        selected_candidate_unchanged=True if args.audit_history else None,
        public_possession_models_fitted=0, public_possession_score_rows_added=0,
        fresh_weak_confirmation_run=False, selected_appearance_model=None,
        private_accuracy_verified=False, maximum_score_verified=False)
    (root/'integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
