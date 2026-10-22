"""Local three-frame weak moving-ball appearance; no public possession fitting."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_appearance import decode_local_frame
from ball_motion_appearance import MOTION_FEATURES, motion_frame_features
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from research_ball_pooled import weak_metrics
from starter.load_homography import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = ROOT/'data/control_ball_appearance_v1'
    output = root/'motion_feasibility'
    output.mkdir(exist_ok=args.resume)
    plan_path = ROOT/'experiments/BALL_MOTION_APPEARANCE_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['inputs'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen motion-appearance input changed')
    if (output/'plan.json').exists() and (output/'plan.json').read_bytes() != plan_path.read_bytes():
        raise ValueError('Motion appearance plan changed')
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    original = np.load(root/'feasibility/weak_ball_cache.npz')
    frames = original['frames']
    source = load_source('tactical')
    matrices, _, valid = load()
    values = np.full((2, len(frames), 11, MOTION_FEATURES), np.nan, np.float32)
    target = np.full((len(frames), 11), -1, dtype=int)
    done = np.zeros(len(frames), bool)
    checkpoint = output/'motion_ball_cache.npz'
    if checkpoint.exists():
        saved = np.load(checkpoint)
        np.testing.assert_array_equal(saved['frames'], frames)
        values, target, done = saved['features'], saved['target'], saved['done']
    def save():
        temporary = output/'cache_in_progress.npz'
        np.savez_compressed(temporary, frames=frames, features=values, target=target, done=done)
        temporary.replace(checkpoint)
    def extract(index):
        frame = int(frames[index])
        neighbors = np.array([frame-5, frame, frame+5])
        if (neighbors < 0).any() or (neighbors >= len(matrices)).any() or not valid[neighbors].all():
            return index, np.full_like(values[:, index], np.nan), np.full(11, -1, int)
        images = [decode_local_frame(f) for f in neighbors]
        features, labels = motion_frame_features(images, matrices[neighbors], source['ball'][frame], source['positions'][frame], [1., 2.], index < 160)
        return index, features, labels
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(extract, i) for i in np.flatnonzero(~done)]
        for future in as_completed(futures):
            index, features, labels = future.result()
            values[:, index], target[index], done[index] = features, labels, True
            if done.sum() % 20 == 0 or done.all():
                save()
                print('Local compensated ball frames processed', int(done.sum()), '/', len(frames), flush=True)
    rows, selected = [], None
    training = np.arange(len(frames))[:, None] < 160
    for profile in range(2):
        observed = np.isfinite(values[profile]).all(axis=2) & (target >= 0)
        fit, test = observed & training, observed & ~training
        x, y, vx, vy = values[profile][fit], target[fit], values[profile][test], target[test]
        for kind in ('logistic', 'forest'):
            name = f'profile{profile}_{kind}'
            model = (make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced', max_iter=1000, random_state=42))
                if kind == 'logistic' else RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                    max_features='sqrt', class_weight='balanced_subsample', n_jobs=1, random_state=42))
            model.fit(x, y)
            row, probability = weak_metrics(model, vx, vy, 60)
            joblib.dump(model, output/(name+'.joblib'))
            if not np.array_equal(probability, joblib.load(output/(name+'.joblib')).predict_proba(vx)):
                raise AssertionError('Motion ball checkpoint reload differs')
            row.update(model=name, profile=profile, fitting_crops=len(y), features=MOTION_FEATURES)
            if selected is None and row['feasible']:
                selected = name
            rows.append(row)
    pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
    names = ('prepare_ball_motion_appearance.py', 'ball_motion_appearance.py', 'ball_pooled.py', 'ball_appearance.py', 'research_ball_pooled.py')
    manifest = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256={name:digest(ROOT/name) for name in plan['inputs']}, cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        selected_model=selected, all_models_reload_exactly=True, public_possession_targets_not_read=True,
        original_weak_validation_reused_adaptively=True, fresh_confirmation_required=True, source_context_radius=5,
        no_media_displayed_or_uploaded=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(pd.DataFrame(rows).round(5).to_string(index=False), flush=True)
    print('Selected compensated weak ball model:', selected, flush=True)


if __name__ == '__main__':
    main()
