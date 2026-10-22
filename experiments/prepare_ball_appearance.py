"""Train/check excluded weak local ball appearance; no possession targets or images shown."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_appearance import FEATURES, decode_local_frame, frame_features, weak_eligibility, weak_split
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from games import GAME, half_video_blocks, require
from starter.load_homography import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Resume a confirmed stopped extraction from its saved numeric cache')
    args = parser.parse_args()
    output = ROOT/'data/control_ball_appearance_v1/feasibility'
    output.mkdir(exist_ok=args.resume)
    plan_path = ROOT/'experiments/BALL_APPEARANCE_PLAN.json'
    plan = json.loads(plan_path.read_text())
    if (output/'plan.json').exists() and (output/'plan.json').read_bytes() != plan_path.read_bytes():
        raise ValueError('Frozen ball plan changed')
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    frozen = ROOT/'data/continuation_20261021/annotation_freeze/windows.json'
    if digest(frozen) != '5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c':
        raise ValueError('Reserved windows changed')
    source = load_source('tactical')
    matrices, error, valid = load()
    train, validation, candidates, available = weak_split(weak_eligibility(source, error, valid), half_video_blocks(GAME), json.loads(frozen.read_text())['windows'])
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,r_frame_rate,start_time,nb_frames', '-of', 'json', require(['video'])[0]], text=True))['streams'][0]
    if info != dict(width=1920, height=1080, r_frame_rate='25/1', start_time='0.000000', nb_frames='199319'):
        raise ValueError('Tactical video contract differs')
    print('Excluded eligible frames', candidates, 'training candidates after exclusion', available, flush=True)
    frames = np.r_[train, validation]
    profiles = plan['crops']['profiles']
    features = np.full((len(profiles), len(frames), 11, FEATURES), np.nan, np.float32)
    target = np.full((len(frames), 11), -1, dtype=int)
    done = np.zeros(len(frames), bool)
    checkpoint = output/'weak_ball_cache.npz'
    if checkpoint.exists():
        saved = np.load(checkpoint)
        np.testing.assert_array_equal(saved['frames'], frames)
        features, target, done = saved['features'], saved['target'], saved['done']
    def save():
        temporary = output/'cache_in_progress.npz'
        np.savez_compressed(temporary, features=features, target=target, done=done, frames=frames)
        temporary.replace(checkpoint)
    def extract(index):
        frame = int(frames[index])
        values, y = frame_features(decode_local_frame(frame), matrices[frame], source['ball'][frame], source['positions'][frame], profiles, index < len(train))
        return index, values, y
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(extract, i) for i in np.flatnonzero(~done)]
        for future in as_completed(futures):
            index, values, y = future.result()
            features[:, index], target[index], done[index] = values, y, True
            if done.sum() % 20 == 0 or done.all():
                save()
                print('Local ball frames processed', int(done.sum()), '/', len(frames), flush=True)
    rows, selected = [], None
    training_rows = np.arange(len(frames))[:, None] < len(train)
    for profile, ratio in enumerate(profiles):
        observed = np.isfinite(features[profile]).all(axis=2) & (target >= 0)
        fitting, testing = observed & training_rows, observed & ~training_rows
        x, y = features[profile][fitting], target[fitting]
        vx, vy = features[profile][testing], target[testing]
        if set(y) != {0, 1} or set(vy) != {0, 1}:
            raise ValueError('Weak ball fitting/validation need both classes')
        for kind in ('logistic', 'forest'):
            name = f'profile{profile}_{kind}'
            model = (make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced', max_iter=1000, random_state=42))
                if kind == 'logistic' else RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                    max_features='sqrt', class_weight='balanced_subsample', n_jobs=1, random_state=42))
            model.fit(x, y)
            probability = model.predict_proba(vx)
            joblib.dump(model, output/(name+'.joblib'))
            if not np.array_equal(probability, joblib.load(output/(name+'.joblib')).predict_proba(vx)):
                raise AssertionError('Ball model reload differs')
            confident = probability.max(axis=1) >= .8
            row = dict(model=name, profile=ratio, fitting_crops=len(y), validation_crops=len(vy),
                positive_validation=int((vy == 1).sum()), negative_validation=int((vy == 0).sum()),
                positive_valid_fraction=float((testing & (target == 1)).sum()/len(validation)),
                balanced_accuracy=float(balanced_accuracy_score(vy, probability.argmax(axis=1))),
                confident_coverage=float(confident.mean()), confident_agreement=float((probability[confident].argmax(axis=1) == vy[confident]).mean()) if confident.any() else 0.)
            row['feasible'] = bool(row['positive_validation'] >= 40 and row['negative_validation'] >= 120 and row['positive_valid_fraction'] >= .5 and
                row['balanced_accuracy'] >= .8 and row['confident_coverage'] >= .5 and row['confident_agreement'] >= .9)
            if selected is None and row['feasible']:
                selected = name
            rows.append(row)
    pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
    names = ('ball_appearance.py', 'prepare_ball_appearance.py', 'broadcast_alignment.py')
    result = dict(plan_sha256=digest(plan_path), selected_model=selected, source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256={f'tactical/{name}':digest(require([name])[0]) for name in ('atd_meta', 'atd_raw', 'homography', 'video')},
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        public_possession_targets_not_read=True, public_reserved_windows_excluded=True, fitting_validation_buffer=250,
        train_frames=len(train), validation_frames=len(validation), candidate_frames=candidates, annotation_metadata_sha256=digest(frozen),
        no_media_displayed_or_uploaded=True, all_models_reload_exactly=True, video_contract=info,
        weak_label_limit='Tracking-located ball proxy agreement,not human/corrected ball accuracy or possession accuracy')
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(pd.DataFrame(rows).round(5).to_string(index=False), flush=True)
    print('Selected weak ball appearance model:', selected, flush=True)


if __name__ == '__main__':
    main()
