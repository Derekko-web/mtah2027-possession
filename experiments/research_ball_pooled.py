"""Four prospective pooled weak-appearance models, retaining the failed originals."""

import json
from pathlib import Path
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
from ball_pooled import pooled_features
from build_temporal_candidate import digest


def weak_metrics(model, x, y, validation_frames):
    probability = model.predict_proba(x)
    confident = probability.max(axis=1) >= .8
    row = dict(positive_validation=int((y == 1).sum()), negative_validation=int((y == 0).sum()),
        positive_valid_fraction=float((y == 1).sum()/validation_frames), balanced_accuracy=float(balanced_accuracy_score(y, probability.argmax(axis=1))),
        confident_coverage=float(confident.mean()), confident_agreement=float((probability[confident].argmax(axis=1) == y[confident]).mean()) if confident.any() else 0.)
    row['feasible'] = bool(row['positive_validation'] >= 40 and row['negative_validation'] >= 120 and row['positive_valid_fraction'] >= .5 and
        row['balanced_accuracy'] >= .8 and row['confident_coverage'] >= .5 and row['confident_agreement'] >= .9)
    return row, probability


def main():
    root = ROOT/'data/control_ball_appearance_v1'
    output = root/'pooled_feasibility'
    output.mkdir(exist_ok=False)
    plan_path = ROOT/'experiments/BALL_POOLED_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['inputs'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen weak inputs changed')
    original = json.loads((root/'feasibility/manifest.json').read_text())
    for name, expected in original['cache_sha256'].items():
        if digest(root/'feasibility'/name) != expected:
            raise ValueError('Original ball cache changed')
    for name, expected in original['source_sha256'].items():
        if digest(ROOT/'experiments'/name) != expected:
            raise ValueError('Original ball source changed')
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    saved = np.load(root/'feasibility/weak_ball_cache.npz')
    features, target = saved['features'], saved['target']
    training = np.arange(len(target))[:, None] < 160
    rows, selected = [], None
    for profile in range(2):
        valid = np.isfinite(features[profile]).all(axis=2) & (target >= 0)
        fit, test = valid & training, valid & ~training
        print('Pooling weak patch profile', profile, flush=True)
        x, vx = pooled_features(features[profile][fit]), pooled_features(features[profile][test])
        y, vy = target[fit], target[test]
        np.savez_compressed(output/f'profile{profile}_features.npz', fitting=x, validation=vx, fitting_target=y, validation_target=vy)
        for kind in ('logistic', 'forest'):
            name = f'profile{profile}_{kind}'
            model = (make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced', max_iter=1000, random_state=42))
                if kind == 'logistic' else RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                    max_features='sqrt', class_weight='balanced_subsample', n_jobs=1, random_state=42))
            model.fit(x, y)
            row, probability = weak_metrics(model, vx, vy, 60)
            joblib.dump(model, output/(name+'.joblib'))
            if not np.array_equal(probability, joblib.load(output/(name+'.joblib')).predict_proba(vx)):
                raise AssertionError('Pooled ball checkpoint reload differs')
            row.update(model=name, profile=profile, fitting_crops=len(y), features=x.shape[1])
            if selected is None and row['feasible']:
                selected = name
            rows.append(row)
    pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
    names = ('research_ball_pooled.py', 'ball_pooled.py')
    result = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256={name:digest(ROOT/name) for name in plan['inputs']}, cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        selected_model=selected, original_weak_validation_reused_adaptively=True, fresh_confirmation_required=True,
        all_models_reload_exactly=True, no_possession_targets_read=True)
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(pd.DataFrame(rows).round(5).to_string(index=False), flush=True)
    print('Selected pooled weak screen model:', selected, flush=True)


if __name__ == '__main__':
    main()
