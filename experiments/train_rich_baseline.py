"""Fixed richer-feature experiment with buffered out-of-fold predictions.

No validation early stopping or parameter search. Both geometry-only and
geometry+tags models use the same regularized tree settings and team-swap
augmentation. Cache contains inputs only, not targets.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from rich_features import MAX_CONTEXT_SECONDS, build_features, swap_teams
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing

PARAMS = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
              tree_method="hist", n_estimators=200, max_depth=3, learning_rate=.04,
              min_child_weight=20, reg_lambda=10, subsample=.8, colsample_bytree=.9,
              random_state=42, n_jobs=2)


def main():
    output = ROOT / "data/rich_baseline"
    output.mkdir(exist_ok=True)
    gt = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    labels = gt.label.to_numpy()
    public = np.flatnonzero(labels != "X")
    cache = output / "input_features.pkl"
    if cache.exists():
        features = pd.read_pickle(cache)
    else:
        print("Building label-free geometry and temporal features...", flush=True)
        features = build_features(int(public[-1]) + 1)
        features.to_pickle(cache)
    assert MAX_CONTEXT_SECONDS < 10
    groups = {
        "rich_geometry": [name for name in features if "tag_" not in name],
        "rich_with_tags": list(features.columns),
    }
    oof = {name: np.full((len(public), 3), np.nan) for name in groups}
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    per_fold = []
    for minute, train, validation in folds:
        print(f"Fold {minute}: {len(train)} train / {len(validation)} validation frames", flush=True)
        y = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
        for name, columns in groups.items():
            x_train = features.iloc[train][columns]
            x_flip = swap_teams(x_train)
            y_flip = np.where(y == 0, 1, np.where(y == 1, 0, 2))
            x_aug = pd.concat([x_train, x_flip], ignore_index=True)
            y_aug = np.r_[y, y_flip]
            weights = len(y_aug) / (3 * np.bincount(y_aug)[y_aug])
            model = XGBClassifier(**PARAMS)
            model.fit(x_aug, y_aug, sample_weight=weights)
            x_val = features.iloc[validation][columns]
            p = (model.predict_proba(x_val) +
                 model.predict_proba(swap_teams(x_val))[:, [1, 0, 2]]) / 2
            oof[name][validation - public[0]] = p
            model.save_model(output / f"{name}_fold_{minute}.ubj")
            pd.DataFrame(p, columns=CLASSES).assign(frame=validation).to_csv(
                output / f"{name}_fold_{minute}_probabilities.csv", index=False)
            for method, probability in [("raw", p), ("smoothed", probability_smoothing(p))]:
                prediction = CLASSES[decode(probability)]
                result = pm.evaluate(pm.label_codes(labels[validation]), pm.label_codes(prediction))
                per_fold.append(dict(model=name, validation_minute=minute, decoder=method,
                                     spellF1=result["spell_f1_mean"], macroF1=result["macro_f1"],
                                     possessions=result["n_pred_spells"]))
    pd.DataFrame(per_fold).to_csv(output / "fold_scores.csv", index=False)
    full_predictions = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
    rows = []
    for name, p in oof.items():
        if not np.isfinite(p).all():
            raise ValueError("Incomplete OOF probabilities")
        for i, label in enumerate(CLASSES):
            full_predictions[f"{name}_probability_{label}"] = p[:, i]
        for method, smoothing in [("raw", False), ("smoothed", True)]:
            # Decode each held-out block separately; no probability mixing across model boundaries.
            prediction = np.empty(len(public), dtype="<U1")
            for _, _, validation in folds:
                index = validation - public[0]
                block = probability_smoothing(p[index]) if smoothing else p[index]
                prediction[index] = CLASSES[decode(block)]
            key = name + "_" + method
            full_predictions[key] = prediction
            result = pm.evaluate(pm.label_codes(labels[public]), pm.label_codes(prediction), bootstrap_n=1000)
            rows.append(dict(model=key, spellF1=result["spell_f1_mean"], macroF1=result["macro_f1"],
                             possessions=result["n_pred_spells"],
                             lower=result["spell_bootstrap_low"], upper=result["spell_bootstrap_high"]))
    for variant in ["smooth50", "smooth125"]:
        prediction = pm.read_label_csv(ROOT / f"evaluation/results/baseline_{variant}.csv").label.to_numpy()[public]
        full_predictions["organiser_" + variant] = prediction
        result = pm.evaluate(pm.label_codes(labels[public]), pm.label_codes(prediction), bootstrap_n=1000)
        rows.append(dict(model="organiser_" + variant, spellF1=result["spell_f1_mean"],
                         macroF1=result["macro_f1"], possessions=result["n_pred_spells"],
                         lower=result["spell_bootstrap_low"], upper=result["spell_bootstrap_high"]))
    full_predictions.to_csv(output / "oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "oof_scores.csv", index=False)
    (output / "experiment.json").write_text(json.dumps(dict(
        params=PARAMS, feature_groups=groups, max_context_seconds=MAX_CONTEXT_SECONDS,
        training="5 buffered folds, class-balanced sample weights, team-swap augmentation",
        decoder="fixed penalty 3, raw or original probability smoothing",
        caveat="Adaptive same-match development on 17 public possessions; not independent full-match validation",
    ), indent=2) + "\n")
    print("\nPer-minute results:")
    print(pd.DataFrame(per_fold).pivot(index=["model", "decoder"], columns="validation_minute", values="spellF1").round(4))
    print("\nStitched out-of-fold results versus original public baselines:")
    print(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
