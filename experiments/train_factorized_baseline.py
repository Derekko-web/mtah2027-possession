"""Separate dead-ball detection from team control on fixed buffered folds.

Two predeclared alternatives: separate regularized binary trees, and a physical
nearest-player control estimate paired with the same dead-ball detector. Compact
features exclude tactical tags and long-window means that erase short possessions.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from rich_features import swap_teams
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing


def compact_columns(features):
    shared = ["ball_seen", "ball_speed", "ball_displacement_0.2s", "ball_displacement_1s",
              "ball_goal_line_distance", "ball_touchline_distance",
              "ball_speed__past_1s", "ball_speed__future_1s"]
    paired = ["distance_1", "distance_2", "distance_3", "nearest_speed", "speed_median",
              "moving_fraction", "within_1m", "within_3m", "within_5m",
              "control_share", "control_share__past_1s", "control_share__future_1s"]
    return shared + [prefix + name for prefix in ["h__", "a__"] for name in paired]


def binary_model():
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                         max_depth=2, n_estimators=120, learning_rate=.04, reg_lambda=10,
                         min_child_weight=15, subsample=.8, colsample_bytree=.9,
                         random_state=42, n_jobs=2)


def compose(p_dead, p_home):
    return np.column_stack(((1-p_dead)*p_home, (1-p_dead)*(1-p_home), p_dead))


def main():
    output = ROOT / "data/factorized_baseline"
    output.mkdir(exist_ok=True)
    features = pd.read_pickle(ROOT / "data/rich_baseline/input_features.pkl")
    columns = compact_columns(features)
    x = features[columns]
    # Symmetric dead-ball inputs: closeness, relative team proximity, and movement,
    # without learning a preference for the home/away team.
    dead = x[[name for name in columns if not name.startswith(("h__", "a__"))]].copy()
    for name in ["distance_1", "distance_2", "nearest_speed", "speed_median", "moving_fraction",
                 "within_1m", "within_3m", "within_5m"]:
        pair = x[["h__" + name, "a__" + name]]
        dead[name + "_minimum"] = pair.min(axis=1)
        dead[name + "_maximum"] = pair.max(axis=1)
    gt = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    labels = gt.label.to_numpy()
    public = np.flatnonzero(labels != "X")
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    oof = {name: np.full((len(public), 3), np.nan)
           for name in ["compact_multiclass", "factorized", "physics_control"]}
    for minute, train, validation in folds:
        print("Fold", minute, flush=True)
        x_train = x.iloc[train]
        y = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
        x_aug = pd.concat([x_train, swap_teams(x_train)], ignore_index=True)
        y_aug = np.r_[y, np.where(y == 0, 1, np.where(y == 1, 0, 2))]
        weight = len(y_aug) / (3*np.bincount(y_aug)[y_aug])
        model = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                              tree_method="hist", max_depth=2, n_estimators=120,
                              learning_rate=.04, reg_lambda=10, min_child_weight=15,
                              subsample=.8, colsample_bytree=.9, random_state=42, n_jobs=2)
        model.fit(x_aug, y_aug, sample_weight=weight)
        p = (model.predict_proba(x.iloc[validation]) +
             model.predict_proba(swap_teams(x.iloc[validation]))[:, [1, 0, 2]]) / 2
        oof["compact_multiclass"][validation-public[0]] = p
        model.save_model(output / f"compact_fold_{minute}.ubj")

        is_dead = (labels[train] == "D").astype(int)
        detector = binary_model()
        dw = len(is_dead)/(2*np.bincount(is_dead)[is_dead])
        detector.fit(dead.iloc[train], is_dead, sample_weight=dw)
        pd_ = detector.predict_proba(dead.iloc[validation])[:, 1]
        detector.save_model(output / f"dead_fold_{minute}.ubj")
        alive = y_aug != 2
        controller = binary_model()
        controller.fit(x_aug.loc[alive], (y_aug[alive] == 0).astype(int))
        ph = (controller.predict_proba(x.iloc[validation])[:, 1] +
              1-controller.predict_proba(swap_teams(x.iloc[validation]))[:, 1])/2
        controller.save_model(output / f"control_fold_{minute}.ubj")
        oof["factorized"][validation-public[0]] = compose(pd_, ph)

        # Geometric odds from distances; missing observations borrow up to five
        # seconds of surrounding input evidence, never labels.
        raw = features.h__control_share.copy()
        # Equivalent to soft inverse-distance odds; .5 shrinks weak evidence.
        ph_all = (.5 + (raw-.5)*1.8).clip(.01,.99)
        # A bounded local mean avoids interpolating toward distant future endpoints.
        filled = ph_all.fillna(ph_all.rolling(251, center=True, min_periods=1).mean()).fillna(.5)
        oof["physics_control"][validation-public[0]] = compose(pd_, filled.iloc[validation].to_numpy())

    prediction_table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
    rows = []
    fold_rows = []
    for name, p in oof.items():
        for index, label in enumerate(CLASSES):
            prediction_table[f"{name}_probability_{label}"] = p[:, index]
        for smoothing in [False, True]:
            method = name + ("_smoothed" if smoothing else "_raw")
            prediction = np.empty(len(public), dtype="<U1")
            for minute, _, validation in folds:
                index = validation-public[0]
                block = probability_smoothing(p[index]) if smoothing else p[index]
                prediction[index] = CLASSES[decode(block)]
                result = pm.evaluate(pm.label_codes(labels[validation]), pm.label_codes(prediction[index]))
                fold_rows.append(dict(model=method, minute=minute, spellF1=result["spell_f1_mean"]))
            prediction_table[method] = prediction
            result = pm.evaluate(pm.label_codes(labels[public]), pm.label_codes(prediction), bootstrap_n=1000)
            rows.append(dict(model=method, spellF1=result["spell_f1_mean"], macroF1=result["macro_f1"],
                             possessions=result["n_pred_spells"]))
    pd.DataFrame(rows).to_csv(output / "oof_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_scores.csv", index=False)
    prediction_table.to_csv(output / "oof_predictions.csv", index=False)
    print(pd.DataFrame(fold_rows).pivot(index="model", columns="minute", values="spellF1").round(4))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
