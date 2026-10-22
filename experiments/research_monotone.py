"""Constrain visible-ball specialist to physically consistent response directions.

No pitch location inputs: a stoppage in one area must transfer to another. Slower
balls/players and greater ball-player separation cannot reduce the model's D
probability, with other inputs fixed. This is a hypothesis, not a football law;
base predictions retain responsibility for occluded/fast-ball stoppages.
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
from evaluate_physics_context import apply_context, gap_context
from research_sequence import adaptive_decode, control_support
from research_stoppages import binary_weights, record_predictions
from run_continuation import purged_spell_blocks
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
              max_depth=3, n_estimators=160, learning_rate=.04, reg_lambda=10,
              min_child_weight=15, subsample=.8, colsample_bytree=1., random_state=42, n_jobs=2)


def monotone_inputs(base):
    features = {}
    directions = []
    for name in ("ball_displacement_0.2s", "ball_displacement_1s"):
        features[name] = base[name]
        directions.append(-1)
    for name, direction in (("distance_1", 1), ("nearest_speed", -1),
                            ("speed_median", -1), ("moving_fraction", -1)):
        pair = base[["h__"+name, "a__"+name]]
        features[name+"_minimum"] = pair.min(axis=1)
        features[name+"_maximum"] = pair.max(axis=1)
        directions.extend([direction, direction])
    return pd.DataFrame(features), tuple(directions)


def main():
    output = ROOT/"data/monotone_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    x, direction = monotone_inputs(base)
    x.to_pickle(output/"input_features.pkl")
    support = control_support(raw)[public]
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        base_p = apply_context(np.load(ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy"),
                               gap_context(raw)[public], np.zeros(len(public)), False)
        ph = base_p[:, 0]/base_p[:, :2].sum(axis=1)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for monotone in (False, True):
            for segment_balance in (False, True):
                family = ("monotone" if monotone else "unconstrained")+("_segments" if segment_balance else "_class")
                print(protocol, family, flush=True)
                head = np.full(len(public), np.nan)
                for minute, train, validation in blocks:
                    eligible = train[base.ball_seen.iloc[train].to_numpy() == 1]
                    y = (labels[eligible] == "D").astype(int)
                    model = XGBClassifier(**PARAMS, monotone_constraints=direction if monotone else tuple(0 for _ in direction))
                    model.fit(x.iloc[eligible], y, sample_weight=binary_weights(y, eligible, segment_balance))
                    head[validation-public[0]] = model.predict_proba(x.iloc[validation])[:, 1]
                    model.save_model(folder/f"{family}_fold_{minute}.ubj")
                seen = base.ball_seen.iloc[public].to_numpy() == 1
                head[~seen] = base_p[~seen, 2]
                np.save(folder/f"{family}_head.npy", head)
                for fusion in ("replace", "equal", "union"):
                    dead = head if fusion == "replace" else (head+base_p[:, 2])/2 if fusion == "equal" else np.maximum(base_p[:, 2], head)
                    p = np.column_stack(((1-dead)*ph, (1-dead)*(1-ph), dead))
                    for adaptive in (False, True):
                        method = family+"_"+fusion+("_adaptive" if adaptive else "_fixed")
                        prediction = np.empty(len(public), dtype="<U1")
                        for _, _, validation in blocks:
                            ix = validation-public[0]
                            smooth = probability_smoothing(p[ix])
                            states = adaptive_decode(smooth, support[ix]) if adaptive else decode(smooth)
                            prediction[ix] = CLASSES[states]
                        table[method] = prediction
                        record_predictions(output, protocol, method, public, labels, blocks, prediction,
                                           rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
