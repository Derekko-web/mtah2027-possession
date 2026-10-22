"""Test ordered control observations with two training objectives and purged CV."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_candidate import PARAMS, objective_weights
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from rich_features import swap_teams
from research_sequence import adaptive_decode, control_support
from research_stoppages import record_predictions
from run_continuation import purged_spell_blocks
from temporal_control import contact_memory, temporal_taps
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing

CONFIGS = {
    "contact_memory": dict(features="memory", depth=2, trees=120),
    "ordered_taps": dict(features="taps", depth=2, trees=120),
    "ordered_depth3": dict(features="taps", depth=3, trees=160),
    "ordered_without_location": dict(features="no_location", depth=3, trees=160),
    "compact_without_location": dict(features="base_no_location", depth=2, trees=120),
    "compact_without_location_depth3": dict(features="base_no_location", depth=3, trees=160),
}


def fit_temporal(inputs, labels, train, validation, config, output, minute):
    y = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
    augmented = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
    params = dict(PARAMS, max_depth=config["depth"], n_estimators=config["trees"])
    predictions = []
    for segment, name in ((False, "frame"), (True, "segment")):
        target, weight = objective_weights(y, train, segment)
        model = XGBClassifier(**params)
        model.fit(augmented, target, sample_weight=weight)
        predictions.append((model.predict_proba(inputs.iloc[validation])+
                            model.predict_proba(swap_teams(inputs.iloc[validation]))[:, [1, 0, 2]])/2)
        model.save_model(output/f"fold_{minute}_{name}.ubj")
    return (predictions[0].astype(float)+predictions[1].astype(float))/2


def main():
    output = ROOT/"data/temporal_control_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    memory, taps = contact_memory(base), temporal_taps(base)
    inputs = {"memory": base.join(memory), "taps": base.join(memory).join(taps)}
    inputs["no_location"] = inputs["taps"].drop(columns=["ball_goal_line_distance", "ball_touchline_distance"])
    inputs["base_no_location"] = base.drop(columns=["ball_goal_line_distance", "ball_touchline_distance"])
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    side = gap_context(raw)[public]
    support = control_support(raw)[public]
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, config in CONFIGS.items():
            print(protocol, name, flush=True)
            target = folder/name
            target.mkdir(exist_ok=True)
            p = np.full((len(public), 3), np.nan)
            for minute, train, validation in blocks:
                p[validation-public[0]] = fit_temporal(inputs[config["features"]], labels, train, validation, config, target, minute)
            np.save(target/"oof_probabilities.npy", p)
            p = apply_context(p, side, np.zeros(len(public)), False)
            for adaptive in (False, True):
                method = name+("_adaptive" if adaptive else "_fixed")
                prediction = np.empty(len(public), dtype="<U1")
                for _, _, validation in blocks:
                    ix = validation-public[0]
                    smooth = probability_smoothing(p[ix])
                    prediction[ix] = CLASSES[adaptive_decode(smooth, support[ix]) if adaptive else decode(smooth)]
                table[method] = prediction
                record_predictions(output, protocol, method, public, labels, blocks, prediction, rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    (output/"manifest.json").write_text(json.dumps(dict(configs=CONFIGS, params=PARAMS,
                features={name: list(x.columns) for name,x in inputs.items()},
                warning="Adaptive same-match public development, ordered input context <=5 seconds, no target-derived inputs."), indent=2)+"\n")
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
