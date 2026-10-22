"""Fixed CatBoost alternative on geography-free compact and ordered observations."""

import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_candidate import objective_weights
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from research_sequence import adaptive_decode, control_support
from research_stoppages import record_predictions
from rich_features import swap_teams
from run_continuation import purged_spell_blocks
from temporal_control import contact_memory, temporal_taps
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing

PARAMS = dict(loss_function="MultiClass", depth=4, iterations=300, learning_rate=.04,
              l2_leaf_reg=10, random_seed=42, thread_count=2, allow_writing_files=False,
              verbose=False, nan_mode="Min", boosting_type="Ordered")


def main():
    output = ROOT/"data/catboost_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    views = {"compact": base.drop(columns=["ball_goal_line_distance", "ball_touchline_distance"]),
             "ordered": base.join(contact_memory(base)).join(temporal_taps(base)).drop(
                 columns=["ball_goal_line_distance", "ball_touchline_distance"])}
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    side = gap_context(raw)[public]
    support = control_support(raw)[public]
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for view, inputs in views.items():
            objectives = {name: np.full((len(public), 3), np.nan) for name in ("frame", "segment")}
            for minute, train, validation in blocks:
                print(protocol, view, "fold", minute, flush=True)
                target = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
                augmented_x = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
                for balance, name in ((False, "frame"), (True, "segment")):
                    augmented_y, weight = objective_weights(target, train, balance)
                    model = CatBoostClassifier(**PARAMS)
                    model.fit(augmented_x, augmented_y, sample_weight=weight)
                    objectives[name][validation-public[0]] = (model.predict_proba(inputs.iloc[validation])+
                         model.predict_proba(swap_teams(inputs.iloc[validation]))[:, [1,0,2]])/2
                    model.save_model(str(folder/f"{view}_{name}_fold_{minute}.cbm"))
            objectives["ensemble"] = (objectives["frame"]+objectives["segment"])/2
            for objective, p in objectives.items():
                np.save(folder/f"{view}_{objective}_oof_probabilities.npy", p)
                p = apply_context(p, side, np.zeros(len(public)), False)
                for adaptive in (False, True):
                    method = view+"_"+objective+("_adaptive" if adaptive else "_fixed")
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
    (output/"parameters.json").write_text(json.dumps(PARAMS, indent=2)+"\n")
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
