"""Concentrate existing training mass near true training-only state boundaries.

The class-balanced model retains each class's total weight. The segment model
retains every eligible segment's total weight. Within those budgets, frames
within one second of an observed training change get three times the emphasis.
No boundary is inferred across a gap or from held-out labels.
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
from build_candidate import PARAMS, objective_weights
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from research_sequence import adaptive_decode, control_support
from research_stoppages import record_predictions
from research_temporal_decoding import smooth
from rich_features import swap_teams
from run_continuation import purged_spell_blocks
from temporal_control import contact_memory, temporal_taps
from xgboost_baseline import CLASSES


def boundary_weights(target, frames, segment_balance):
    target = np.asarray(target)
    frames = np.asarray(frames)
    _, augmented_weights = objective_weights(target, frames, segment_balance)
    original = augmented_weights[:len(target)]
    boundaries = frames[1:][(np.diff(frames) == 1) & (target[1:] != target[:-1])]
    edge = np.zeros(len(frames), dtype=bool)
    for boundary in boundaries:
        edge |= np.abs(frames-boundary) <= 25
    weights = original*np.where(edge, 3., 1.)
    if segment_balance:
        group = np.cumsum(np.r_[True, (np.diff(frames) != 1) | (target[1:] != target[:-1])])-1
    else:
        # Home/away augmentation gives both team classes the same total mass.
        group = np.where(target == 2, 1, 0)
    for index in np.unique(group):
        mask = group == index
        weights[mask] *= original[mask].sum()/weights[mask].sum()
    return np.r_[weights, weights]


def main():
    output = ROOT/"data/boundary_training_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    inputs = base.join(contact_memory(base)).join(temporal_taps(base)).drop(
        columns=["ball_goal_line_distance", "ball_touchline_distance"])
    support = control_support(raw)[public]
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    params = dict(PARAMS, max_depth=3, n_estimators=160)
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        p = np.full((len(public), 3), np.nan)
        for minute, train, validation in blocks:
            print(protocol, "fold", minute, flush=True)
            target = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
            augmented_x = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
            predictions = []
            for balance, name in ((False, "frame"), (True, "segment")):
                augmented_y, _ = objective_weights(target, train, balance)
                weights = boundary_weights(target, train, balance)
                model = XGBClassifier(**params)
                model.fit(augmented_x, augmented_y, sample_weight=weights)
                model.save_model(folder/f"fold_{minute}_{name}.ubj")
                predictions.append((model.predict_proba(inputs.iloc[validation])+
                    model.predict_proba(swap_teams(inputs.iloc[validation]))[:, [1,0,2]])/2)
            p[validation-public[0]] = (predictions[0].astype(float)+predictions[1].astype(float))/2
        np.save(folder/"oof_probabilities.npy", p)
        p = apply_context(p, gap_context(raw)[public], np.zeros(len(public)), False)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for window in (0, 5, 25):
            for adaptive in (False, True):
                method = f"boundary_half{window}"+("_adaptive" if adaptive else "_fixed")
                prediction = np.empty(len(public), dtype="<U1")
                for _, _, validation in blocks:
                    ix = validation-public[0]
                    probability = smooth(p[ix], window)
                    if adaptive:
                        prediction[ix] = CLASSES[adaptive_decode(probability, support[ix])]
                    else:
                        from transition_penalty import decode
                        prediction[ix] = CLASSES[decode(probability)]
                table[method] = prediction
                record_predictions(output, protocol, method, public, labels, blocks, prediction, rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    (output/"parameters.json").write_text(json.dumps(dict(params=params, boundary_radius=25,
                emphasis=3, preserved_mass="Class totals or individual segment totals"), indent=2)+"\n")
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
