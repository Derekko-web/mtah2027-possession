"""Teach missing-ball robustness with label-blind synthetic occlusions.

Every ten seconds a seeded two-second span hides the ball in a separate local
tracking copy. All ball-dependent inputs and temporal history are rebuilt from
that copy. Actual validation inputs remain untouched. Original and occluded
training copies each receive half of the original fitting weight.
"""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_candidate import PARAMS, objective_weights
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from games import require
import ball_kinematics as kin
import relative_motion as rel
import rich_features as rich
from research_stoppages import record_predictions
from research_temporal_decoding import smooth
from rich_features import swap_teams
from run_continuation import purged_spell_blocks
from temporal_control import contact_memory, temporal_taps
from train_factorized_baseline import compact_columns
from transition_penalty import decode
from xgboost_baseline import CLASSES


def ordered_inputs(base):
    return base.join(contact_memory(base)).join(temporal_taps(base)).drop(
        columns=["ball_goal_line_distance", "ball_touchline_distance"])


def augmented_inputs(n_frames, output):
    metadata, tracking = require(["atd_meta", "atd_raw"])
    rng = np.random.default_rng(42)
    masked = np.zeros(n_frames, dtype=bool)
    for start in range(0, n_frames, 250):
        left = start+int(rng.integers(0, 200))
        masked[left:min(left+50, n_frames)] = True
    path = output/"occluded_tracking.txt"
    with open(tracking) as source, path.open("w") as target:
        for frame, line in enumerate(source):
            if frame >= n_frames:
                break
            if masked[frame]:
                index, players, _ = line.strip().split(":")
                target.write(index+":"+players+":NaN,NaN,NaN\n")
            else:
                target.write(line)
    # Scoped dependency injection preserves all selected feature code and data.
    with patch.object(rich, "require", return_value=[metadata, str(path)]), \
         patch.object(kin, "require", return_value=[str(path)]), \
         patch.object(rel, "require", return_value=[metadata, str(path)]):
        raw = rich.build_features(n_frames)
        base = raw[compact_columns(raw)].join(kin.build_kinematics(n_frames)).join(rel.build_relative_motion(n_frames))
    return ordered_inputs(base)


def main():
    output = ROOT/"data/ball_dropout_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    actual = ordered_inputs(base)
    augmented = augmented_inputs(len(base), output)
    augmented.to_pickle(output/"augmented_input_features.pkl")
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
            x = pd.concat([actual.iloc[train], swap_teams(actual.iloc[train]),
                           augmented.iloc[train], swap_teams(augmented.iloc[train])], ignore_index=True)
            probabilities = []
            for segment, name in ((False, "frame"), (True, "segment")):
                y, weight = objective_weights(target, train, segment)
                model = XGBClassifier(**params)
                model.fit(x, np.tile(y, 2), sample_weight=np.tile(weight/2, 2))
                model.save_model(folder/f"fold_{minute}_{name}.ubj")
                probabilities.append((model.predict_proba(actual.iloc[validation])+
                    model.predict_proba(swap_teams(actual.iloc[validation]))[:, [1,0,2]])/2)
            p[validation-public[0]] = (probabilities[0].astype(float)+probabilities[1].astype(float))/2
        np.save(folder/"oof_probabilities.npy", p)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        context = apply_context(p, gap_context(raw)[public], np.zeros(len(public)), False)
        for name, probability in (("no_prior", p), ("prior", context)):
            for half_window in (5, 25):
                method = name+f"_half{half_window}"
                prediction = np.empty(len(public), dtype="<U1")
                for _, _, validation in blocks:
                    ix = validation-public[0]
                    prediction[ix] = CLASSES[decode(smooth(probability[ix], half_window))]
                table[method] = prediction
                record_predictions(output, protocol, method, public, labels, blocks, prediction, rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    (output/"parameters.json").write_text(json.dumps(dict(params=params, seed=42,
                occlusion_frames=50, occlusion_per_frames=250, synthetic_copy_weight=.5), indent=2)+"\n")
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
