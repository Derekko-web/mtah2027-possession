"""Disaggregate the ordered model's objectives and test direct vs penalized decoding."""

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
from research_stoppages import record_predictions
from research_temporal_decoding import smooth
from rich_features import swap_teams
from run_continuation import purged_spell_blocks
from temporal_control import contact_memory, temporal_taps
from transition_penalty import decode
from xgboost_baseline import CLASSES


def main():
    output = ROOT/"data/objective_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    inputs = base.join(contact_memory(base)).join(temporal_taps(base)).drop(
        columns=["ball_goal_line_distance", "ball_touchline_distance"])
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        source = ROOT/"data/temporal_control_research"/protocol/"ordered_without_location"
        probabilities = {name: np.full((len(public), 3), np.nan) for name in ("frame", "segment")}
        for minute, _, validation in blocks:
            for name in probabilities:
                model = XGBClassifier()
                model.load_model(source/f"fold_{minute}_{name}.ubj")
                probabilities[name][validation-public[0]] = (model.predict_proba(inputs.iloc[validation])+
                    model.predict_proba(swap_teams(inputs.iloc[validation]))[:, [1,0,2]])/2
        probabilities["ensemble"] = (probabilities["frame"]+probabilities["segment"])/2
        np.testing.assert_allclose(probabilities["ensemble"], np.load(source/"oof_probabilities.npy"), atol=1e-7)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, p in probabilities.items():
            np.save(folder/f"{name}_oof_probabilities.npy", p)
            p = apply_context(p, gap_context(raw)[public], np.zeros(len(public)), False)
            for half_window in (5, 25):
                for penalty in (0, 3):
                    method = name+f"_half{half_window}_penalty{penalty}"
                    prediction = np.empty(len(public), dtype="<U1")
                    for _, _, validation in blocks:
                        ix = validation-public[0]
                        prediction[ix] = CLASSES[decode(smooth(p[ix], half_window), penalty=penalty)]
                    table[method] = prediction
                    record_predictions(output, protocol, method, public, labels, blocks, prediction, rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
