"""Restrict fast ownership changes to close-control evidence; retain slow D."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from control_refinement import contact_support, pooled_control_evidence, smooth, supported_emissions
from evaluate_physics_context import apply_context
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES

CONFIGS = {"current": None}
for threshold in (.65, .75):
    for dead in ("none", "pooled"):
        CONFIGS[f"fast{threshold}_{dead}"] = dict(threshold=threshold, dead_mode=dead, strength=.75)
CONFIGS["fast0.65_pooled_half"] = dict(threshold=.65, dead_mode="pooled", strength=.5)
CONFIGS["fast0.65_pooled_quarter"] = dict(threshold=.65, dead_mode="pooled", strength=.25)


def main():
    output = ROOT/"data/supported_control_research"
    output.mkdir(exist_ok=True)
    source = ROOT/"data/advanced_validation"
    raw = pd.read_pickle(source/"raw.pkl")
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    side = gap_sides(raw)[public]
    support, moving = contact_support(raw)[public], pooled_control_evidence(raw)[public]
    splits = json.loads((source/"splits.json").read_text())
    rows, details, segments = [], [], []
    (output/"configs.json").write_text(json.dumps(CONFIGS, indent=2)+"\n")
    for protocol in ("buffered_minutes", "purged_spells", "buffered_100_seconds"):
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        p = np.load(source/protocol/"ordered_xgboost_frame_seed42_probabilities.npy")
        p = apply_context(p, side, np.zeros(7500), False)
        team = np.load(ROOT/"data/control_refinement"/protocol/"fast_probabilities.npy")
        live = apply_context(np.column_stack((team, np.zeros(7500))), side, np.zeros(7500), False)[:, :2]
        live /= live.sum(axis=1, keepdims=True)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, config in CONFIGS.items():
            prediction = np.empty(7500, dtype="<U1")
            for _, _, validation in folds:
                ix = validation-public[0]
                slow, fast = smooth(p[ix], 25), smooth(live[ix], 5)
                emissions = slow if config is None else supported_emissions(
                    slow, fast, support[ix], moving[ix], **config)
                prediction[ix] = CLASSES[decode(emissions)]
            table[name] = prediction
            record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
        add_short_metrics(rows, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
        pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
