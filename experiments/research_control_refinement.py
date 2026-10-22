"""Fit bounded fast/live heads and score every fixed refinement on held-out rows."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from control_refinement import (compose, control_evidence, fast_columns, fit_control,
                                fit_dead_weight, predict_control, smooth)
from evaluate_physics_context import apply_context
from research_stoppages import record_predictions
from temporal_candidate import gap_sides, predict_models
from transition_penalty import decode
from xgboost_baseline import CLASSES


def add_short_metrics(rows, segments):
    for row in rows:
        short = [s for s in segments if s["protocol"] == row["protocol"] and s["model"] == row["model"]
                 and s["label"] != "D" and s["frames"] <= 125]
        row["short_live_mean_recall"] = float(np.mean([s["recall"] for s in short]))
        row["short_live_mostly_recovered"] = sum(s["recall"] >= .5 for s in short)


def main():
    output = ROOT/"data/control_refinement"
    output.mkdir(exist_ok=True)
    source = ROOT/"data/advanced_validation"
    raw, base, ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw", "base", "ordered")]
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    splits = json.loads((source/"splits.json").read_text())
    side = gap_sides(raw)[public]
    evidence = control_evidence(raw)[public]
    views = {"fast": base[fast_columns(base)], "ordered": ordered}
    rows, details, segments = [], [], []
    for protocol in ("buffered_minutes", "purged_spells", "buffered_100_seconds"):
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        heads = {name: np.full((7500, 2), np.nan) for name in views}
        weights = {mode: np.full((7500, 3), np.nan) for mode in ("empirical", "dead075")}
        for block, train, validation in folds:
            print(protocol, "block", block, flush=True)
            ix = validation-public[0]
            for name, inputs in views.items():
                model = fit_control(inputs, labels, train)
                heads[name][ix] = predict_control(model, inputs.iloc[validation])
                model.save_model(folder/f"{name}_block{block}.ubj")
            for mode in weights:
                model = fit_dead_weight(ordered, labels, train, mode)
                weights[mode][ix] = predict_models([model], ordered.iloc[validation])
                model.save_model(folder/f"{mode}_block{block}.ubj")
        for name, values in {**heads, **weights}.items():
            np.save(folder/(name+"_probabilities.npy"), values)
        original = np.load(source/protocol/"ordered_xgboost_frame_seed42_probabilities.npy")
        adjusted = apply_context(original, side, np.zeros(7500), False)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        predictions = {}
        for block, _, validation in folds:
            ix = validation-public[0]
            p = adjusted[ix]
            slow = smooth(p, 25)
            methods = {"current": slow}
            for mode, value in weights.items():
                methods[mode] = smooth(apply_context(value[ix], side[ix], np.zeros(len(ix)), False), 25)
            methods["dead_odds075"] = slow*np.array([1, 1, .75])
            methods["dead_odds050"] = slow*np.array([1, 1, .5])
            for name, values in heads.items():
                fast = smooth(values[ix], 5)
                # Keep original continuity evidence during qualified occlusions.
                live = np.column_stack((fast, np.zeros(len(ix))))
                fast = apply_context(live, side[ix], np.zeros(len(ix)), False)[:, :2]
                fast /= fast.sum(axis=1, keepdims=True)
                old_team = slow[:, :2]/slow[:, :2].sum(axis=1, keepdims=True)
                for mix in (.5, 1.):
                    team = mix*fast+(1-mix)*old_team
                    key = name+f"_team{mix}"
                    methods[key] = compose(slow[:, 2], team)
                    methods[key+"_contact"] = compose(slow[:, 2]*(1-.75*evidence[ix].max(axis=1)), team)
            methods["current_contact"] = compose(slow[:, 2]*(1-.75*evidence[ix].max(axis=1)),
                                                 slow[:, :2]/slow[:, :2].sum(axis=1, keepdims=True))
            for name, value in methods.items():
                predictions.setdefault(name, np.empty(7500, dtype="<U1"))
                predictions[name][ix] = CLASSES[decode(value)]
        for name, prediction in predictions.items():
            table[name] = prediction
            record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
        add_short_metrics(rows, segments)
        pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    (output/"plan.json").write_bytes((ROOT/"experiments/CONTROL_REFINEMENT_PLAN.json").read_bytes())
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
