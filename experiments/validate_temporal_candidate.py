"""Frozen finalists: padded observations, three exclusion protocols, three seeds."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from research_stoppages import record_predictions
from run_continuation import purged_spell_blocks
from temporal_candidate import METHODS, build_inputs, decode_blocks, fit_models, gap_sides, predict_models

PLAN = ROOT/"experiments/ADVANCED_VALIDATION_PLAN.json"


def long_blocks(public, size=2500, gap=250):
    if len(public) % size or not np.array_equal(public, np.arange(public[0], public[-1]+1)):
        raise ValueError("Expected contiguous complete long blocks")
    result = []
    for i, start in enumerate(range(int(public[0]), int(public[-1])+1, size), 1):
        end = start+size
        train = public[(public < start-gap) | (public >= end+gap)]
        result.append((i, train, np.arange(start, end)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"data/advanced_validation")
    args = parser.parse_args()
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text())
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    # Always rebuild: a same-shaped cache could contain older feature logic or
    # observations from different assets and would invalidate the comparison.
    print("Building padded observation context", flush=True)
    raw, base, ordered = build_inputs(plan["feature_frames_exclusive_end"])
    raw.to_pickle(output/"raw.pkl")
    base.to_pickle(output/"base.pkl")
    ordered.to_pickle(output/"ordered.pkl")
    if len(raw) != plan["feature_frames_exclusive_end"] or len(base.columns) != 46 or len(ordered.columns) != 102:
        raise ValueError("Unexpected cached feature layout")
    minute_folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": minute_folds,
                 "purged_spells": purged_spell_blocks(labels, minute_folds),
                 "buffered_100_seconds": long_blocks(public)}
    (output/"plan.json").write_bytes(PLAN.read_bytes())
    (output/"plan_sha256.txt").write_text(hashlib.sha256(PLAN.read_bytes()).hexdigest()+"\n")
    splits = []
    for protocol, folds in protocols.items():
        for i, train, validation in folds:
            if np.intersect1d(train, validation).size:
                raise ValueError("Training/validation overlap")
            splits.append(dict(protocol=protocol, block=i, train=train.tolist(), validation=validation.tolist()))
    (output/"splits.json").write_text(json.dumps(splits)+"\n")
    side = gap_sides(raw)[public]
    rows, details, segments = [], [], []
    for protocol, folds in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for method in METHODS:
            inputs = base if method == "previous_ensemble" else ordered
            for seed in plan["seeds"]:
                name = method+f"_seed{seed}"
                p = np.full((len(public), 3), np.nan)
                for i, train, validation in folds:
                    print(protocol, name, "block", i, flush=True)
                    models = fit_models(method, inputs, labels, train, seed)
                    p[validation-public[0]] = predict_models(models, inputs.iloc[validation])
                    for j, model in enumerate(models):
                        suffix = "cbm" if "catboost" in method else "ubj"
                        model.save_model(str(folder/f"{name}_block{i}_{j}.{suffix}"))
                np.save(folder/f"{name}_probabilities.npy", p)
                blocks = [(int(v[0]-public[0]), int(v[-1]-public[0]+1)) for _, _, v in folds]
                prediction = decode_blocks(p, side, blocks)
                table[name] = prediction
                record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
                # Save progress after each complete method/seed; no partial OOF scoring.
                pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
                pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
                pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
                table.to_csv(folder/"oof_predictions.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
