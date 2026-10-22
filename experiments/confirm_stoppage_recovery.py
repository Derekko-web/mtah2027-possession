"""Frozen two-view stoppage recovery across all paired seeds and protocols."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from stoppage_consensus import decode_stoppage_consensus
from stoppage_context import fit_context


def main():
    output = ROOT/"data/control_renewal_v1/recovery_confirmation"
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/STOPPAGE_RECOVERY_CONFIRMATION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    cache = ROOT/"data/control_renewal_v1/stoppage_inputs"
    cache_manifest = json.loads((cache/"manifest.json").read_text())
    for name, expected in cache_manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Cached feature/training code changed")
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Confirmation source changed")
    raw, physical, restart = [pd.read_pickle(cache/(name+".pkl")) for name in ("raw", "physical", "restart")]
    weak = np.load(cache/"weak_samples.npz")
    weak_frames, weak_target = weak["frames"], weak["target"]
    labels_path = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(labels_path).label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    if np.any((weak_frames >= public[0]-250) & (weak_frames <= public[-1]+250)):
        raise ValueError("Weak fitting touches public exclusion")
    source = ROOT/"data/advanced_validation"
    splits = json.loads((source/"splits.json").read_text())
    (output/"splits.json").write_bytes((source/"splits.json").read_bytes())
    rows, details, segments = [], [], []
    for protocol in plan["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        blocks = [(int(v[0]), int(v[-1])+1) for _, _, v in folds]
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for seed in plan["seeds"]:
            heads = np.full((len(raw), 2), np.nan)
            for index, (name, inputs) in enumerate((("context_weak", physical), ("context_restart_weak", restart))):
                for block, train, validation in folds:
                    print(protocol, seed, name, "block", block, flush=True)
                    model = fit_context(inputs, labels, train, weak_frames, weak_target, .5, seed)
                    model.save_model(folder/f"{name}_seed{seed}_block{block}.ubj")
                    heads[validation, index] = model.predict_proba(inputs.iloc[validation])[:, 1]
                if seed == 42:
                    previous = np.load(ROOT/"data/control_renewal_v1/stoppage_screen"/protocol/f"{name}_seed42_probabilities.npy")
                    np.testing.assert_array_equal(heads[public, index], previous)
            np.save(folder/f"heads_seed{seed}_probabilities.npy", heads[public])
            arrays = []
            paths = [source/protocol/f"{name}_seed{seed}_probabilities.npy" for name in
                     ("ordered_xgboost_frame", "previous_ensemble")]
            paths.append(ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy")
            for path, width in zip(paths, (3, 3, 2)):
                value = np.full((len(raw), width), np.nan)
                value[public] = np.load(path)
                arrays.append(value)
            for name in plan["methods"]:
                method = f"{name}_seed{seed}"
                prediction = decode_stoppage_consensus(*arrays, heads, raw, blocks, name != "current")[public]
                if name == "current":
                    reference = pd.read_csv(ROOT/"data/control_renewal_v1/baseline_confirmation"/protocol/"oof_predictions.csv")
                    np.testing.assert_array_equal(prediction, reference[f"anchored_rescue_seed{seed}"].to_numpy())
                table[method] = prediction
                record_predictions(output, protocol, method, public, labels, folds, prediction, rows, details, segments)
                add_overlap_metrics(rows, segments, table, protocol, method)
            add_short_metrics(rows, segments)
            table.to_csv(folder/"oof_predictions.csv", index=False)
            pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
            pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
            pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    manifest = dict(plan_sha256=digest(plan_path), public_gt_sha256=digest(labels_path),
                    cache_sha256={name: digest(cache/name) for name in ("raw.pkl", "physical.pkl", "restart.pkl", "weak_samples.npz", "manifest.json")},
                    source_sha256=plan["source_sha256"], seed42_head_probability_parity=True)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol", "model", "spellF1", "macroF1", "false_dead_frames", "dead_recall", "missed_stoppages"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
