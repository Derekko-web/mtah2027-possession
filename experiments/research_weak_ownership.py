"""Fixed ownership weak-supervision screen; all competition processing is local."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from control_refinement import contact_support, predict_control
from phase_control import phase_control
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import ordered_inputs
from train_factorized_baseline import compact_columns
from weak_ownership import fit_weak_ownership, ownership_prediction, sample_weak_ownership
from xgboost_baseline import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/control_precision_v1/weak_ownership_screen")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/WEAK_OWNERSHIP_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    cache = ROOT/"data/control_renewal_v1/stoppage_inputs"
    frozen_cache = json.loads((ROOT/"data/control_precision_v1/baseline_confirmation/manifest.json").read_text())["cache_sha256"]
    for name, expected in frozen_cache.items():
        if digest(cache/name) != expected:
            raise ValueError("Frozen stoppage cache identity changed: "+name)
    cache_manifest = json.loads((cache/"manifest.json").read_text())
    for name, expected in cache_manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Full-cache source changed: "+name)
    raw, physical, restart = [pd.read_pickle(cache/(name+".pkl")) for name in ("raw", "physical", "restart")]
    phase = phase_control(len(raw))
    inputs = ordered_inputs(raw[compact_columns(raw)])
    views = {"geometry": inputs, "semantic": inputs.join(phase)}
    # Public padding must reproduce full-match feature construction exactly.
    prefix = ordered_inputs(raw.iloc[:18935][compact_columns(raw)])
    pd.testing.assert_frame_equal(inputs.iloc[11185:18685], prefix.iloc[11185:18685])
    pd.testing.assert_frame_equal(phase.iloc[11185:18685], phase_control(18935).iloc[11185:18685])
    reserved = ROOT/"data/continuation_20261021/annotation_freeze/windows.json"
    if digest(reserved) != "5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c":
        raise ValueError("Frozen annotation windows changed")
    windows = json.loads(reserved.read_text())["windows"]
    weak = sample_weak_ownership(raw, physical, restart, phase, windows)
    counts = {}
    for name, (frames, target) in weak.items():
        if np.any((frames >= 10935) & (frames <= 18934)):
            raise AssertionError("Weak samples touch public exclusion")
        for window in windows:
            if np.any((frames >= window["start"]-250) & (frames <= window["end"]+250)):
                raise AssertionError("Weak samples touch reserved exclusion")
        counts[name] = dict(H=int((target == 0).sum()), A=int((target == 1).sum()))
        np.savez(output/f"weak_{name}.npz", frames=frames, target=target)
    print("Weak ownership counts", counts, flush=True)
    for name, x in views.items():
        x.to_pickle(output/f"{name}_inputs.pkl")
    source = ROOT/"data/advanced_validation"
    reference = ROOT/"data/control_precision_v1/baseline_confirmation"
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    phase_support = phase[[prefix+"phase_control_support" for prefix in ("h__", "a__")]].to_numpy()[public]
    support = contact_support(raw)[public]
    splits_path = source/"splits.json"
    splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    rows, details, segments = [], [], []
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in
              (gt, reserved, splits_path, plan_path, cache/"manifest.json", cache/"raw.pkl", cache/"physical.pkl", cache/"restart.pkl")}
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        ref_path = reference/protocol/"oof_predictions.csv"
        hashes[str(ref_path.relative_to(ROOT))] = digest(ref_path)
        original = pd.read_csv(ref_path)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for seed in plan["screen"]["seeds"]:
            selected = original[f"stoppage_consensus_seed{seed}"].to_numpy()
            states = np.array([np.flatnonzero(CLASSES == label)[0] for label in selected])
            method = f"current_seed{seed}"
            table[method] = selected
            record_predictions(output, protocol, method, public, labels, folds, selected, rows, details, segments)
            add_overlap_metrics(rows, segments, table, protocol, method)
            for name, config in plan["screen"]["models"].items():
                x = views["semantic" if config["semantic"] else "geometry"]
                frames, target = weak.get(config["weak"], (np.array([], int), np.array([], int)))
                probability = np.full((len(public), 2), np.nan)
                for block, train, validation in folds:
                    print(protocol, seed, name, "block", block, flush=True)
                    model = fit_weak_ownership(x, labels, train, frames, target, config["weak_mass"], seed)
                    model.save_model(folder/f"{name}_seed{seed}_block{block}.ubj")
                    probability[validation-public[0]] = predict_control(model, x.iloc[validation])
                np.save(folder/f"{name}_seed{seed}_probabilities.npy", probability)
                for mode in plan["screen"]["decoders"]:
                    prediction = np.empty(len(public), dtype="<U1")
                    for _, _, validation in folds:
                        ix = validation-public[0]
                        prediction[ix] = CLASSES[ownership_prediction(states[ix], probability[ix], support[ix], phase_support[ix], mode)]
                    np.testing.assert_array_equal(prediction == "D", selected == "D")
                    method = f"{name}_{mode}_seed{seed}"
                    table[method] = prediction
                    record_predictions(output, protocol, method, public, labels, folds, prediction, rows, details, segments)
                    add_overlap_metrics(rows, segments, table, protocol, method)
                add_short_metrics(rows, segments)
                table.to_csv(folder/"oof_predictions.csv", index=False)
                pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
                pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
                pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    names = ("research_weak_ownership.py", "weak_ownership.py", "phase_control.py", "temporal_candidate.py",
             "temporal_control.py", "train_factorized_baseline.py", "control_refinement.py", "rich_features.py")
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=digest(plan_path), input_sha256=hashes,
        source_sha256={name:digest(ROOT/"experiments"/name) for name in names}, weak_counts=counts,
        public_prefix_parity=True, all_D_masks_unchanged=True, public_reserved_exclusions_verified=True,
        feature_columns={name:list(x) for name,x in views.items()}), indent=2)+"\n")
    columns = ["protocol", "model", "spellF1", "macroF1", "short_live_mean_best_iou", "short_live_matched_iou50"]
    print(pd.DataFrame(rows)[columns].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
