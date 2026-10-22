"""Run the prospective boundary screen or unchanged paired confirmation locally."""

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
from boundary_refinement import fit_micro, micro_inputs, refine_block
from control_refinement import contact_support, control_evidence, predict_control
from evaluate_physics_context import apply_context
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import gap_sides
from xgboost_baseline import CLASSES


def add_overlap_metrics(rows, segments, table, protocol, method):
    """Best same-label IoU is diagnostic only; no target enters inference."""
    prediction = table[method].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(prediction[1:] != prediction[:-1])+1, len(prediction)]
    short = [s for s in segments if s["protocol"] == protocol and s["model"] == method
             and s["label"] != "D" and s["frames"] <= 125]
    for span in short:
        start, end = span["start"]-int(table.frame.iloc[0]), span["end"]-int(table.frame.iloc[0])+1
        overlap = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if prediction[left] == span["label"]:
                intersection = max(0, min(end, right)-max(start, left))
                overlap.append(intersection/(end-start+right-left-intersection))
        span["best_same_label_iou"] = max(overlap, default=0.)
    row = next(r for r in rows if r["protocol"] == protocol and r["model"] == method)
    row["short_live_mean_best_iou"] = float(np.mean([s["best_same_label_iou"] for s in short]))
    row["short_live_matched_iou50"] = sum(s["best_same_label_iou"] >= .5 for s in short)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "confirmation"), default="screen")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan_path = ROOT/"experiments"/("BOUNDARY_REFINEMENT_PLAN.json" if args.stage == "screen"
                                      else "BOUNDARY_CONFIRMATION_PLAN.json")
    plan = json.loads(plan_path.read_text())
    settings = plan["screen"] if args.stage == "screen" else plan
    output = args.output or ROOT/"data/control_boundary_v1"/args.stage
    output.mkdir(parents=True, exist_ok=False)
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    (output/"plan_sha256.txt").write_text(hashlib.sha256(plan_path.read_bytes()).hexdigest()+"\n")
    source = ROOT/"data/advanced_validation"
    confirmed = ROOT/"data/control_boundary_v1/baseline_confirmation"
    raw, base = [pd.read_pickle(confirmed/(name+".pkl")) for name in ("raw", "base")]
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    side = gap_sides(raw)[public]
    support, evidence = contact_support(raw)[public], control_evidence(raw)[public]
    micro = micro_inputs(base)
    micro.to_pickle(output/"micro_inputs.pkl")
    splits_path = source/"splits.json"
    splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    rows, details, segments, hashes = [], [], [], {}
    for protocol in settings["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for seed in settings["seeds"]:
            print(protocol, "seed", seed, flush=True)
            arrays = []
            paths = (source/protocol/f"ordered_xgboost_frame_seed{seed}_probabilities.npy",
                     source/protocol/f"previous_ensemble_seed{seed}_probabilities.npy",
                     confirmed/protocol/f"fast_seed{seed}_probabilities.npy")
            for path in paths:
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                arrays.append(np.load(path))
            current, previous, fast = arrays
            current = apply_context(current, side, np.zeros(len(public)), False)
            previous = apply_context(previous, side, np.zeros(len(public)), False)
            heads = {"original": fast}
            for name in ("micro_frame", "micro_segments"):
                if name not in settings["methods"]:
                    continue
                values = np.full_like(fast, np.nan)
                for block, train, validation in folds:
                    model = fit_micro(micro, labels, train, name == "micro_segments", seed)
                    model.save_model(folder/f"{name}_seed{seed}_block{block}.ubj")
                    values[validation-public[0]] = predict_control(model, micro.iloc[validation])
                np.save(folder/f"{name}_seed{seed}_probabilities.npy", values)
                heads[name] = values
            for name, values in heads.items():
                values = apply_context(np.column_stack((values, np.zeros(len(public)))), side,
                                       np.zeros(len(public)), False)[:, :2]
                heads[name] = values/values.sum(axis=1, keepdims=True)
            for method in settings["methods"]:
                name = f"{method}_seed{seed}"
                prediction = np.empty(len(public), dtype="<U1")
                for _, _, validation in folds:
                    ix = validation-public[0]
                    prediction[ix] = CLASSES[refine_block(
                        current[ix], previous[ix], heads.get(method, heads["original"])[ix],
                        support[ix], evidence[ix], method)]
                if method == "current":
                    reference = pd.read_csv(confirmed/protocol/"oof_predictions.csv")[f"consensus_contact_seed{seed}"]
                    np.testing.assert_array_equal(prediction, reference.to_numpy())
                table[name] = prediction
                record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
                add_overlap_metrics(rows, segments, table, protocol, name)
            table.to_csv(folder/"oof_predictions.csv", index=False)
            add_short_metrics(rows, segments)
            pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
            pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
            pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    sources = ("boundary_refinement.py", "research_boundary_refinement.py", "consensus_control.py",
               "control_refinement.py", "temporal_candidate.py", "evaluate_physics_context.py",
               "transition_penalty.py", "research_stoppages.py", "sequence_diagnostics.py")
    manifest = dict(source_sha256={str(Path("experiments")/name): hashlib.sha256(
        (ROOT/"experiments"/name).read_bytes()).hexdigest() for name in sources}, input_sha256=hashes,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        warning="Adaptive dependent public-window evaluations, no independent test or private labels.")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    columns = ["protocol", "model", "spellF1", "macroF1", "false_dead_frames", "dead_recall",
               "missed_stoppages", "short_live_mean_recall", "short_live_mean_best_iou", "short_live_matched_iou50"]
    print(pd.DataFrame(rows)[columns].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
