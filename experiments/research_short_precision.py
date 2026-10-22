"""Frozen endpoint screen from verified six-model OOF references; no fitting."""

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
from evaluate_physics_context import apply_context
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from short_precision import ownership_evidence, refine_short_endpoints
from stoppage_consensus import decode_stoppage_consensus
from temporal_candidate import gap_sides
from xgboost_baseline import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/control_precision_v1/screen")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/SHORT_PRECISION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    source = ROOT/"data/advanced_validation"
    reference = ROOT/"data/control_precision_v1/baseline_confirmation"
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    raw = pd.read_pickle(source/"raw.pkl")
    labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    splits_path = source/"splits.json"
    splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in (gt, source/"raw.pkl", splits_path, plan_path)}
    side = gap_sides(raw)[public]
    rows, details, segments, signals = [], [], [], []
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        blocks = [(int(v[0]), int(v[-1])+1) for _, _, v in folds]
        original = pd.read_csv(reference/protocol/"oof_predictions.csv")
        hashes[str((reference/protocol/"oof_predictions.csv").relative_to(ROOT))] = digest(reference/protocol/"oof_predictions.csv")
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for seed in plan["screen"]["seeds"]:
            paths = [source/protocol/f"{name}_seed{seed}_probabilities.npy"
                     for name in ("ordered_xgboost_frame", "previous_ensemble")]
            paths += [ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy",
                      reference/protocol/f"heads_seed{seed}_probabilities.npy"]
            arrays = []
            for path, width in zip(paths, (3, 3, 2, 2)):
                hashes[str(path.relative_to(ROOT))] = digest(path)
                value = np.full((len(raw), width), np.nan)
                value[public] = np.load(path)
                arrays.append(value)
            selected = decode_stoppage_consensus(*arrays, raw, blocks)[public]
            np.testing.assert_array_equal(selected, original[f"stoppage_consensus_seed{seed}"])
            previous = apply_context(arrays[1][public], side, np.zeros(len(public)), False)
            fast = apply_context(np.column_stack((arrays[2][public], np.zeros(len(public)))),
                                 side, np.zeros(len(public)), False)[:, :2]
            fast /= fast.sum(axis=1, keepdims=True)
            for name, config in {"current": None, **plan["screen"]["methods"]}.items():
                prediction = selected.copy()
                if config is not None:
                    for _, _, validation in folds:
                        ix = validation-public[0]
                        states = np.array([np.flatnonzero(CLASSES == label)[0] for label in selected[ix]])
                        evidence = ownership_evidence(previous[ix], fast[ix], **{k:v for k,v in config.items() if k != "radius"})
                        prediction[ix] = CLASSES[refine_short_endpoints(states, evidence, config["radius"])]
                    np.testing.assert_array_equal(prediction == "D", selected == "D")
                method = f"{name}_seed{seed}"
                table[method] = prediction
                record_predictions(output, protocol, method, public, labels, folds, prediction, rows, details, segments)
                add_overlap_metrics(rows, segments, table, protocol, method)
                print(protocol, method, "spellF1", rows[-1]["spellF1"], flush=True)
            # Aggregate model evidence by short true interval; never fed back to inference.
            selected_spans = [s for s in segments if s["protocol"] == protocol and
                              s["model"] == f"current_seed{seed}" and s["label"] != "D" and s["frames"] <= 125]
            for span in selected_spans:
                ix = np.arange(span["start"], span["end"]+1)-public[0]
                for name, p in (("previous", previous[:, :2]), ("fast", fast)):
                    signals.append(dict(protocol=protocol, seed=seed, start=span["start"], end=span["end"],
                                        frames=span["frames"], head=name,
                                        raw_correct_argmax=float(np.mean(CLASSES[p[ix].argmax(axis=1)] == span["label"]))))
            add_short_metrics(rows, segments)
            table.to_csv(folder/"oof_predictions.csv", index=False)
            pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
            pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
            pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
            pd.DataFrame(signals).to_csv(output/"short_head_diagnostics.csv", index=False)
    names = ("research_short_precision.py", "short_precision.py", "stoppage_consensus.py",
             "renewal_control.py", "boundary_refinement.py", "control_refinement.py", "transition_penalty.py")
    manifest = dict(plan_sha256=digest(plan_path), input_sha256=hashes,
                    source_sha256={name:digest(ROOT/"experiments"/name) for name in names},
                    reference_decoder_parity=True, all_D_masks_unchanged=True, no_models_fitted=True)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    columns = ["protocol", "model", "spellF1", "macroF1", "short_live_mean_best_iou", "short_live_matched_iou50"]
    print(pd.DataFrame(rows)[columns].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
