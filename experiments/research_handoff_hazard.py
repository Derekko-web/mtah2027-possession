"""Frozen directed handoff timing screen against anchored rescue."""

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
from boundary_refinement import micro_inputs
from control_chronology import contact_chronology
from control_refinement import contact_support, control_evidence
from evaluate_physics_context import apply_context
from handoff_hazard import fit_hazard, predict_hazard, relocate_handoffs
from renewal_control import anchored_emissions
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/control_renewal_v1/transition_screen")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/TRANSITION_RESEARCH_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    source = ROOT/"data/advanced_validation"
    raw, base = [pd.read_pickle(source/(name+".pkl")) for name in ("raw", "base")]
    receiver = pd.read_pickle(ROOT/"data/control_renewal_v1/ownership_screen/receiver_inputs.pkl")
    receiver = receiver[[c for c in receiver if "receiver_" in c]]
    inputs = micro_inputs(base).join(contact_chronology(base)).join(receiver)
    inputs.to_pickle(output/"inputs.pkl")
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    side, support, evidence = gap_sides(raw)[public], contact_support(raw)[public], control_evidence(raw)[public]
    splits = json.loads((source/"splits.json").read_text())
    rows, details, segments = [], [], []
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        for seed in plan["screen"]["seeds"]:
            print(protocol, seed, flush=True)
            p = np.load(source/protocol/f"ordered_xgboost_frame_seed{seed}_probabilities.npy")
            q = np.load(source/protocol/f"previous_ensemble_seed{seed}_probabilities.npy")
            fast = np.load(ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy")
            p = apply_context(p, side, np.zeros(len(public)), False)
            q = apply_context(q, side, np.zeros(len(public)), False)
            fast = apply_context(np.column_stack((fast, np.zeros(len(public)))), side, np.zeros(len(public)), False)[:, :2]
            fast /= fast.sum(axis=1, keepdims=True)
            hazard = np.full((len(public), 2), np.nan)
            emissions, baseline = {}, np.empty(len(public), int)
            for block, train, validation in folds:
                ix = validation-public[0]
                model = fit_hazard(inputs, labels, train, seed)
                model.save_model(folder/f"hazard_seed{seed}_block{block}.ubj")
                hazard[ix] = predict_hazard(model, inputs.iloc[validation])
                emissions[block] = anchored_emissions(p[ix], q[ix], fast[ix], support[ix], evidence[ix])
                baseline[ix] = decode(emissions[block])
            np.save(folder/f"hazard_seed{seed}_probabilities.npy", hazard)
            table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
            reference = pd.read_csv(ROOT/"data/control_renewal_v1/baseline_confirmation"/protocol/"oof_predictions.csv")
            np.testing.assert_array_equal(CLASSES[baseline], reference[f"anchored_rescue_seed{seed}"].to_numpy())
            for method, config in {"current": None, **plan["screen"]["methods"]}.items():
                name = f"{method}_seed{seed}"
                prediction = np.empty(len(public), dtype="<U1")
                for block, _, validation in folds:
                    ix = validation-public[0]
                    states = baseline[ix] if config is None else relocate_handoffs(
                        baseline[ix], emissions[block], hazard[ix], **config)
                    np.testing.assert_array_equal(states == 2, baseline[ix] == 2)
                    prediction[ix] = CLASSES[states]
                table[name] = prediction
                record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
                add_overlap_metrics(rows, segments, table, protocol, name)
            add_short_metrics(rows, segments)
            table.to_csv(folder/"oof_predictions.csv", index=False)
            pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
            pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
            pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    names = ("research_handoff_hazard.py", "handoff_hazard.py", "renewal_control.py", "control_chronology.py")
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT/"experiments"/name).read_bytes()).hexdigest() for name in names}), indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol", "model", "spellF1", "macroF1", "short_live_mean_best_iou"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
