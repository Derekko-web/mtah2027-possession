"""Fixed physical release tests inside brief anchored ownership islands."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from control_refinement import contact_support, control_evidence
from evaluate_physics_context import apply_context
from island_endpoints import exclusive_contacts, trim_islands
from renewal_control import anchored_emissions
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES


def main():
    output = ROOT/"data/control_renewal_v1/island_screen"
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/ISLAND_ENDPOINT_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    source = ROOT/"data/advanced_validation"
    raw = pd.read_pickle(source/"raw.pkl")
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    side, support, evidence = gap_sides(raw)[public], contact_support(raw)[public], control_evidence(raw)[public]
    contacts = exclusive_contacts(raw)[public]
    splits = json.loads((source/"splits.json").read_text())
    rows, details, segments = [], [], []
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        for seed in plan["screen"]["seeds"]:
            p = apply_context(np.load(source/protocol/f"ordered_xgboost_frame_seed{seed}_probabilities.npy"), side, np.zeros(len(public)), False)
            q = apply_context(np.load(source/protocol/f"previous_ensemble_seed{seed}_probabilities.npy"), side, np.zeros(len(public)), False)
            fast = np.load(ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy")
            fast = apply_context(np.column_stack((fast, np.zeros(len(public)))), side, np.zeros(len(public)), False)[:, :2]
            fast /= fast.sum(axis=1, keepdims=True)
            table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
            emissions = {}
            for block, _, validation in folds:
                ix = validation-public[0]
                emissions[block] = anchored_emissions(p[ix], q[ix], fast[ix], support[ix], evidence[ix])
            for name, config in {"current": None, **plan["screen"]["methods"]}.items():
                prediction = np.empty(len(public), dtype="<U1")
                for block, _, validation in folds:
                    ix = validation-public[0]
                    emission = emissions[block] if config is None else trim_islands(emissions[block], p[ix], contacts[ix], **config)
                    prediction[ix] = CLASSES[decode(emission)]
                method = f"{name}_seed{seed}"
                table[method] = prediction
                record_predictions(output, protocol, method, public, labels, folds, prediction, rows, details, segments)
                add_overlap_metrics(rows, segments, table, protocol, method)
            add_short_metrics(rows, segments)
            table.to_csv(folder/"oof_predictions.csv", index=False)
            pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
            pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
            pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=digest(plan_path), source_sha256={
        name: digest(ROOT/"experiments"/name) for name in ("research_island_endpoints.py", "island_endpoints.py")}), indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol", "model", "spellF1", "macroF1", "false_dead_frames", "dead_recall", "short_live_mean_best_iou"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
