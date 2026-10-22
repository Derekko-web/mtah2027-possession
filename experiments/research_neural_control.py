"""Evaluate frozen neural heads against anchored rescue without held-out fitting."""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from boundary_refinement import refine_block
from build_temporal_candidate import digest
from control_chronology import contact_chronology
from control_refinement import compose, contact_support, control_evidence
from evaluate_physics_context import apply_context
from neural_control import fit_neural, predict_neural
from renewal_control import anchored_emissions, short_proposals
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES


def main():
    output = ROOT/"data/control_renewal_v1/neural_screen"
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/NEURAL_CONTROL_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    source = ROOT/"data/advanced_validation"
    raw, base = [pd.read_pickle(source/(name+".pkl")) for name in ("raw", "base")]
    views = {name: pd.read_pickle(ROOT/"data/control_renewal_v1/ownership_screen"/(name+"_inputs.pkl"))
             for name in ("receiver", "semantic")}
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    side, support, evidence = gap_sides(raw)[public], contact_support(raw)[public], control_evidence(raw)[public]
    chronology = contact_chronology(base)
    bracket = chronology[[prefix+"touch2_bracket_75" for prefix in ("h__", "a__")]].to_numpy()[public] >= .1
    splits = json.loads((source/"splits.json").read_text())
    rows, details, segments = [], [], []
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        for seed in plan["screen"]["seeds"]:
            raw_current = np.load(source/protocol/f"ordered_xgboost_frame_seed{seed}_probabilities.npy")
            current = apply_context(raw_current, side, np.zeros(len(public)), False)
            previous = apply_context(np.load(source/protocol/f"previous_ensemble_seed{seed}_probabilities.npy"), side, np.zeros(len(public)), False)
            fast = np.load(ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy")
            fast = apply_context(np.column_stack((fast, np.zeros(len(public)))), side, np.zeros(len(public)), False)[:, :2]
            fast /= fast.sum(axis=1, keepdims=True)
            emissions = {block: anchored_emissions(current[v-public[0]], previous[v-public[0]], fast[v-public[0]],
                         support[v-public[0]], evidence[v-public[0]]) for block, _, v in folds}
            table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
            for name, config in {"current": None, **plan["screen"]["models"]}.items():
                if config is not None:
                    width = 2 if config["live_only"] else 3
                    probability = np.full((len(public), width), np.nan)
                    for block, train, validation in folds:
                        print(protocol, seed, name, "block", block, flush=True)
                        model = fit_neural(views[config["features"]], labels, train, config["live_only"], seed)
                        joblib.dump(model, folder/f"{name}_seed{seed}_block{block}.joblib")
                        probability[validation-public[0]] = predict_neural(model, views[config["features"]].iloc[validation])
                    np.save(folder/f"{name}_seed{seed}_probabilities.npy", probability)
                    if config["live_only"]:
                        probability = compose(raw_current[:, 2], probability)
                    probability = apply_context(probability, side, np.zeros(len(public)), False)
                for mode in (["reference"] if config is None else plan["screen"]["decoders"]):
                    method = f"current_seed{seed}" if config is None else f"{name}_{mode}_seed{seed}"
                    prediction = np.empty(len(public), dtype="<U1")
                    for block, _, validation in folds:
                        ix = validation-public[0]
                        if config is None:
                            states = decode(emissions[block])
                        elif mode == "replace_current":
                            states = refine_block(probability[ix], previous[ix], fast[ix], support[ix], evidence[ix], "anchored_rescue")
                        else:
                            states = decode(short_proposals(emissions[block], probability[ix], support[ix], bracket[ix]))
                        prediction[ix] = CLASSES[states]
                    table[method] = prediction
                    record_predictions(output, protocol, method, public, labels, folds, prediction, rows, details, segments)
                    add_overlap_metrics(rows, segments, table, protocol, method)
                add_short_metrics(rows, segments)
                table.to_csv(folder/"oof_predictions.csv", index=False)
                pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
                pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
                pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=digest(plan_path), source_sha256={
        name: digest(ROOT/"experiments"/name) for name in ("research_neural_control.py", "neural_control.py")}), indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol", "model", "spellF1", "macroF1", "false_dead_frames", "dead_recall"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
