"""Prospective new-feature models against the frozen anchored default."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from boundary_refinement import refine_block
from build_candidate import PARAMS, objective_weights
from control_chronology import build_receiver_geometry, contact_chronology
from control_refinement import contact_support, control_evidence
from evaluate_physics_context import apply_context
from phase_control import phase_control
from renewal_control import anchored_emissions, short_proposals
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from rich_features import swap_teams
from temporal_candidate import gap_sides, predict_models
from transition_penalty import decode
from xgboost_baseline import CLASSES


def fit_renewal(inputs, labels, train, objective, seed):
    if set(labels[train]) != set(CLASSES):
        raise ValueError("Eligible training rows must contain only H/A/D and all three classes")
    target = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
    y, weight = objective_weights(target, train, objective == "runs")
    augmented = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
    model = XGBClassifier(**dict(PARAMS, max_depth=3, n_estimators=160, random_state=seed))
    model.fit(augmented, y, sample_weight=weight)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/control_renewal_v1/ownership_screen")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/RENEWAL_RESEARCH_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    source = ROOT/"data/advanced_validation"
    confirmed = ROOT/"data/control_boundary_v1/baseline_confirmation"
    raw, base, ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw", "base", "ordered")]
    chronology = contact_chronology(base)
    receiver = build_receiver_geometry(len(base))
    views = {"chronology": ordered.join(chronology), "receiver": ordered.join(chronology).join(receiver)}
    views["semantic"] = views["receiver"].join(phase_control(len(base)))
    for name, values in views.items():
        values.to_pickle(output/(name+"_inputs.pkl"))
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    side, support, evidence = gap_sides(raw)[public], contact_support(raw)[public], control_evidence(raw)[public]
    bracket = chronology[[prefix+"touch2_bracket_75" for prefix in ("h__", "a__")]].to_numpy()[public] >= .1
    splits_path = source/"splits.json"
    splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    rows, details, segments, hashes = [], [], [], {}
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        for seed in plan["screen"]["seeds"]:
            paths = [source/protocol/f"{name}_seed{seed}_probabilities.npy" for name in
                     ("ordered_xgboost_frame", "previous_ensemble")]
            paths.append(confirmed/protocol/f"fast_seed{seed}_probabilities.npy")
            arrays = []
            for path in paths:
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                value = np.load(path)
                if value.shape[1] == 2:
                    value = apply_context(np.column_stack((value, np.zeros(len(public)))), side, np.zeros(len(public)), False)[:, :2]
                    value /= value.sum(axis=1, keepdims=True)
                else:
                    value = apply_context(value, side, np.zeros(len(public)), False)
                arrays.append(value)
            current, previous, fast = arrays
            table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
            emissions = {}
            for block, _, validation in folds:
                ix = validation-public[0]
                emissions[block] = anchored_emissions(current[ix], previous[ix], fast[ix], support[ix], evidence[ix])
                np.testing.assert_array_equal(decode(emissions[block]), refine_block(
                    current[ix], previous[ix], fast[ix], support[ix], evidence[ix], "anchored_rescue"))
            configs = {"current": None, **plan["screen"]["models"]}
            for name, config in configs.items():
                print(protocol, seed, name, flush=True)
                if config is not None:
                    probability = np.full((len(public), 3), np.nan)
                    for block, train, validation in folds:
                        model = fit_renewal(views[config["features"]], labels, train, config["objective"], seed)
                        model.save_model(folder/f"{name}_seed{seed}_block{block}.ubj")
                        probability[validation-public[0]] = predict_models([model], views[config["features"]].iloc[validation])
                    np.save(folder/f"{name}_seed{seed}_probabilities.npy", probability)
                    probability = apply_context(probability, side, np.zeros(len(public)), False)
                for mode in (["reference"] if config is None else plan["screen"]["decoders"]):
                    method = f"{name}_{mode}_seed{seed}" if config is not None else f"current_seed{seed}"
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
                    if config is None:
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
    names = ("research_control_renewal.py", "control_chronology.py", "renewal_control.py", "boundary_refinement.py")
    manifest = dict(source_sha256={str(Path("experiments")/name): hashlib.sha256((ROOT/"experiments"/name).read_bytes()).hexdigest()
                                  for name in names}, input_sha256=hashes, plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    warning=plan["limitations"])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    columns = ["protocol", "model", "spellF1", "macroF1", "false_dead_frames", "dead_recall", "missed_stoppages",
               "short_live_mean_best_iou", "short_live_mostly_recovered"]
    print(pd.DataFrame(rows)[columns].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
