"""Train frozen contact-dynamics heads and preserve every OOF screen result."""

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
from control_refinement import contact_support, fit_control, predict_control
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from stoppage_consensus import decode_stoppage_consensus
from weak_ownership import ownership_prediction
from xgboost_baseline import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/control_dynamics_v1/screen")
    parser.add_argument("--inputs", type=Path, default=ROOT/"data/control_dynamics_v1/inputs")
    args = parser.parse_args()
    output, cache = args.output, args.inputs
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/CONTACT_DYNAMICS_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    manifest = json.loads((cache/"manifest.json").read_text())
    if manifest["plan_sha256"] != digest(plan_path):
        raise ValueError("Plan changed after physical diagnostic")
    for name, expected in manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Feature-cache source changed: "+name)
    for name, expected in manifest["cache_sha256"].items():
        if digest(cache/name) != expected:
            raise ValueError("Feature-cache bytes changed: "+name)
    source = ROOT/"data/advanced_validation"
    raw, ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw", "ordered")]
    geometry = {name:pd.read_pickle(cache/f"{name}_geometry.pkl") for name in ("raw", "median", "robust")}
    for team in ("h__", "a__"):
        np.testing.assert_array_equal(geometry["raw"][team+"clean_distance_1"], raw[team+"distance_1"])
    views = dict(ordered_control=ordered,
                 median_control=ordered.join(geometry["median"]),
                 robust_control=ordered.join(geometry["robust"]),
                 impulse_control=ordered.join(geometry["robust"]).join(pd.read_pickle(cache/"impulse.pkl")))
    supports = {"ordered_control": contact_support(raw)}
    for name in ("median_control", "robust_control", "impulse_control"):
        view = geometry["median" if name == "median_control" else "robust"]
        supports[name] = contact_support(view.rename(columns={prefix+"clean_distance_1":prefix+"distance_1" for prefix in ("h__", "a__")}))
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    splits_path = source/"splits.json"
    splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    reference = ROOT/"data/control_dynamics_v1/baseline_confirmation"
    rows, details, segments = [], [], []
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in
              (gt, splits_path, source/"raw.pkl", source/"ordered.pkl", cache/"manifest.json")}
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        blocks = [(int(v[0]), int(v[-1])+1) for _, _, v in folds]
        ref_path = reference/protocol/"oof_predictions.csv"
        hashes[str(ref_path.relative_to(ROOT))] = digest(ref_path)
        original = pd.read_csv(ref_path)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for seed in plan["screen"]["seeds"]:
            paths = [source/protocol/f"{name}_seed{seed}_probabilities.npy" for name in ("ordered_xgboost_frame", "previous_ensemble")]
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
            selected_states = np.array([np.flatnonzero(CLASSES == label)[0] for label in selected])
            method = f"current_seed{seed}"
            table[method] = selected
            record_predictions(output, protocol, method, public, labels, folds, selected, rows, details, segments)
            add_overlap_metrics(rows, segments, table, protocol, method)
            for name in plan["screen"]["models"]:
                x = views[name]
                probability = np.full((len(raw), 2), np.nan)
                for block, train, validation in folds:
                    print(protocol, seed, name, "block", block, flush=True)
                    model = fit_control(x, labels, train, seed)
                    model.save_model(folder/f"{name}_seed{seed}_block{block}.ubj")
                    probability[validation] = predict_control(model, x.iloc[validation])
                np.save(folder/f"{name}_seed{seed}_probabilities.npy", probability[public])
                for mode in plan["screen"]["decoders"]:
                    if mode == "replace_fast":
                        prediction = decode_stoppage_consensus(arrays[0], arrays[1], probability, arrays[3], raw, blocks)[public]
                    else:
                        prediction = selected.copy()
                        for _, _, validation in folds:
                            ix = validation-public[0]
                            prediction[ix] = CLASSES[ownership_prediction(selected_states[ix], probability[validation],
                                                   supports[name][validation], np.zeros((len(ix), 2)), mode)]
                    method = f"{name}_{mode}_seed{seed}"
                    table[method] = prediction
                    record_predictions(output, protocol, method, public, labels, folds, prediction, rows, details, segments)
                    add_overlap_metrics(rows, segments, table, protocol, method)
                add_short_metrics(rows, segments)
                table.to_csv(folder/"oof_predictions.csv", index=False)
                pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
                pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
                pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    names = ("research_contact_dynamics.py", "contact_dynamics.py", "prepare_contact_dynamics.py",
             "control_refinement.py", "stoppage_consensus.py", "weak_ownership.py")
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=digest(plan_path), input_sha256=hashes,
        source_sha256={name:digest(ROOT/"experiments"/name) for name in names},
        feature_columns={name:list(x) for name,x in views.items()}, reference_decoder_parity=True,
        raw_nearest_geometry_parity=True), indent=2)+"\n")
    columns = ["protocol", "model", "spellF1", "macroF1", "false_dead_frames", "dead_recall", "short_live_mean_best_iou", "short_live_matched_iou50"]
    print(pd.DataFrame(rows)[columns].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
