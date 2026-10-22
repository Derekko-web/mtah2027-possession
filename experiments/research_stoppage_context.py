"""Public-only and excluded weak-supervision stoppage contexts, kept locally."""

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
from build_temporal_candidate import digest
from control_refinement import compose, contact_support, control_evidence, smooth
from evaluate_physics_context import apply_context
from games import require, source_frames
from renewal_control import anchored_emissions
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from stoppage_context import context_features, fit_context, load_ball, weak_samples
from temporal_candidate import build_inputs, gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES


def prepare_inputs(folder):
    folder.mkdir(parents=True, exist_ok=False)
    print("Building full-match label-free physical context", flush=True)
    raw, base, _ = build_inputs(source_frames())
    ball = load_ball(len(raw))
    physical, restart = context_features(raw, base, ball)
    prefix_physical, prefix_restart = context_features(raw.iloc[:18935], base.iloc[:18935], ball[:18935])
    pd.testing.assert_frame_equal(physical.iloc[11185:18685], prefix_physical.iloc[11185:18685])
    pd.testing.assert_frame_equal(restart.iloc[11185:18685], prefix_restart.iloc[11185:18685])
    reserved_path = ROOT/"data/continuation_20261021/annotation_freeze/windows.json"
    if digest(reserved_path) != "5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c":
        raise ValueError("Reserved window manifest changed")
    windows = json.loads(reserved_path.read_text())["windows"]
    weak_frames, weak_target = weak_samples(raw, physical, restart, windows)
    np.savez(folder/"weak_samples.npz", frames=weak_frames, target=weak_target)
    for name, value in (("physical", physical), ("restart", restart), ("raw", raw)):
        value.to_pickle(folder/(name+".pkl"))
    sources = ("stoppage_context.py", "stoppage_features.py", "research_stoppage_context.py")
    manifest = dict(source_sha256={name: digest(ROOT/"experiments"/name) for name in sources},
                    input_sha256={Path(p).name: digest(p) for p in require(["atd_meta", "atd_raw", "smart_tagging"])},
                    annotation_manifest_sha256=digest(reserved_path), public_prefix_parity=True,
                    weak_counts={"live": int((weak_target == 0).sum()), "dead": int((weak_target == 1).sum())},
                    excluded_public_and_reserved=True, physical_columns=list(physical), restart_columns=list(restart))
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Weak sample counts", manifest["weak_counts"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/control_renewal_v1/stoppage_screen")
    parser.add_argument("--inputs", type=Path, default=ROOT/"data/control_renewal_v1/stoppage_inputs")
    args = parser.parse_args()
    output, cache = args.output.resolve(), args.inputs.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/STOPPAGE_CONTEXT_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    if not cache.exists():
        prepare_inputs(cache)
    cache_manifest = json.loads((cache/"manifest.json").read_text())
    for name, expected in cache_manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Stoppage input source changed: "+name)
    raw, physical, restart = [pd.read_pickle(cache/(name+".pkl")) for name in ("raw", "physical", "restart")]
    weak = np.load(cache/"weak_samples.npz")
    weak_frames, weak_target = weak["frames"], weak["target"]
    if set(weak_target) != {0, 1}:
        raise ValueError("Insufficient weak classes")
    source = ROOT/"data/advanced_validation"
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    side, support, evidence = gap_sides(raw)[public], contact_support(raw)[public], control_evidence(raw)[public]
    visibility = raw.ball_seen.rolling(51, center=True, min_periods=1).mean().to_numpy()[public]
    seen = raw.ball_seen.to_numpy()[public] == 1
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
            emissions = {}
            for block, _, validation in folds:
                ix = validation-public[0]
                emissions[block] = anchored_emissions(p[ix], q[ix], fast[ix], support[ix], evidence[ix])
            table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
            for name, config in {"current": None, **plan["screen"]["models"]}.items():
                print(protocol, seed, name, flush=True)
                if config is not None:
                    inputs = restart if config["restart"] else physical
                    head = np.full(len(public), np.nan)
                    for block, train, validation in folds:
                        model = fit_context(inputs, labels, train, weak_frames, weak_target, config["weak_mass"], seed)
                        model.save_model(folder/f"{name}_seed{seed}_block{block}.ubj")
                        head[validation-public[0]] = model.predict_proba(inputs.iloc[validation])[:, 1]
                    np.save(folder/f"{name}_seed{seed}_probabilities.npy", head)
                for mode in (["reference"] if config is None else plan["screen"]["decoders"]):
                    method = f"current_seed{seed}" if config is None else f"{name}_{mode}_seed{seed}"
                    prediction = np.empty(len(public), dtype="<U1")
                    for block, _, validation in folds:
                        ix = validation-public[0]
                        emission = emissions[block]
                        if config is not None:
                            dead = smooth(head[ix, None], 25)[:, 0]
                            if mode == "replace_dead":
                                dead = np.where(seen[ix], dead, emission[:, 2])
                            else:
                                strong = (dead > .8) & (visibility[ix] >= .5) & (evidence[ix].max(axis=1) <= .2)
                                keep = np.zeros(len(ix), bool)
                                edges = np.flatnonzero(np.diff(np.r_[False, strong, False]))
                                for start, end in zip(edges[::2], edges[1::2]):
                                    if end-start >= 50:
                                        keep[start:end] = True
                                dead = np.maximum(emission[:, 2], np.where(keep, dead, 0))
                            total = emission[:, :2].sum(axis=1, keepdims=True)
                            owner = np.divide(emission[:, :2], total, out=np.full((len(ix), 2), .5), where=total > 0)
                            emission = compose(dead, owner)
                        prediction[ix] = CLASSES[decode(emission)]
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
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=digest(plan_path), cache_manifest_sha256=digest(cache/"manifest.json"),
        source_sha256={name: digest(ROOT/"experiments"/name) for name in ("research_stoppage_context.py", "stoppage_context.py")}), indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol", "model", "spellF1", "macroF1", "false_dead_frames", "dead_recall", "missed_stoppages"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
