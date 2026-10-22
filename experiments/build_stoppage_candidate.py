"""Reproduce the confirmed six-model stoppage candidate from local inputs.

Requires the original anchored candidate plus the deterministic stoppage-context
cache and completed confirmation. Every output uses a fresh directory. No upload.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from games import GAME, half_video_blocks, require, source_frames
from stoppage_consensus import decode_stoppage_consensus
from stoppage_context import fit_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/candidate_stoppage_consensus")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose a fresh directory to preserve candidates")
    reference = ROOT/"data/candidate_anchored_control"
    prior = json.loads((reference/"manifest.json").read_text())
    expected_csv = "973e1670e9e1240ecdde3bbabebf1e803f0d3b6259aad1b9b3223c413d26dcfd"
    if prior["csv_sha256"] != expected_csv or digest(reference/"candidate.csv") != expected_csv:
        raise ValueError("Unexpected anchored reference")
    confirmation = ROOT/"data/control_renewal_v1/recovery_confirmation"
    checks_path = confirmation/"promotion_checks.json"
    checks = json.loads(checks_path.read_text())
    if checks["selected"] != "stoppage_consensus" or not checks["passed"]:
        raise ValueError("Stoppage recovery has not passed its frozen objective")
    plan_path = ROOT/"experiments/STOPPAGE_RECOVERY_CONFIRMATION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    if (confirmation/"plan.json").read_bytes() != plan_path.read_bytes():
        raise ValueError("Confirmation plan changed")
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Frozen recovery source changed")
    for group in ("source_sha256", "reference_source_sha256"):
        for path, expected in prior[group].items():
            if digest(ROOT/path) != expected:
                raise ValueError("Anchored source changed: "+path)
    inputs = {Path(p).name: digest(p) for p in require(["atd_meta", "atd_raw", "smart_tagging"])}
    if inputs != prior["input_sha256"]:
        raise ValueError("Competition inputs changed")
    cache = ROOT/"data/control_renewal_v1/stoppage_inputs"
    confirmed = json.loads((confirmation/"manifest.json").read_text())
    for name, expected in confirmed["cache_sha256"].items():
        if digest(cache/name) != expected:
            raise ValueError("Confirmed feature/weak cache changed: "+name)
    cache_manifest = json.loads((cache/"manifest.json").read_text())
    if inputs != cache_manifest["input_sha256"]:
        raise ValueError("Cached input provenance differs")
    for name, expected in cache_manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Cached feature-generation source changed: "+name)
    annotation = ROOT/"data/continuation_20261021/annotation_freeze/windows.json"
    if digest(annotation) != cache_manifest["annotation_manifest_sha256"]:
        raise ValueError("Reserved windows changed")
    gt_path = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    if digest(gt_path) != prior["public_gt_sha256"] or digest(gt_path) != confirmed["public_gt_sha256"]:
        raise ValueError("Public labels changed")
    labels = pm.read_label_csv(gt_path).label.to_numpy()
    train = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(train, np.arange(11185, 18685))
    raw, physical, restart = [pd.read_pickle(cache/(name+".pkl")) for name in ("raw", "physical", "restart")]
    if len(raw) != source_frames() or not (len(physical) == len(restart) == len(raw)):
        raise ValueError("Full-match feature length mismatch")
    weak = np.load(cache/"weak_samples.npz")
    weak_frames, weak_target = weak["frames"], weak["target"]
    if np.any((weak_frames >= 11185-250) & (weak_frames <= 18684+250)):
        raise ValueError("Weak labels enter public exclusion")
    for window in json.loads(annotation.read_text())["windows"]:
        if np.any((weak_frames >= window["start"]-250) & (weak_frames <= window["end"]+250)):
            raise ValueError("Weak labels enter reserved windows")
    for group in ("reference_model_sha256", "probability_sha256"):
        for name, expected in prior[group].items():
            if digest(reference/name) != expected:
                raise ValueError("Retained artifact changed: "+name)
    output.mkdir(parents=True)
    heads = np.empty((len(raw), 2))
    print("Fitting two fixed D heads on public labels plus excluded weak samples", flush=True)
    for index, (name, features) in enumerate((("weak_physical_model", physical), ("weak_restart_model", restart))):
        model = fit_context(features, labels, train, weak_frames, weak_target, .5, 42)
        path = output/(name+".ubj")
        model.save_model(path)
        reloaded = XGBClassifier()
        reloaded.load_model(path)
        for start in range(0, len(raw), 20000):
            end = min(len(raw), start+20000)
            heads[start:end, index] = model.predict_proba(features.iloc[start:end])[:, 1]
            np.testing.assert_array_equal(heads[start:end, index], reloaded.predict_proba(features.iloc[start:end])[:, 1])
    arrays = [np.load(reference/(name+"_probabilities.npy")) for name in ("current", "previous", "fast")]
    blocks = [(start, end+1) for _, start, end in half_video_blocks(GAME)]
    prediction = decode_stoppage_consensus(*arrays, heads, raw, blocks)
    anchored = decode_stoppage_consensus(*arrays, heads, raw, blocks, False)
    np.testing.assert_array_equal(anchored, pd.read_csv(reference/"candidate.csv").label.to_numpy())
    table = pd.DataFrame({"frame": np.arange(len(raw)), "label": prediction})
    reason = pm.validate_submission(table, source_frames())
    if reason:
        raise ValueError(reason)
    table.to_csv(output/"candidate.csv", index=False)
    np.save(output/"heads_probabilities.npy", heads)
    for name in list(prior["reference_model_sha256"])+list(prior["probability_sha256"]):
        shutil.copy2(reference/name, output/name)
    (output/"reference_manifest.json").write_bytes((reference/"manifest.json").read_bytes())
    source_names = ("build_stoppage_candidate.py", "confirm_stoppage_recovery.py", "finalize_stoppage_recovery.py",
                    "stoppage_consensus.py", "stoppage_context.py", "stoppage_hysteresis.py", "renewal_control.py",
                    "stoppage_features.py", "research_stoppage_context.py", "STOPPAGE_CONTEXT_PLAN.json",
                    "STOPPAGE_HYSTERESIS_PLAN.json", "STOPPAGE_RECOVERY_CONFIRMATION_PLAN.json")
    models = sorted(output.glob("*.ubj"))
    probabilities = sorted(output.glob("*.npy"))
    manifest = dict(model="Anchored ownership plus two-view weak-supervised stoppage consensus", seed=42,
                    reference_csv_sha256=expected_csv, csv_sha256=digest(output/"candidate.csv"),
                    source_sha256={str(Path("experiments")/name): digest(ROOT/"experiments"/name) for name in source_names},
                    reference_source_sha256={**prior["source_sha256"], **prior["reference_source_sha256"]},
                    model_sha256={p.name: digest(p) for p in models}, probability_sha256={p.name: digest(p) for p in probabilities},
                    input_sha256=inputs, public_gt_sha256=digest(gt_path), cache_sha256=confirmed["cache_sha256"],
                    confirmation_plan_sha256=digest(plan_path), promotion_checks_sha256=digest(checks_path),
                    annotation_manifest_sha256=digest(annotation), public_training_frames=[11185, 18684],
                    public_training_count=7500, weak_training_counts=cache_manifest["weak_counts"], weak_mass=.5,
                    new_models_reload_exactly=True, reference_decode_identical=True, retained_artifacts_unchanged=True,
                    output_class_counts=pd.Series(prediction).value_counts().to_dict(),
                    changed_frames=int((prediction != anchored).sum()),
                    original_substantial_primary_criterion_met=False,
                    limitations=["Selected for separately frozen adaptive stoppage-recovery objective, not the initial primary-gain criterion.",
                                 "Same repeatedly inspected public labels; no independent/private full-match accuracy estimate.",
                                 "Weak labels from automatic tags and motion are fallible training proxies.",
                                 "Short misses and brief-run endpoint errors remain. Nothing submitted or uploaded."])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Saved", output/"candidate.csv", flush=True)
    print("CSV SHA256", manifest["csv_sha256"], flush=True)
    print("Changed frames", manifest["changed_frames"], flush=True)
    print("Class counts", manifest["output_class_counts"], flush=True)


if __name__ == "__main__":
    main()
