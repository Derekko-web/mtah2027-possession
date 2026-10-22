"""Build anchored rescue from the reproducible frozen four-model reference.

The reference's saved models must reproduce all three saved probability arrays
exactly. No models are refitted or selected here. Earlier candidates are read-only.
See BOUNDARY_REFINEMENT_REPORT.md for cache preparation and confirmation commands.
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
from boundary_refinement import decode_refinement
from build_temporal_candidate import digest
from control_refinement import fast_columns, predict_control
from games import GAME, half_video_blocks, require, source_frames
from temporal_candidate import build_inputs, predict_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT/"data/candidate_control_consensus")
    parser.add_argument("--output", type=Path, default=ROOT/"data/candidate_anchored_control")
    args = parser.parse_args()
    reference, output = args.reference.resolve(), args.output.resolve()
    # Exclusive creation also protects current and historical candidates.
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    plan_path = ROOT/"experiments/BOUNDARY_CONFIRMATION_PLAN.json"
    checks_path = ROOT/"data/control_boundary_v1/confirmation/promotion_checks.json"
    plan, checks = json.loads(plan_path.read_text()), json.loads(checks_path.read_text())
    if checks["selected"] != "anchored_rescue" or not checks["methods"]["anchored_rescue"]["passed"]:
        raise ValueError("Anchored rescue did not clear the frozen checks")
    for path, expected in plan["source_sha256"].items():
        if digest(ROOT/path) != expected:
            raise ValueError("Frozen implementation changed: "+path)
    prior = json.loads((reference/"manifest.json").read_text())
    expected_csv = "19dd8ecac4fe0c3fcbdc11d18d41cd9a38d1cf995a0defb05ff816b6057a64f1"
    if prior["csv_sha256"] != expected_csv or digest(reference/"candidate.csv") != expected_csv:
        raise ValueError("Unexpected reference candidate")
    for group in ("source_sha256", "shared_source_sha256"):
        for path, expected in prior[group].items():
            if digest(ROOT/path) != expected:
                raise ValueError("Reference source changed: "+path)
    actual_inputs = {Path(path).name: digest(path) for path in require(["atd_meta", "atd_raw", "smart_tagging"])}
    if actual_inputs != prior["input_sha256"]:
        raise ValueError("Competition inputs differ from the frozen reference")
    if digest(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv") != prior["public_gt_sha256"]:
        raise ValueError("Public labels differ from the frozen reference")
    models = []
    for name in ("current_model.ubj", "previous_frame.ubj", "previous_segment.ubj", "fast_model.ubj"):
        if digest(reference/name) != prior["model_sha256"][name]:
            raise ValueError("Reference model changed: "+name)
        model = XGBClassifier()
        model.load_model(reference/name)
        models.append(model)
    output.mkdir(parents=True)
    print("Building full-match features and checking saved-model predictions", flush=True)
    raw, base, ordered = build_inputs(source_frames())
    confirmed = ROOT/"data/control_boundary_v1/baseline_confirmation"
    for name, values in (("base", base), ("ordered", ordered)):
        pd.testing.assert_frame_equal(values.iloc[11185:18685], pd.read_pickle(confirmed/(name+".pkl")).iloc[11185:18685])
    # Raw includes one-hot automatic event vocabularies, which can legitimately
    # grow outside the public window. Only these columns enter shared inference.
    required_raw = ["ball_seen", "ball_displacement_1s", "h__distance_1", "a__distance_1",
                    "h__control_share", "a__control_share", "h__speed_median", "a__speed_median"]
    pd.testing.assert_frame_equal(raw[required_raw].iloc[11185:18685],
                                  pd.read_pickle(confirmed/"raw.pkl")[required_raw].iloc[11185:18685])
    arrays = [np.load(reference/(name+"_probabilities.npy")) for name in ("current", "previous", "fast")]
    fast_inputs = base[fast_columns(base)]
    for start in range(0, len(raw), 20000):
        end = min(len(raw), start+20000)
        predicted = (predict_models(models[:1], ordered.iloc[start:end]),
                     predict_models(models[1:3], base.iloc[start:end]),
                     predict_control(models[3], fast_inputs.iloc[start:end]))
        for actual, expected in zip(predicted, arrays):
            np.testing.assert_array_equal(actual, expected[start:end])
    blocks = [(start, end+1) for _, start, end in half_video_blocks(GAME)]
    prediction = decode_refinement(*arrays, raw, blocks)
    table = pd.DataFrame({"frame": np.arange(len(raw)), "label": prediction})
    reason = pm.validate_submission(table, source_frames())
    if reason:
        raise ValueError(reason)
    table.to_csv(output/"candidate.csv", index=False)
    for name in prior["model_sha256"]:
        shutil.copy2(reference/name, output/name)
    probability_hashes = {}
    for name in ("current", "previous", "fast"):
        filename = name+"_probabilities.npy"
        shutil.copy2(reference/filename, output/filename)
        probability_hashes[filename] = digest(output/filename)
    (output/"reference_manifest.json").write_bytes((reference/"manifest.json").read_bytes())
    sources = ("build_anchored_candidate.py", "boundary_refinement.py", "research_boundary_refinement.py",
               "finalize_boundary_refinement.py", "BOUNDARY_REFINEMENT_PLAN.json", "BOUNDARY_CONFIRMATION_PLAN.json")
    manifest = dict(model="Anchored short-live consensus rescue", seed=42, method="anchored_rescue",
                    reference_csv_sha256=expected_csv, csv_sha256=digest(output/"candidate.csv"),
                    reference_model_sha256=prior["model_sha256"], probability_sha256=probability_hashes,
                    reference_source_sha256={**prior["source_sha256"], **prior["shared_source_sha256"]},
                    source_sha256={str(Path("experiments")/name): digest(ROOT/"experiments"/name) for name in sources},
                    confirmation_plan_sha256=digest(plan_path), promotion_checks_sha256=digest(checks_path),
                    input_sha256=actual_inputs, public_gt_sha256=prior["public_gt_sha256"],
                    public_training_frames=prior["public_training_frames"], public_training_count=7500,
                    saved_model_probability_parity=True, padded_feature_parity=True,
                    output_class_counts=pd.Series(prediction).value_counts().to_dict(),
                    changed_frames=int((prediction != pd.read_csv(reference/"candidate.csv").label.to_numpy()).sum()),
                    limitations=["Adaptive within-match public development; no independent or private accuracy estimate.",
                                 "Two short possessions and visible-ball stoppage errors remain.",
                                 "All boundaries are algorithmic; offline future observations and separate half decoding.",
                                 "Nothing submitted or uploaded."])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Saved", output/"candidate.csv", flush=True)
    print("CSV SHA256", manifest["csv_sha256"], flush=True)
    print("Changed frames", manifest["changed_frames"], flush=True)
    print("Class counts", manifest["output_class_counts"], flush=True)


if __name__ == "__main__":
    main()
