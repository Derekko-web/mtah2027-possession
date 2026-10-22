"""Refit an ordered finalist using only the public labels.

The old data/candidate/ is preserved. Scores on this refitted CSV are training
scores, not validation. Use data/advanced_validation/ for held-out comparisons.
"""

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
from games import GAME, half_video_blocks, require, source_frames
from temporal_candidate import CAT_PARAMS, XGB_PARAMS, build_inputs, decode_blocks, fit_models, gap_sides, predict_models

SOURCES = ("temporal_candidate.py", "temporal_control.py", "build_temporal_candidate.py",
           "build_candidate.py", "rich_features.py", "ball_kinematics.py", "relative_motion.py",
           "train_factorized_baseline.py", "evaluate_physics_context.py", "transition_penalty.py",
           "xgboost_baseline.py", "requirements.txt", "requirements-research.txt")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Default: separate method-specific directory under data/")
    parser.add_argument("--method", choices=("ordered_xgboost_frame", "ordered_catboost_frame"),
                        default="ordered_xgboost_frame")
    args = parser.parse_args()
    default_folder = "candidate_temporal_xgb" if args.method == "ordered_xgboost_frame" else "candidate_temporal"
    output = (args.output or ROOT/"data"/default_folder).resolve()
    if output == (ROOT/"data/candidate").resolve():
        parser.error("Choose a separate output directory to preserve the original candidate")
    output.mkdir(parents=True, exist_ok=True)
    n_frames = source_frames()
    print("Building full-match ordered features", flush=True)
    raw, base, inputs = build_inputs(n_frames)
    public_path = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(public_path).label.to_numpy()
    train = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(train, np.arange(11185, 18685))
    padded_path = ROOT/"data/advanced_validation/ordered.pkl"
    if padded_path.exists():
        # Refitting/inference must use the exact same public feature values as CV.
        padded = pd.read_pickle(padded_path)
        pd.testing.assert_frame_equal(inputs.iloc[train], padded.iloc[train])
        print("Full-match public inputs exactly match padded validation inputs", flush=True)
    print("Fitting", args.method, "; seed 42", flush=True)
    models = fit_models(args.method, inputs, labels, train, seed=42)
    suffix = "cbm" if args.method == "ordered_catboost_frame" else "ubj"
    model_path = output/("frame_model."+suffix)
    models[0].save_model(str(model_path))
    p = np.empty((n_frames, 3), dtype=float)
    for start in range(0, n_frames, 20000):
        end = min(start+20000, n_frames)
        p[start:end] = predict_models(models, inputs.iloc[start:end])
    blocks = [(start, end+1) for _, start, end in half_video_blocks(GAME)]
    prediction = decode_blocks(p, gap_sides(raw), blocks)
    table = pd.DataFrame({"frame": np.arange(n_frames), "label": prediction})
    reason = pm.validate_submission(table, n_frames)
    if reason:
        raise ValueError(reason)
    candidate = output/"candidate.csv"
    table.to_csv(candidate, index=False)
    np.save(output/"probabilities.npy", p)
    parameters = CAT_PARAMS if args.method == "ordered_catboost_frame" else dict(
        XGB_PARAMS, max_depth=3, n_estimators=160)
    manifest = dict(
        model=args.method, description="Class-balanced frame model with ordered tracking observations and team symmetry",
        parameters=parameters, features=list(inputs.columns),
        public_training_frames=[int(train[0]), int(train[-1])], public_training_count=len(train),
        public_gt_sha256=digest(public_path), csv_sha256=digest(candidate), model_sha256=digest(model_path),
        source_sha256={str(Path("experiments")/name): digest(ROOT/"experiments"/name) for name in SOURCES},
        shared_source_sha256={name: digest(ROOT/name) for name in ("games.py", "starter/baseline.py", "starter/load_smart_tagging.py")},
        input_sha256={Path(path).name: digest(path) for path in require(["atd_meta", "atd_raw", "smart_tagging"])},
        output_class_counts=pd.Series(prediction).value_counts().to_dict(),
        training_classes=pd.Series(labels[train]).value_counts().to_dict(),
        half_blocks_inclusive=half_video_blocks(GAME),
        postprocessing=dict(probability_smoothing_half_window_frames=25, switch_penalty=3,
                            occlusion_max_frames=237, occlusion_player_speed_threshold_m_s=1.5,
                            preceding_control_window_frames=13, context_mix_weight=.75),
        validation_plan_sha256=digest(ROOT/"experiments/ADVANCED_VALIDATION_PLAN.json"),
        limitations=["Private full-match labels were not accessed; full-match accuracy is unknown.",
                     "Repeated adaptive selection used the same 17 public possessions.",
                     "Additional protocols and seeds are development stress checks, not independent tests.",
                     "Some short genuine possessions are missed and false dead-ball frames increase.",
                     "This is an offline model with future tracking observations, not a causal live predictor.",
                     "This local candidate has not been submitted or published."])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Saved", candidate, flush=True)
    print("CSV SHA256", manifest["csv_sha256"], flush=True)
    print("Class counts", manifest["output_class_counts"], flush=True)


if __name__ == "__main__":
    main()
