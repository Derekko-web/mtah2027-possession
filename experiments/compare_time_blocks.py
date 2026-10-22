"""Compare feature groups on fixed expanding chronological development splits.

Run: .venv/bin/python experiments/compare_time_blocks.py
The final three public minutes are separate one-minute validation blocks. Each
model trains on all earlier public frames except a ten-second gap. All previous
experiment outputs are preserved. These are development checks, not final tests.
Use --buffered-blocks to hold out each of five minutes and train on the others,
excluding a ten-second buffer on both sides of validation.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from games import GAME, GAMES
from inspect_player_motion import MOVING, SPEED, read_motion
from xgboost_baseline import (
    CLASSES, FEATURES, GAP_SECONDS, PARAMETERS, SMOOTH_HALF_WINDOW, SPEED_FEATURE,
    probability_smoothing, read_distances,
)

FEATURE_GROUPS = {
    "distances": FEATURES,
    "plus_ball_speed": FEATURES + [SPEED_FEATURE],
    "plus_player_motion": FEATURES + [SPEED_FEATURE, SPEED, MOVING],
}
VALIDATION_START_SECONDS = (120, 180, 240)  # Fixed before fitting any fold.
VALIDATION_SECONDS = 60


def time_blocks(public_start, public_end, fps):
    blocks = []
    for minute_start in VALIDATION_START_SECONDS:
        val_start = public_start + minute_start * fps
        val_end = val_start + VALIDATION_SECONDS * fps
        train_end = val_start - GAP_SECONDS * fps
        train = np.arange(public_start, train_end)
        validation = np.arange(val_start, val_end)
        if not len(train) or val_end - 1 > public_end:
            raise ValueError("Insufficient public labels for the fixed splits")
        if validation[0] - train[-1] - 1 != GAP_SECONDS * fps:
            raise ValueError("Incorrect separation between train and validation")
        if train[-1] >= validation[0] - fps:
            raise ValueError("Training must end before the validation feature lookback")
        blocks.append((minute_start // 60 + 1, train, validation))
    flat = np.concatenate([validation for _, _, validation in blocks])
    if len(np.unique(flat)) != len(flat):
        raise ValueError("Validation blocks overlap")
    return blocks


def buffered_blocks(public_start, public_end, fps):
    """Each minute held out once, with training outside two ten-second buffers."""
    public_frames = np.arange(public_start, public_end + 1)
    block_size = VALIDATION_SECONDS * fps
    if len(public_frames) != 5 * block_size:
        raise ValueError("Expected exactly five public minutes")
    gap = GAP_SECONDS * fps
    blocks = []
    for minute in range(1, 6):
        start = public_start + (minute - 1) * block_size
        end = start + block_size  # exclusive
        validation = np.arange(start, end)
        train = public_frames[(public_frames < start - gap) | (public_frames >= end + gap)]
        # The longest feature lookback is one second. Later training rows must
        # not read validation observations through their causal input windows.
        if not len(train) or not np.all((train < start - gap) | (train >= end + gap)):
            raise ValueError("Training violates the validation buffers")
        if gap < fps:
            raise ValueError("Buffer must cover the feature lookback")
        blocks.append((minute, train, validation))
    validation_frames = np.concatenate([validation for _, _, validation in blocks])
    np.testing.assert_array_equal(validation_frames, public_frames)
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffered-blocks", action="store_true",
                        help="hold out each public minute; train outside buffers on both sides")
    args = parser.parse_args()
    mode = "buffered_block_comparison" if args.buffered_blocks else "time_block_comparison"
    output = ROOT / "data" / mode
    output.mkdir(exist_ok=True)
    gt = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    labels = gt.label.to_numpy()
    public = np.flatnonzero(labels != "X")
    if not np.array_equal(public, np.arange(public[0], public[-1] + 1)):
        raise ValueError("Expected one contiguous public window")
    fps = GAMES[GAME]["fps"]
    make_blocks = buffered_blocks if args.buffered_blocks else time_blocks
    blocks = make_blocks(int(public[0]), int(public[-1]), fps)
    print("Extracting fixed features; no parameter search...", flush=True)
    features = read_distances(with_ball_speed=True)
    motion = read_motion(int(public[-1])).set_index("frame")
    features = features.join(motion[[SPEED, MOVING]], validate="one_to_one")
    if len(features) != len(gt) or pm.validate_submission(gt, len(features)):
        raise ValueError("Feature/ground-truth alignment failed")
    nearest = np.where(features[FEATURES].fillna(np.inf).to_numpy().argmin(axis=1) == 0,
                       "H", "A")
    nearest[features[FEATURES].isna().all(axis=1)] = "D"
    reference_raw = pm.read_label_csv(ROOT / "evaluation/results/baseline_raw.csv")
    np.testing.assert_array_equal(nearest[public], reference_raw.label.to_numpy()[public])
    references = {
        f"organiser_{variant}": pm.read_label_csv(
            ROOT / f"evaluation/results/baseline_{variant}.csv",
        ).label.to_numpy() for variant in ("smooth50", "smooth125")
    }
    scores, manifests = [], []
    for minute, train, validation in blocks:
        print(f"Validating public minute {minute}; "
              f"training on {len(train) / fps:.0f} seconds outside validation buffers", flush=True)
        fold_output = output / f"minute_{minute}"
        fold_output.mkdir(exist_ok=True)
        targets = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
        if not np.array_equal(np.unique(targets), np.arange(3)):
            raise ValueError("A training block lacks one of H/A/D")
        truth = pm.label_codes(labels[validation])
        details = features.iloc[validation].copy()
        details.insert(0, "frame", validation)
        details["ground_truth"] = labels[validation]
        predictions = {}
        for name, columns in FEATURE_GROUPS.items():
            model = XGBClassifier(**PARAMETERS)
            # Validation is never passed to fitting or early stopping.
            model.fit(features.iloc[train][columns], targets)
            probability = model.predict_proba(features.iloc[validation][columns])
            if not np.isfinite(probability).all():
                raise ValueError("Non-finite probabilities")
            predictions[f"{name}_raw"] = CLASSES[probability.argmax(axis=1)]
            predictions[f"{name}_smoothed"] = CLASSES[
                probability_smoothing(probability).argmax(axis=1)
            ]
            for index, label in enumerate(CLASSES):
                details[f"{name}_probability_{label}"] = probability[:, index]
            model.save_model(fold_output / f"{name}.ubj")
        predictions.update({name: prediction[validation]
                            for name, prediction in references.items()})
        for name, prediction in predictions.items():
            details[name] = prediction
            result = pm.evaluate(truth, pm.label_codes(prediction), bootstrap_n=1000)
            scores.append(dict(
                validation_minute=minute, model=name, macroF1=result["macro_f1"],
                spellF1=result["spell_f1_mean"], dead_ball_F1=result["per_class"]["D"]["f1"],
                predicted_possessions=result["n_pred_spells"],
                true_possessions=result["n_gt_spells"],
                spellF1_low=result["spell_bootstrap_low"],
                spellF1_high=result["spell_bootstrap_high"],
            ))
        details.to_csv(fold_output / "validation_predictions.csv", index=False)
        manifests.append(dict(
            validation_minute=minute,
            training_frame_blocks_inclusive=[
                [int(block[0]), int(block[-1])]
                for block in np.split(train, np.flatnonzero(np.diff(train) != 1) + 1)
            ],
            validation_frames_inclusive=[int(validation[0]), int(validation[-1])],
            training_class_counts=pd.Series(labels[train]).value_counts().to_dict(),
            validation_class_counts=pd.Series(labels[validation]).value_counts().to_dict(),
        ))
    scores = pd.DataFrame(scores)
    scores.to_csv(output / "validation_scores.csv", index=False)
    summary = scores.pivot(index="model", columns="validation_minute", values="spellF1")
    mean_column = "mean_of_five_minutes" if args.buffered_blocks else "mean_of_three_minutes"
    summary[mean_column] = summary.mean(axis=1)
    summary.to_csv(output / "spellF1_comparison.csv")
    caveats = [
        "These small same-match development blocks have already informed feature exploration.",
        "Training sets overlap; folds are not independent replicates.",
        "Minute boundaries truncate possessions; mean spellF1 is not a full-match score.",
        "Organiser reference files use full-match smoothing context; model smoothing "
        "is clipped to each validation block.",
        "Minute 3 contains no D ground-truth frames; macroF1 still averages H/A/D.",
    ]
    if args.buffered_blocks:
        caveats.append("Offline blocked validation trains on earlier and later labelled public "
                       "frames outside buffers; this does not test forecasting or a new match.")
    else:
        caveats.append("Expanding chronological splits use earlier training labels only; "
                       "earlier validation frames enter later training sets.")
    manifest = dict(
        mode=mode,
        feature_groups=FEATURE_GROUPS, model_parameters=PARAMETERS, splits=manifests,
        gap_seconds=GAP_SECONDS, smoothing_half_window_frames=SMOOTH_HALF_WINDOW,
        caveats=caveats,
    )
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nspellF1 per validation minute (all model settings fixed):")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nResults saved to", output)


if __name__ == "__main__":
    main()
