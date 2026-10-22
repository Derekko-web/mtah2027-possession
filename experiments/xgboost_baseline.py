"""Small XGBoost experiments, evaluated only on held-out public frames.

Run from the project root: .venv/bin/python experiments/xgboost_baseline.py
Outputs live under ignored data/xgboost_baseline/; these are not submissions.
Add --with-ball-speed to save the three-feature experiment separately.
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
from games import GAME, GAMES, require, source_frames
from starter.baseline import team_of_each_track

FEATURES = ["nearest_home_m", "nearest_away_m"]
SPEED_FEATURE = "ball_speed_past_second_m_s"
CLASSES = np.array(["H", "A", "D"])
# Chosen before running the experiment; no hyperparameter search on the holdout.
TRAIN_SECONDS = 180
GAP_SECONDS = 10
SMOOTH_HALF_WINDOW = 25  # +/- 1 second, 51 frames including the current frame.
PARAMETERS = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    tree_method="hist", n_estimators=100, max_depth=2, learning_rate=0.05,
    min_child_weight=10, reg_lambda=5, random_state=42, n_jobs=2,
)


def recent_ball_speed(ball_xy_m, fps):
    """Past-second median of speeds from consecutive observed ball positions.

    Never interpolate across missing frames. Require more than half of the
    one-second window to contain measured steps, and leave other outputs NaN.
    This is apparent tracked speed, including measurement noise, not true speed.
    """
    speed = np.r_[np.nan, np.linalg.norm(np.diff(ball_xy_m, axis=0), axis=1) * fps]
    speed[~np.isfinite(speed)] = np.nan
    return pd.Series(speed).rolling(fps, min_periods=fps // 2 + 1).median().to_numpy()


def read_distances(with_ball_speed=False):
    """Stream tracking into two distances in metres, one row per video frame.

    Distances use current-frame inputs only, never labels. A missing ball or
    absent team leaves its distance NaN, which XGBoost can handle directly.
    Optional speed uses a trailing one-second window of ball positions.
    """
    metadata, tracking = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(metadata)
    n_frames = source_frames()
    features = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_xy_m = np.full((n_frames, 2), np.nan)
    scale = np.array(GAMES[GAME]["pitch"])
    count = 0
    with open(tracking) as handle:
        for count, line in enumerate(handle, start=1):
            frame, players_raw, ball_raw = line.strip().split(":")
            if int(frame) != count or count > n_frames:
                raise ValueError("Tracking frames must be contiguous and start at 1")
            ball = np.fromstring(ball_raw, sep=",")
            players = np.fromstring(players_raw.replace(";", ","), sep=",")
            if ball.size != 3 or players.size != len(teams) * 3:
                raise ValueError(f"Unexpected tracking layout at frame {frame}")
            if not np.isfinite(ball[:2]).all():
                continue
            ball_xy_m[count - 1] = ball[:2] * scale
            positions = players.reshape(-1, 3)[:, :2]
            distances = np.linalg.norm((positions - ball[:2]) * scale, axis=1)
            for team in (0, 1):
                usable = distances[(teams == team) & np.isfinite(distances)]
                if usable.size:
                    features[count - 1, team] = usable.min()
    if count != n_frames:
        raise ValueError(f"Expected {n_frames} tracking records, found {count}")
    table = pd.DataFrame(features, columns=FEATURES)
    if with_ball_speed:
        table[SPEED_FEATURE] = recent_ball_speed(ball_xy_m, GAMES[GAME]["fps"])
    return table


def probability_smoothing(probabilities):
    """Average class probabilities within this block only, then pick a class.

    Centred windows use future input frames, allowed for this offline task.
    Calling separately on validation prevents any window crossing the split.
    Unlike the organiser's smoothing, D participates in this three-class vote.
    """
    return pd.DataFrame(probabilities).rolling(
        2 * SMOOTH_HALF_WINDOW + 1, center=True, min_periods=1,
    ).mean().to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-ball-speed", action="store_true",
                        help="add a past-one-second ball-speed feature; save separately")
    args = parser.parse_args()
    experiment = "xgboost_with_ball_speed" if args.with_ball_speed else "xgboost_baseline"
    output = ROOT / "data" / experiment
    output.mkdir(exist_ok=True)
    print("Extracting features from tactical tracking...", flush=True)
    features = read_distances(with_ball_speed=args.with_ball_speed)
    gt = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    reason = pm.validate_submission(gt, len(features))
    if reason:
        raise ValueError(reason)
    labels = gt["label"].to_numpy()
    public = np.flatnonzero(labels != "X")
    if not np.array_equal(public, np.arange(public[0], public[-1] + 1)):
        raise ValueError("Expected one contiguous public-labelled window")
    fps = GAMES[GAME]["fps"]
    train_end = int(public[0] + TRAIN_SECONDS * fps)
    val_start = train_end + GAP_SECONDS * fps
    train = np.arange(public[0], train_end)
    validation = np.arange(val_start, public[-1] + 1)
    if args.with_ball_speed and GAP_SECONDS * fps < fps:
        raise ValueError("Gap must exceed the speed feature's one-second lookback")
    if not len(validation) or not np.isin(labels[train], CLASSES).all():
        raise ValueError("Invalid train/validation split")
    train_targets = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
    if not np.array_equal(np.unique(train_targets), np.arange(3)):
        raise ValueError("Training block must contain all three classes")

    # Independent alignment/feature sanity check against the shipped raw baseline.
    nearest = np.where(
        features[FEATURES].fillna(np.inf).to_numpy().argmin(axis=1) == 0, "H", "A",
    )
    nearest[features[FEATURES].isna().all(axis=1).to_numpy()] = "D"
    shipped_raw = pm.read_label_csv(ROOT / "evaluation/results/baseline_raw.csv")
    mismatches = int(np.sum(nearest[public] != shipped_raw["label"].to_numpy()[public]))
    if mismatches:
        raise ValueError(f"Raw nearest-player sanity check failed: {mismatches} frames")

    print(f"Train: {len(train)} frames; gap: {GAP_SECONDS * fps}; "
          f"validation: {len(validation)} frames", flush=True)
    model = XGBClassifier(**PARAMETERS)
    # Validation is not used for training, early stopping or parameter selection.
    model.fit(features.iloc[train], train_targets)
    probabilities = model.predict_proba(features.iloc[validation])
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite class probabilities")
    predictions = {
        "xgboost_raw": CLASSES[probabilities.argmax(axis=1)],
        "xgboost_smooth25": CLASSES[probability_smoothing(probabilities).argmax(axis=1)],
    }
    for variant in ("raw", "smooth25", "smooth50", "smooth125"):
        baseline = pm.read_label_csv(ROOT / f"evaluation/results/baseline_{variant}.csv")
        predictions[f"organiser_{variant}"] = baseline["label"].to_numpy()[validation]

    truth = pm.label_codes(labels[validation])
    scores = []
    for name, prediction in predictions.items():
        result = pm.evaluate(truth, pm.label_codes(prediction), bootstrap_n=1000)
        scores.append(dict(
            model=name, macroF1=result["macro_f1"], spellF1=result["spell_f1_mean"],
            possessions=result["n_pred_spells"], true_possessions=result["n_gt_spells"],
            spellF1_low=result["spell_bootstrap_low"],
            spellF1_high=result["spell_bootstrap_high"],
            dead_ball_F1=result["per_class"]["D"]["f1"],
        ))
    scoreboard = pd.DataFrame(scores).sort_values("spellF1", ascending=False)
    scoreboard.to_csv(output / "validation_scores.csv", index=False)
    details = features.iloc[validation].copy()
    details.insert(0, "frame", validation)
    details["ground_truth"] = labels[validation]
    for i, label in enumerate(CLASSES):
        details[f"probability_{label}"] = probabilities[:, i]
    for name, prediction in predictions.items():
        details[name] = prediction
    details.to_csv(output / "validation_predictions.csv", index=False)
    model.save_model(output / "model.ubj")
    manifest = dict(
        features=list(features.columns), parameters=PARAMETERS,
        speed_window_frames=fps if args.with_ball_speed else None,
        speed_min_observations=fps // 2 + 1 if args.with_ball_speed else None,
        train_frames_inclusive=[int(train[0]), int(train[-1])],
        validation_frames_inclusive=[int(validation[0]), int(validation[-1])],
        training_class_counts=pd.Series(labels[train]).value_counts().to_dict(),
        validation_class_counts=pd.Series(labels[validation]).value_counts().to_dict(),
        missing_feature_fraction=features.iloc[validation].isna().mean().to_dict(),
        training_feature_correlation=features.iloc[train].corr().to_dict(),
        smoothing_half_window_frames=SMOOTH_HALF_WINDOW,
        raw_baseline_sanity_check_mismatches=mismatches,
        caveat="One small holdout, truncated boundary possessions, no reliable full-match ranking. "
               "Organiser baselines use full-match input context; XGBoost smoothing is block-local. "
               "The saved model was trained on 180 seconds, not the entire public window.",
    )
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(scoreboard.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved diagnostic outputs to {output}; these are NOT submission CSVs.")


if __name__ == "__main__":
    main()
