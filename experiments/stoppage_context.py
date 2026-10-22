"""Longer local stoppage context and strictly excluded weak training samples."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from control_refinement import control_evidence
from games import GAME, GAMES, half_video_blocks, require
from stoppage_features import restart_features
from starter.load_smart_tagging import load as load_tags


def ball_context(ball):
    valid = np.isfinite(ball).all(axis=1)
    xy = pd.DataFrame(np.where(valid[:, None], ball, np.nan))
    result = {}
    for radius in (25, 75, 125):
        width = 2*radius+1
        rolling = xy.rolling(width, center=True, min_periods=(width+1)//2)
        spread = rolling.quantile(.75)-rolling.quantile(.25)
        result[f"ball_iqr_radius{radius}"] = np.hypot(spread[0], spread[1])
        result[f"ball_observed_radius{radius}"] = pd.Series(valid.astype(float)).rolling(
            width, center=True, min_periods=1).mean()
    return pd.DataFrame(result).astype(np.float32)


def load_ball(n_frames):
    path = require(["atd_raw"])[0]
    ball = np.full((n_frames, 2), np.nan)
    count = 0
    with open(path) as handle:
        for frame, line in enumerate(handle):
            if frame >= n_frames:
                break
            frame_id, _, ball_raw = line.strip().split(":")
            if int(frame_id) != frame+1:
                raise ValueError("Bad tracking frame")
            ball[frame] = np.fromstring(ball_raw, sep=",")[:2]*np.array(GAMES[GAME]["pitch"])
            count += 1
    if count != n_frames:
        raise ValueError("Tracking ended early")
    return ball


def context_features(raw, base, ball):
    shared = base[[c for c in base if not c.startswith(("h__", "a__")) and
                   c not in ("ball_goal_line_distance", "ball_touchline_distance")]].copy()
    for column in base:
        if column.startswith("h__"):
            pair = base[[column, "a__"+column[3:]]]
            shared[column[3:]+"_min"] = pair.min(axis=1)
            shared[column[3:]+"_max"] = pair.max(axis=1)
    for name in ("spread_x", "spread_y", "center_ball_distance", "distance_median", "speed_p90"):
        pair = raw[[prefix+name for prefix in ("h__", "a__")]]
        shared[name+"_min"] = pair.min(axis=1)
        shared[name+"_max"] = pair.max(axis=1)
    summaries = {}
    for name in ("ball_seen", "ball_displacement_1s", "distance_1_min", "speed_median_min", "speed_median_max",
                 "moving_fraction_min", "moving_fraction_max", "spread_x_min", "spread_y_min"):
        column = shared[name]
        for window in (25, 75, 125):
            summaries[name+f"_context_past_{window}"] = column.rolling(window, min_periods=1).mean()
            summaries[name+f"_context_future_{window}"] = column.iloc[::-1].rolling(window, min_periods=1).mean().iloc[::-1]
    physical = shared.join(pd.DataFrame(summaries)).join(ball_context(ball)).astype(np.float32)
    restarts = restart_features(len(raw), load_tags())
    return physical, physical.join(restarts)


def weak_samples(raw, physical, restarts, reserved_windows):
    n = len(raw)
    eligible = np.zeros(n, bool)
    for _, start, end in half_video_blocks(GAME):
        eligible[max(start, 125):min(end+1, n-125)] = True
    eligible[max(0, 11185-250):min(n, 18685+250)] = False
    for window in reserved_windows:
        eligible[max(0, window["start"]-250):min(n, window["end"]+251)] = False
    live_fraction = pd.Series(control_evidence(raw).max(axis=1)).rolling(25, center=True, min_periods=1).mean().to_numpy()
    live = live_fraction >= .5
    dead = ((restarts.restart_all_active.to_numpy() > 0) &
            (physical.ball_observed_radius25.to_numpy() >= .7) &
            (physical.ball_iqr_radius25.to_numpy() < 1.5))
    unambiguous = dead ^ live
    frames = np.flatnonzero(eligible & unambiguous & (np.arange(n) % 10 == 0))
    return frames, dead[frames].astype(int)


def fit_context(inputs, labels, train, weak_frames, weak_target, weak_mass, seed=42):
    target = (labels[train] == "D").astype(int)
    if set(labels[train]) != {"H", "A", "D"}:
        raise ValueError("Public training must contain H/A/D only")
    weight = len(target)/(2*np.bincount(target)[target])
    x = inputs.iloc[train]
    if weak_mass:
        if set(weak_target) != {0, 1} or np.intersect1d(train, weak_frames).size:
            raise ValueError("Weak rows need both classes and cannot overlap public training")
        weak_weight = len(target)*weak_mass/(2*np.bincount(weak_target)[weak_target])
        x = pd.concat([x, inputs.iloc[weak_frames]], ignore_index=True)
        target, weight = np.r_[target, weak_target], np.r_[weight, weak_weight]
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          max_depth=2, n_estimators=200, learning_rate=.04, reg_lambda=20,
                          min_child_weight=20, subsample=.8, colsample_bytree=.9, random_state=seed, n_jobs=2)
    model.fit(x, target, sample_weight=weight)
    return model
