"""Excluded weak H/A labels and ownership-only research inference."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from boundary_refinement import runs
from control_refinement import control_evidence, smooth
from games import GAME, half_video_blocks
from rich_features import swap_teams
from transition_penalty import decode


def sample_weak_ownership(raw, physical, restart, phase, reserved_windows):
    n = len(raw)
    if any(len(value) != n for value in (physical, restart, phase)):
        raise ValueError("Weak sample observations must align")
    eligible = np.zeros(n, bool)
    for _, start, end in half_video_blocks(GAME):
        eligible[max(start, 125):min(end+1, n-125)] = True
    eligible[max(0, 11185-250):min(n, 18685+250)] = False
    for window in reserved_windows:
        eligible[max(0, window["start"]-250):min(n, window["end"]+251)] = False
    eligible &= np.arange(n) % 10 == 0
    contact = smooth(control_evidence(raw), 12)
    contact_target = np.full(n, -1, dtype=int)
    phase_target = np.full(n, -1, dtype=int)
    phase_support = phase[[prefix+"phase_control_support" for prefix in ("h__", "a__")]].to_numpy()
    moving = ((physical.ball_observed_radius25.to_numpy() >= .7) &
              (physical.ball_iqr_radius25.to_numpy() > 2) & (restart.restart_all_active.to_numpy() == 0))
    for team in (0, 1):
        contact_target[(contact[:, team] >= .5) & (contact[:, 1-team] <= .05)] = team
        exclusive = (phase_support[:, team] >= .5) & (phase_support[:, 1-team] == 0)
        stable = pd.Series(exclusive.astype(float)).rolling(25, center=True, min_periods=25).min().to_numpy() == 1
        phase_target[stable & moving & (contact[:, 1-team] < .5)] = team
    conflict = (contact_target >= 0) & (phase_target >= 0) & (contact_target != phase_target)
    phase_target[conflict] = -1
    contact_target[conflict] = -1
    agreement = np.where((contact_target >= 0) & (contact_target == phase_target), contact_target, -1)
    result = {}
    for name, target in (("contact", contact_target), ("phase", phase_target), ("agreement", agreement)):
        frames = np.flatnonzero(eligible & (target >= 0))
        result[name] = (frames, target[frames])
    return result


def fit_weak_ownership(inputs, labels, train, weak_frames, weak_target, weak_mass, seed=42):
    train = np.asarray(train, dtype=int)
    if not len(train) or not set(labels[train]) <= {"H", "A", "D"}:
        raise ValueError("Training rows must have eligible public targets")
    live = train[np.isin(labels[train], ["H", "A"])]
    target = (labels[live] == "A").astype(int)
    if set(target) != {0, 1}:
        raise ValueError("Both public live teams required")
    x = inputs.iloc[live]
    weight = len(target)/(2*np.bincount(target)[target])
    if weak_mass:
        weak_frames, weak_target = np.asarray(weak_frames), np.asarray(weak_target)
        if len(weak_frames) != len(weak_target) or set(weak_target) != {0, 1} or np.intersect1d(train, weak_frames).size:
            raise ValueError("Weak rows need both classes and must be disjoint from training")
        weak_weight = len(target)*weak_mass/(2*np.bincount(weak_target)[weak_target])
        x = pd.concat([x, inputs.iloc[weak_frames]], ignore_index=True)
        target, weight = np.r_[target, weak_target], np.r_[weight, weak_weight]
    augmented = pd.concat([x, swap_teams(x)], ignore_index=True)
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          max_depth=3, n_estimators=200, learning_rate=.04, reg_lambda=20,
                          min_child_weight=20, subsample=.8, colsample_bytree=.9,
                          random_state=seed, n_jobs=2)
    model.fit(augmented, np.r_[target, 1-target], sample_weight=np.r_[weight, weight])
    return model


def ownership_prediction(reference, probability, support, phase_support, mode):
    """Keep D fixed while replacing ownership or adding independently gated islands."""
    reference, probability = np.asarray(reference), np.asarray(probability, dtype=float)
    n = len(reference)
    if probability.shape != (n, 2) or np.asarray(support).shape != (n, 2) or np.asarray(phase_support).shape != (n, 2):
        raise ValueError("Ownership inference inputs must align")
    if not np.isin(reference, [0, 1, 2]).all() or not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1):
        raise ValueError("Invalid states or ownership probabilities")
    result = reference.copy()
    if mode == "replace_live":
        edges = np.flatnonzero(np.diff(np.r_[False, reference != 2, False]))
        for start, end in zip(edges[::2], edges[1::2]):
            result[start:end] = decode(smooth(probability[start:end], 5))
    elif mode == "short_proposals":
        p = smooth(probability, 5)
        for start, end, team in runs(decode(p)):
            if not 8 <= end-start <= 125 or start == 0 or end == n or np.any(reference[start:end] == 2):
                continue
            if p[start:end, team].mean() < .65:
                continue
            if support[start:end, team].sum() < 3 and (phase_support[start:end, team] >= .5).sum() < 3:
                continue
            before, after = reference[max(0, start-13):start], reference[end:min(n, end+13)]
            if np.mean(before == 1-team) <= .5 or np.mean(after == 1-team) <= .5:
                continue
            result[start:end] = team
    else:
        raise ValueError("Unknown ownership mode")
    if not np.array_equal(result == 2, reference == 2):
        raise AssertionError("Ownership inference changed D")
    return result
