"""Fast control evidence and bounded physical support for the refinement round."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from build_candidate import PARAMS, objective_weights
from rich_features import swap_teams
from xgboost_baseline import CLASSES


def fast_columns(base):
    return [name for name in base if name not in ("ball_goal_line_distance", "ball_touchline_distance")
            and "__past_" not in name and "__future_" not in name
            and "future_" not in name and "past_" not in name]


def fit_control(inputs, labels, train, seed=42):
    live = train[np.isin(labels[train], ["H", "A"])]
    if set(labels[live]) != {"H", "A"}:
        raise ValueError("Team head requires both live classes")
    target = (labels[live] == "A").astype(int)
    augmented = pd.concat([inputs.iloc[live], swap_teams(inputs.iloc[live])], ignore_index=True)
    y = np.r_[target, 1-target]
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          max_depth=2, n_estimators=160, learning_rate=.04, reg_lambda=10,
                          min_child_weight=15, subsample=.8, colsample_bytree=.9, random_state=seed, n_jobs=2)
    model.fit(augmented, y)
    return model


def predict_control(model, inputs):
    away = (model.predict_proba(inputs)[:, 1] +
            1-model.predict_proba(swap_teams(inputs))[:, 1])/2
    return np.column_stack((1-away, away))


def fit_dead_weight(inputs, labels, train, mode, seed=42):
    if set(labels[train]) != set(CLASSES):
        raise ValueError("Three-state training requires H/A/D only")
    target = np.array([np.flatnonzero(CLASSES == value)[0] for value in labels[train]])
    y, weight = objective_weights(target, train, False)
    if mode == "empirical":
        weight[:] = 1
    elif mode == "dead075":
        weight[y == 2] *= .75
    else:
        raise ValueError("Unknown dead weight")
    model = XGBClassifier(**dict(PARAMS, max_depth=3, n_estimators=160, random_state=seed))
    augmented = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
    model.fit(augmented, y, sample_weight=weight)
    return model


def control_evidence(raw):
    """Close, exclusive, observed and moving ball; no targets or episode endpoints."""
    moving = raw.ball_displacement_1s.to_numpy() > 4.5
    evidence = []
    for own, other in (("h__", "a__"), ("a__", "h__")):
        close = ((raw[own+"distance_1"].to_numpy() < 2) &
                 (raw[other+"distance_1"].to_numpy()-raw[own+"distance_1"].to_numpy() > .5))
        evidence.append(pd.Series(close & moving).rolling(13, center=True, min_periods=1).mean().to_numpy())
    return np.column_stack(evidence)


def smooth(probability, half_window):
    return pd.DataFrame(probability).rolling(2*half_window+1, center=True, min_periods=1).mean().to_numpy()


def compose(dead_probability, team_probability):
    return np.column_stack(((1-dead_probability)[:, None]*team_probability, dead_probability))


def pooled_control_evidence(raw):
    """Pool asynchronous contact and moving-ball evidence within one second."""
    moving = pd.Series(raw.ball_displacement_1s.to_numpy() > 4.5).rolling(
        25, center=True, min_periods=1).mean().to_numpy()
    result = []
    for own, other in (("h__", "a__"), ("a__", "h__")):
        contact = ((raw[own+"distance_1"].to_numpy() < 2) &
                   (raw[other+"distance_1"].to_numpy()-raw[own+"distance_1"].to_numpy() > .5))
        fraction = pd.Series(contact).rolling(25, center=True, min_periods=1).mean().to_numpy()
        result.append((fraction >= .12) & (moving >= .2))
    return np.column_stack(result)


def contact_support(raw):
    result = []
    for own, other in (("h__", "a__"), ("a__", "h__")):
        contact = ((raw[own+"distance_1"].to_numpy() < 2) &
                   (raw[other+"distance_1"].to_numpy()-raw[own+"distance_1"].to_numpy() > .5))
        result.append(pd.Series(contact).rolling(25, center=True, min_periods=1).sum().to_numpy() >= 3)
    return np.column_stack(result)


def supported_emissions(slow, fast, support, moving_support, threshold=.65,
                        dead_mode="none", strength=.75):
    old_team = slow[:, :2]/slow[:, :2].sum(axis=1, keepdims=True)
    gate = ((fast > threshold) & support).any(axis=1)
    team = np.where(gate[:, None], fast, old_team)
    dead = slow[:, 2].copy()
    if dead_mode == "pooled":
        evidence = ((fast > threshold) & moving_support).any(axis=1)
        dead *= np.where(evidence, 1-strength, 1)
    elif dead_mode != "none":
        raise ValueError("Unknown dead support rule")
    return compose(dead, team)
