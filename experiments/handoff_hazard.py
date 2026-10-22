"""Train-only directed transition targets and topology-preserving handoff timing."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from boundary_refinement import runs
from control_refinement import smooth
from rich_features import swap_teams


def event_targets(labels, train):
    train = np.asarray(train)
    allowed = np.zeros(len(labels), bool)
    allowed[train] = True
    frames, home, away = [], [], []
    for frame in train:
        if frame < 4 or frame+3 >= len(labels):
            continue
        neighborhood = np.arange(frame-4, frame+4)
        if not allowed[neighborhood].all():
            continue
        piece = labels[neighborhood]
        if not np.isin(piece, ["H", "A"]).all():
            continue
        changes = piece[1:] != piece[:-1]
        frames.append(frame)
        home.append(bool(np.any(changes & (piece[1:] == "H"))))
        away.append(bool(np.any(changes & (piece[1:] == "A"))))
    return np.array(frames, int), np.array(home, int), np.array(away, int)


def fit_hazard(inputs, labels, train, seed=42):
    frames, home, away = event_targets(labels, train)
    target = np.r_[home, away]
    if set(target) != {0, 1}:
        raise ValueError("Directed transition training needs both event classes")
    x = pd.concat([inputs.iloc[frames], swap_teams(inputs.iloc[frames])], ignore_index=True)
    weight = len(target)/(2*np.bincount(target)[target])
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          max_depth=2, n_estimators=120, learning_rate=.04, reg_lambda=10,
                          min_child_weight=15, subsample=.8, colsample_bytree=.9, random_state=seed, n_jobs=2)
    model.fit(x, target, sample_weight=weight)
    return model


def predict_hazard(model, inputs):
    return np.column_stack((model.predict_proba(inputs)[:, 1], model.predict_proba(swap_teams(inputs))[:, 1]))


def relocate_handoffs(states, emissions, hazard, radius, weight):
    """Joint timing optimization; fixed D spans, fixed live topology, no labels."""
    states = np.asarray(states)
    if np.asarray(emissions).shape != (len(states), 3) or np.asarray(hazard).shape != (len(states), 2):
        raise ValueError("States, emissions and hazards must align")
    if not np.isfinite(emissions).all() or not np.isfinite(hazard).all() or (emissions < 0).any() or (hazard < 0).any():
        raise ValueError("Invalid boundary evidence")
    if not isinstance(radius, int) or radius < 0 or not np.isfinite(weight) or weight < 0:
        raise ValueError("Invalid boundary settings")
    if len(states) == 0:
        return states.copy()
    spans = runs(states)
    probability = emissions/emissions.sum(axis=1, keepdims=True)
    prefix = np.vstack((np.zeros(3), np.cumsum(-np.log(np.maximum(probability, 1e-12)), axis=0)))
    hazard = smooth(hazard, 3)
    candidates = [np.array([0])]
    for index, (_, boundary, owner) in enumerate(spans[:-1]):
        incoming = spans[index+1][2]
        if owner == 2 or incoming == 2:
            candidates.append(np.array([boundary]))
        else:
            candidates.append(np.arange(max(1, boundary-radius), min(len(states)-1, boundary+radius)+1))
    candidates.append(np.array([len(states)]))
    costs, back = np.zeros(1), []
    for index, (_, end, owner) in enumerate(spans):
        left, right = candidates[index], candidates[index+1]
        trial = costs[:, None]+prefix[right, owner][None, :]-prefix[left, owner][:, None]
        trial = np.where(left[:, None] < right[None, :], trial, np.inf)
        if index < len(spans)-1 and owner != 2 and spans[index+1][2] != 2:
            incoming = spans[index+1][2]
            trial -= weight*np.log(np.maximum(hazard[right, incoming], 1e-12))[None, :]
        trial += 1e-10*np.abs(right-end)[None, :]
        previous = trial.argmin(axis=0)
        costs = trial[previous, np.arange(len(right))]
        back.append(previous)
    if not np.isfinite(costs[0]):
        raise ValueError("No feasible handoff path")
    positions, index = [len(states)], 0
    for position in range(len(spans)-1, -1, -1):
        index = back[position][index]
        positions.append(int(candidates[position][index]))
    positions.reverse()
    result = np.empty_like(states)
    for (start, end), (_, _, owner) in zip(zip(positions[:-1], positions[1:]), spans):
        result[start:end] = owner
    return result
