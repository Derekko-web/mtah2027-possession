"""Prospective boundary and anchored live-rescue experiments; no target inputs."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from consensus_control import consensus_team
from control_refinement import compose, contact_support, control_evidence, fast_columns, smooth
from evaluate_physics_context import apply_context
from rich_features import swap_teams
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES


def runs(states):
    states = np.asarray(states)
    edges = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
    return [(int(a), int(b), int(states[a])) for a, b in zip(edges[:-1], edges[1:])]


def relocate_boundaries(states, emissions, radius):
    """Jointly optimize ordered boundaries, preserving all original run labels.

    Each boundary is within radius of its original location. Runs cannot vanish
    or cross. Equal-likelihood positions prefer the original boundaries.
    """
    states, emissions = np.asarray(states), np.asarray(emissions, dtype=float)
    if states.ndim != 1 or emissions.ndim != 2 or len(states) != len(emissions):
        raise ValueError("States and emissions must align")
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Radius must be a nonnegative integer")
    if not np.isfinite(emissions).all() or (emissions < 0).any():
        raise ValueError("Invalid emissions")
    if not len(states):
        return states.copy()
    if (emissions.sum(axis=1) <= 0).any() or states.min() < 0 or states.max() >= emissions.shape[1]:
        raise ValueError("Invalid state or zero probability mass")
    spans = runs(states)
    if len(spans) == 1:
        return states.copy()
    probability = emissions / emissions.sum(axis=1, keepdims=True)
    prefix = np.vstack((np.zeros(emissions.shape[1]),
                        np.cumsum(-np.log(np.maximum(probability, 1e-12)), axis=0)))
    candidates = [np.array([0])]
    for _, boundary, _ in spans[:-1]:
        candidates.append(np.arange(max(1, boundary-radius), min(len(states)-1, boundary+radius)+1))
    candidates.append(np.array([len(states)]))
    costs, back = np.zeros(1), []
    for i, (_, end, owner) in enumerate(spans):
        left, right = candidates[i], candidates[i+1]
        trial = costs[:, None] + prefix[right, owner][None, :] - prefix[left, owner][:, None]
        trial = np.where(left[:, None] < right[None, :], trial, np.inf)
        # Negligible deterministic preference, used only for numerical ties.
        trial += 1e-10 * np.abs(right-end)[None, :]
        previous = trial.argmin(axis=0)
        costs = trial[previous, np.arange(len(right))]
        back.append(previous)
    if not np.isfinite(costs[0]):
        raise ValueError("No feasible boundary path")
    positions, index = [len(states)], 0
    for i in range(len(spans)-1, -1, -1):
        index = back[i][index]
        positions.append(int(candidates[i][index]))
    positions.reverse()
    result = np.empty_like(states)
    for (start, end), (_, _, owner) in zip(zip(positions[:-1], positions[1:]), spans):
        result[start:end] = owner
    return result


def refine_block(current, previous, fast, support, evidence, method):
    """All arrays are already context adjusted and restricted to one decode block."""
    slow, old, short = smooth(current, 25), smooth(previous, 25), smooth(fast, 5)
    owner = consensus_team(slow, old, short, support)
    attenuation = 1-.75*evidence.max(axis=1)
    dead = slow[:, 2]*attenuation
    emission = compose(dead, owner)
    baseline = decode(emission)
    if method in ("anchored_rescue", "rescue_boundary13"):
        # Determine every gate against the unchanged baseline, preventing
        # recursive expansion from one accepted proposal into another.
        for start, end, team in runs(decode(old)):
            if team == 2 or not 8 <= end-start <= 125:
                continue
            if short[start:end, team].mean() < .65 or support[start:end, team].sum() < 3:
                continue
            anchor = baseline[max(0, start-13):min(len(baseline), end+13)]
            if not np.any(anchor == team) or not np.any(baseline[start:end] == 2):
                continue
            owner[start:end] = short[start:end]
            dead[start:end] = np.minimum(slow[start:end, 2], old[start:end, 2])*attenuation[start:end]
        emission = compose(dead, owner)
    states = decode(emission)
    if method in ("boundary13", "boundary25", "rescue_boundary13"):
        fine = smooth(current, 5)
        total = slow[:, :2].sum(axis=1, keepdims=True)
        reference = np.divide(slow[:, :2], total, out=np.full_like(owner, .5), where=total > 0)
        proposal = np.any(np.abs(owner-reference) > 1e-10, axis=1)
        fine_total = fine[:, :2].sum(axis=1, keepdims=True)
        fine_owner = np.divide(fine[:, :2], fine_total, out=np.full_like(owner, .5), where=fine_total > 0)
        fine_owner[proposal] = short[proposal]
        fine_dead = fine[:, 2]*attenuation
        # Keep accepted rescue evidence in the fine likelihood too.
        if method == "rescue_boundary13":
            fine_dead = np.minimum(fine_dead, dead)
        fine_emission = compose(fine_dead, fine_owner)
        pooled = np.sqrt(np.maximum(emission, 1e-12)*np.maximum(fine_emission, 1e-12))
        states = relocate_boundaries(states, pooled, 25 if method == "boundary25" else 13)
    elif method not in ("current", "anchored_rescue", "micro_frame", "micro_segments"):
        raise ValueError("Unknown refinement method")
    return states


def micro_inputs(base):
    """Short temporal order, in frames; no fit, frame IDs, locations or labels."""
    result = base[fast_columns(base)].copy()
    for team, other in (("h__", "a__"), ("a__", "h__")):
        for name in ("distance_1", "control_share"):
            for lag in (-13, -8, -3, 3, 8, 13):
                result[team+name+f"_micro_lag_{lag}"] = base[team+name].shift(lag)
        result[team+"distance_delta_5"] = base[team+"distance_1"]-base[team+"distance_1"].shift(5)
        contact = (base[team+"distance_1"] < 2) & (base[other+"distance_1"]-base[team+"distance_1"] > .5)
        for window in (7, 13):
            result[team+f"contact_micro_{window}"] = contact.astype(float).rolling(
                window, center=True, min_periods=1).mean()
    return result.astype(np.float32)


def fit_micro(inputs, labels, train, segments=False, seed=42):
    train = np.asarray(train)
    live = train[np.isin(labels[train], ["H", "A"])]
    target = (labels[live] == "A").astype(int)
    if set(target) != {0, 1}:
        raise ValueError("Both live teams are required")
    weight = np.ones(len(live))
    if segments:
        group = np.cumsum(np.r_[True, (np.diff(live) != 1) | (target[1:] != target[:-1])])-1
        weight = 1/np.bincount(group)[group]
        weight /= weight.mean()
    augmented = pd.concat([inputs.iloc[live], swap_teams(inputs.iloc[live])], ignore_index=True)
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          max_depth=2, n_estimators=160, learning_rate=.04, reg_lambda=10,
                          min_child_weight=15, subsample=.8, colsample_bytree=.9, random_state=seed, n_jobs=2)
    model.fit(augmented, np.r_[target, 1-target], sample_weight=np.r_[weight, weight])
    return model


def decode_refinement(current, previous, fast, raw, blocks, method="anchored_rescue"):
    """Shared OOF/full-match inference, with independent block/half decoding."""
    n = len(raw)
    for value, columns in ((current, 3), (previous, 3), (fast, 2)):
        if np.asarray(value).shape != (n, columns):
            raise ValueError("Probabilities and observations must align")
    side = gap_sides(raw)
    support, evidence = contact_support(raw), control_evidence(raw)
    result = np.full(n, "X", dtype="<U1")
    for start, end in blocks:
        if not 0 <= start < end <= n or np.any(result[start:end] != "X"):
            raise ValueError("Invalid or overlapping blocks")
        for value in (current, previous, fast):
            block = value[start:end]
            if not np.isfinite(block).all() or (block < 0).any() or not np.allclose(block.sum(axis=1), 1):
                raise ValueError("Invalid block probabilities")
        sides = side[start:end]
        p = apply_context(current[start:end], sides, np.zeros(end-start), False)
        q = apply_context(previous[start:end], sides, np.zeros(end-start), False)
        live = apply_context(np.column_stack((fast[start:end], np.zeros(end-start))), sides,
                             np.zeros(end-start), False)[:, :2]
        live /= live.sum(axis=1, keepdims=True)
        result[start:end] = CLASSES[refine_block(p, q, live, support[start:end], evidence[start:end], method)]
    return result
