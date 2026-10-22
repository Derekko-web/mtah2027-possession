"""Constrained short H/A endpoint research; no labels enter inference."""

import numpy as np

from boundary_refinement import runs
from control_refinement import smooth


def ownership_evidence(previous, fast, source, half_window, log_pool=False):
    previous, fast = np.asarray(previous, dtype=float), np.asarray(fast, dtype=float)
    if previous.shape != (len(fast), 3) or fast.shape != (len(previous), 2):
        raise ValueError("Expected aligned three-state and live-only probabilities")
    for value in (previous, fast):
        if not np.isfinite(value).all() or (value < 0).any() or not np.allclose(value.sum(axis=1), 1):
            raise ValueError("Invalid probabilities")
    if source not in ("previous", "fast", "joint") or not isinstance(half_window, int) or half_window < 0:
        raise ValueError("Invalid evidence settings")
    total = previous[:, :2].sum(axis=1, keepdims=True)
    old = np.divide(previous[:, :2], total, out=np.full_like(fast, .5), where=total > 0)
    heads = [old] if source == "previous" else [fast] if source == "fast" else [old, fast]
    if log_pool:
        score = smooth(np.mean([np.log(np.maximum(p, 1e-12)) for p in heads], axis=0), half_window)
        probability = np.exp(score-score.max(axis=1, keepdims=True))
    else:
        probability = smooth(np.mean(heads, axis=0), half_window)
    return probability/probability.sum(axis=1, keepdims=True)


def refine_short_endpoints(states, probability, radius):
    """Jointly relocate only boundaries of live islands, preserving D and topology.

    Eligible islands are bracketed by opposite-team live control. Every gate is
    evaluated against the unchanged reference. A global ordered-boundary DP
    handles adjacent islands without cascades, crossings or vanishing runs.
    """
    states, probability = np.asarray(states), np.asarray(probability, dtype=float)
    if states.ndim != 1 or probability.shape != (len(states), 2):
        raise ValueError("States and binary ownership evidence must align")
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Radius must be a nonnegative integer")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability.sum(axis=1) <= 0).any():
        raise ValueError("Invalid ownership evidence")
    if not len(states):
        return states.copy()
    if not np.isin(states, [0, 1, 2]).all():
        raise ValueError("Expected H/A/D state indices")
    spans = runs(states)
    eligible = {i for i, (start, end, team) in enumerate(spans)
                if team != 2 and 8 <= end-start <= 125 and 0 < i < len(spans)-1
                and spans[i-1][2] == 1-team and spans[i+1][2] == 1-team}
    if not eligible or radius == 0:
        return states.copy()
    movable = {boundary for i in eligible for boundary in (i, i+1)}
    positions = [np.array([0])]
    for i, (_, end, _) in enumerate(spans[:-1], 1):
        positions.append(np.arange(max(1, end-radius), min(len(states)-1, end+radius)+1)
                         if i in movable else np.array([end]))
    positions.append(np.array([len(states)]))
    normalized = probability/probability.sum(axis=1, keepdims=True)
    losses = np.column_stack((-np.log(np.maximum(normalized, 1e-12)), np.zeros(len(states))))
    prefix = np.vstack((np.zeros(3), losses.cumsum(axis=0)))
    costs, back = np.zeros(1), []
    for i, (_, original_end, team) in enumerate(spans):
        left, right = positions[i], positions[i+1]
        trial = costs[:, None]+prefix[right, team][None, :]-prefix[left, team][:, None]
        minimum = 8 if i in eligible else 1
        trial = np.where(right[None, :]-left[:, None] >= minimum, trial, np.inf)
        trial += 1e-10*np.abs(right-original_end)[None, :]
        previous = trial.argmin(axis=0)
        costs = trial[previous, np.arange(len(right))]
        back.append(previous)
    if not np.isfinite(costs[0]):
        raise ValueError("No feasible endpoint path")
    cuts, index = [len(states)], 0
    for i in range(len(spans)-1, -1, -1):
        index = back[i][index]
        cuts.append(int(positions[i][index]))
    cuts.reverse()
    result = np.empty_like(states)
    for (start, end), (_, _, team) in zip(zip(cuts[:-1], cuts[1:]), spans):
        result[start:end] = team
    if not np.array_equal(result == 2, states == 2):
        raise AssertionError("Endpoint refinement changed a D frame")
    return result
