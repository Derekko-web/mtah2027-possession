"""Offline binary path decoding with a fixed cost per state transition."""

import numpy as np
from scipy.special import logit

from dfl_submission.decoder import centered_mean


STRIDE = 5


def binary_path(evidence, switch_cost):
    """Maximize summed state-one evidence minus the cost of switching states."""
    delta = np.empty(len(evidence), dtype=float)
    margin = 0.0
    for i, value in enumerate(evidence):
        margin = value + min(switch_cost, max(-switch_cost, margin))
        delta[i] = margin
    path = np.empty(len(evidence), dtype=bool)
    state = bool(delta[-1] >= 0)
    path[-1] = state
    for i in range(len(path) - 2, -1, -1):
        if state:
            state = bool(delta[i] >= -switch_cost)
        else:
            state = bool(delta[i] > switch_cost)
        path[i] = state
    return path


def sequence(probability, cutoff, cost, radius):
    """Decode on a 5 Hz grid, then assign each block to its original 25 Hz frames."""
    size = len(probability)
    smoothed = centered_mean(probability, radius, 0, size)
    evidence = logit(np.clip(smoothed, .001, .999)) - logit(cutoff)
    starts = np.arange(0, size, STRIDE)
    block = np.add.reduceat(evidence, starts) / STRIDE
    return np.repeat(binary_path(block, cost), STRIDE)[:size]


def decode_path(home, dead, *, dead_cutoff, dead_cost, owner_cost, radius):
    stoppage = sequence(dead, dead_cutoff, dead_cost, radius)
    result = np.full(len(dead), 2, dtype=np.int8)
    boundaries = np.r_[0, np.flatnonzero(stoppage[1:] != stoppage[:-1]) + 1, len(dead)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if not stoppage[start]:
            result[start:stop] = ~sequence(home[start:stop], .5, owner_cost, radius)
    return result
