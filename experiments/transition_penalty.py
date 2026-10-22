"""Offline decoding that balances frame evidence against unnecessary state changes.

Minimize sum(-log(probability of selected state)) + penalty * state_changes.
The penalty is fixed at 3 before evaluation; there is no validation-driven search.
"""

import numpy as np

SWITCH_PENALTY = 3.0


def decode(probabilities, penalty=SWITCH_PENALTY):
    """Return the globally cheapest state sequence, using dynamic programming.

    Input is an n_frames x n_states probability matrix; output contains state
    indices. All states can transition to every other state. Equal initial costs,
    no minimum duration, no ground truth and no fixed boundary delay are used.
    Future observations inside this block can affect earlier decisions: this is
    suitable for offline submissions, not a real-time streaming implementation.
    """
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] < 1:
        raise ValueError("Expected a frames-by-states probability matrix")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Switch penalty must be finite and nonnegative")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if len(probability) == 0:
        return np.array([], dtype=int)
    totals = probability.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Every frame needs nonzero probability mass")
    probability = probability / totals
    losses = -np.log(np.maximum(probability, 1e-12))
    n_frames, n_states = losses.shape
    change_cost = penalty * (np.arange(n_states)[:, None] != np.arange(n_states)[None, :])
    back = np.zeros((n_frames, n_states), dtype=int)
    cost = losses[0].copy()
    for frame in range(1, n_frames):
        candidates = cost[:, None] + change_cost
        back[frame] = candidates.argmin(axis=0)
        cost = losses[frame] + candidates[back[frame], np.arange(n_states)]
    states = np.empty(n_frames, dtype=int)
    states[-1] = cost.argmin()
    for frame in range(n_frames - 1, 0, -1):
        states[frame - 1] = back[frame, states[frame]]
    return states
