"""Target-free monotone alignment of a fixed possession-label transcript.

Within each reference-live component, keep the reference run order/count and
choose boundaries lexicographically closest to the proposal, then reference.
"""

import numpy as np


def run_transcript(labels):
    """The ordered labels of maximal runs, including one-frame runs."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Expected a one-dimensional label sequence")
    if not len(labels):
        return labels.copy()
    return labels[np.r_[True, labels[1:] != labels[:-1]]]


def align_component(reference, proposal):
    """Exact stay/advance DP; equal cumulative costs prefer staying in state."""
    reference, proposal = np.asarray(reference), np.asarray(proposal)
    if (reference.ndim != 1 or proposal.shape != reference.shape
            or not np.isin(reference, ["H", "A"]).all()
            or not np.isin(proposal, ["H", "A"]).all()):
        raise ValueError("A component requires aligned one-dimensional H/A labels")
    length = len(reference)
    if not length:
        return reference.copy()
    transcript = run_transcript(reference)
    states = len(transcript)
    # A secondary Hamming difference is at most T, so T+1 gives exact
    # lexicographic ordering without a fitted weight or floating-point ties.
    scale = length + 1
    infinity = np.iinfo(np.int64).max // 4
    if length * (length + 2) >= infinity:
        raise ValueError("Sequence too long for exact integer costs")
    previous = np.full(states, infinity, dtype=np.int64)
    previous[0] = int(transcript[0] != proposal[0]) * scale + int(transcript[0] != reference[0])
    advanced = np.zeros((length, states), dtype=bool)
    for frame in range(1, length):
        advance = np.r_[infinity, previous[:-1]]
        take_advance = advance < previous
        cost = ((transcript != proposal[frame]).astype(np.int64) * scale
                + (transcript != reference[frame]).astype(np.int64))
        previous = np.minimum(previous, advance) + cost
        # Unreachable states remain exactly infinite rather than accumulating
        # sentinel costs, and cannot affect a later valid backtrack.
        previous[frame + 1:] = infinity
        advanced[frame] = take_advance
    if previous[-1] >= infinity:
        raise AssertionError("Reference transcript has no valid alignment")
    result = np.empty_like(reference)
    state = states - 1
    for frame in range(length - 1, -1, -1):
        result[frame] = transcript[state]
        if advanced[frame, state]:
            state -= 1
    if state != 0 or not np.array_equal(run_transcript(result), transcript):
        raise AssertionError("Alignment changed the reference transcript")
    return result


def project_transcript(reference, proposal, blocks):
    """Project each live component separately; D/X and outside blocks stay exact."""
    reference, proposal = np.asarray(reference), np.asarray(proposal)
    if (reference.ndim != 1 or proposal.shape != reference.shape
            or not np.isin(reference, ["H", "A", "D", "X"]).all()
            or not np.isin(proposal, ["H", "A", "D", "X"]).all()):
        raise ValueError("Expected aligned one-dimensional H/A/D/X sequences")
    live = np.isin(reference, ["H", "A"])
    if not np.isin(proposal[live], ["H", "A"]).all():
        raise ValueError("The proposal must assign H/A on reference-live frames")
    result = reference.copy()
    visited = np.zeros(len(reference), dtype=bool)
    for start, end in blocks:
        if (not isinstance(start, (int, np.integer))
                or not isinstance(end, (int, np.integer))
                or not 0 <= start < end <= len(reference)
                or visited[start:end].any()):
            raise ValueError("Invalid or overlapping decode blocks")
        visited[start:end] = True
        edges = np.flatnonzero(np.diff(np.r_[False, live[start:end], False]))
        for left, right in zip(edges[::2], edges[1::2]):
            begin, stop = start + left, start + right
            result[begin:stop] = align_component(reference[begin:stop], proposal[begin:stop])
    np.testing.assert_array_equal(result[~live], reference[~live])
    np.testing.assert_array_equal(result[~visited], reference[~visited])
    return result
