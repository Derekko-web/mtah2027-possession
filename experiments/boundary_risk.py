"""One chronological pass of conditional expected threshold-IoU boundary edits.

This is a local surrogate, not global Bayes-risk optimization of spellF1.
Only the reference's existing H/A boundaries can move. No target labels enter
inference, and D/X, transcript order/count, and decode resets remain fixed.
"""

import numpy as np


PROBABILITY_FLOOR = 1e-12
THRESHOLD_NUMERATORS = (3, 4, 5, 6, 7)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _distribution(probability, columns):
    try:
        probability = np.asarray(probability, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected numeric probabilities") from error
    if (probability.ndim != 2 or probability.shape[1] != columns
            or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()
            or not np.allclose(probability.sum(axis=1), 1., rtol=1e-6, atol=1e-7)):
        raise ValueError("Invalid probability distributions")
    return probability


def conditional_live_probability(probability):
    """Condition raw H/A/D on H/A, then clip at 1e-12 and renormalize.

    All supplied rows must be valid, even outside decode blocks. A raw row
    assigning exactly zero H/A mass has no defined live conditional.
    """
    probability = _distribution(probability, 3)
    live_mass = probability[:, :2].sum(axis=1, keepdims=True)
    if (live_mass <= 0).any():
        raise ValueError("Every row needs positive H/A mass")
    live = np.maximum(probability[:, :2] / live_mass, PROBABILITY_FLOOR)
    return live / live.sum(axis=1, keepdims=True)


def boundary_posterior(live_probability, left_owner):
    """Distribution of t=1..N-1 under two anchored spans and uniform prior.

    Input is already clipped/normalized conditional H/A evidence. The chosen
    scalar likelihood uses every frame in this local pair of spans once, with
    temperature one. This product does not assert independent observations.
    """
    probability = _distribution(live_probability, 2)
    if (len(probability) < 2 or (probability <= 0).any()
            or not _integer(left_owner) or left_owner not in (0, 1)):
        raise ValueError("A boundary needs positive evidence and opposite H/A owners")
    logs = np.log(probability)
    # The full scalar likelihood at t=1 is a common factor for every t.
    # Moving t across frame i changes only that frame's owner, so its ratio
    # is u_i(left)/u_i(right). Canceling the common factor avoids subtracting
    # nearly equal long cumulative sums and leaves uniform evidence uniform.
    weight = np.r_[0., np.cumsum(logs[1:-1, left_owner]
                                - logs[1:-1, 1-left_owner])]
    weight = np.exp(weight - weight.max())
    return weight / weight.sum()


def expected_boundary_utilities(posterior):
    """Expected fraction of ten corresponding-span threshold matches per b.

    posterior[t-1] is P(T=t), t=1..N-1; returned entry b-1 is the utility
    for b=1..N-1. Threshold bounds use exact integer arithmetic for k/10.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    if (posterior.ndim != 1 or not len(posterior)
            or not np.isfinite(posterior).all() or (posterior < 0).any()
            or (posterior > 1).any()
            or not np.isclose(posterior.sum(), 1., rtol=1e-6, atol=1e-7)):
        raise ValueError("Expected a normalized nonempty boundary distribution")
    length = len(posterior) + 1
    boundary = np.arange(1, length, dtype=np.int64)
    # Normalizing the complete CDF keeps its terminal value exactly one and
    # preserves monotonicity despite floating-point cumulative summation.
    cdf = np.r_[0., np.cumsum(posterior)]
    cdf /= cdf[-1]
    utility = np.zeros(len(boundary), dtype=np.float64)

    def mass(low, high):
        low, high = np.maximum(low, 1), np.minimum(high, length-1)
        valid = low <= high
        result = np.zeros(len(boundary), dtype=np.float64)
        result[valid] = cdf[high[valid]] - cdf[low[valid]-1]
        return result

    for numerator in THRESHOLD_NUMERATORS:
        left_low = (numerator*boundary + 9) // 10
        left_high = (10*boundary) // numerator
        right_length = length-boundary
        right_low = length - (10*right_length) // numerator
        right_high = length - (numerator*right_length + 9) // 10
        utility += mass(left_low, left_high) + mass(right_low, right_high)
    return utility / 10.


def minimum_risk_boundary(posterior, original_boundary):
    """Choose the exact computed maximum, then nearest original anchor/smallest.

    The helper accepts anchors outside the current feasible interval as well.
    No tolerance turns a near tie into an exact tie.
    """
    if not _integer(original_boundary):
        raise ValueError("The tie anchor must be an integer")
    utility = expected_boundary_utilities(posterior)
    winners = np.flatnonzero(utility == utility.max()) + 1
    return min((int(value) for value in winners),
               key=lambda value: (abs(value-int(original_boundary)), value))


def _align_component(reference, live_probability):
    endpoints = np.r_[0, np.flatnonzero(reference[1:] != reference[:-1])+1,
                      len(reference)].astype(np.int64)
    original = endpoints.copy()
    owners = reference[endpoints[:-1]]
    for index in range(1, len(endpoints)-1):
        left, right = int(endpoints[index-1]), int(endpoints[index+1])
        owner = 0 if owners[index-1] == "H" else 1
        posterior = boundary_posterior(live_probability[left:right], owner)
        endpoints[index] = left + minimum_risk_boundary(
            posterior, int(original[index])-left)
    if (np.diff(endpoints) <= 0).any():
        raise AssertionError("Boundary pass removed or crossed a reference run")
    result = reference.copy()
    for left, right, owner in zip(endpoints[:-1], endpoints[1:], owners):
        result[left:right] = owner
    return result


def decode_boundary_risk(reference, probability, blocks):
    """Edit reference H/A boundaries once within independent decode blocks.

    probability is a finite N-by-3 H/A/D array, and reference is N H/A/D/X
    labels. Every block is a nonoverlapping half-open integer interval. Values
    outside blocks and every original non-live label are returned unchanged.
    """
    reference = np.asarray(reference)
    if reference.ndim != 1 or not np.isin(reference, ["H", "A", "D", "X"]).all():
        raise ValueError("Expected a one-dimensional H/A/D/X reference")
    live_probability = conditional_live_probability(probability)
    if len(live_probability) != len(reference):
        raise ValueError("Reference and probability rows must align")
    try:
        blocks = list(blocks)
    except TypeError as error:
        raise ValueError("Expected decode intervals") from error
    visited = np.zeros(len(reference), dtype=bool)
    checked_blocks = []
    for block in blocks:
        try:
            start, end = block
        except (TypeError, ValueError) as error:
            raise ValueError("Each decode block needs two integer endpoints") from error
        if (not _integer(start) or not _integer(end)
                or not 0 <= start < end <= len(reference)
                or visited[start:end].any()):
            raise ValueError("Invalid or overlapping decode blocks")
        start, end = int(start), int(end)
        visited[start:end] = True
        checked_blocks.append((start, end))

    result = reference.copy()
    live = np.isin(reference, ["H", "A"])
    for start, end in checked_blocks:
        edges = np.flatnonzero(np.diff(np.r_[False, live[start:end], False]))
        for begin, stop in zip(edges[::2], edges[1::2]):
            left, right = start+int(begin), start+int(stop)
            result[left:right] = _align_component(reference[left:right],
                                                  live_probability[left:right])
    np.testing.assert_array_equal(result[~live | ~visited], reference[~live | ~visited])
    return result
