"""Shared six-model inference: anchored ownership and complementary weak D heads."""

import numpy as np

from control_refinement import compose, contact_support, control_evidence, smooth
from evaluate_physics_context import apply_context
from renewal_control import anchored_emissions
from stoppage_hysteresis import hysteresis_dead
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES


def stoppage_emissions(reference, heads, visibility, moving):
    if np.asarray(heads).shape != (len(reference), 2):
        raise ValueError("Expected two aligned D heads")
    dead = reference[:, 2].copy()
    for index in (0, 1):
        head = smooth(heads[:, index:index+1], 25)[:, 0]
        prior = hysteresis_dead(head, visibility, moving, .5)
        dead = np.maximum(dead, prior)
    total = reference[:, :2].sum(axis=1, keepdims=True)
    owner = np.divide(reference[:, :2], total, out=np.full((len(reference), 2), .5), where=total > 0)
    return compose(dead, owner)


def decode_stoppage_consensus(current, previous, fast, heads, raw, blocks, augment=True):
    n = len(raw)
    for value, width in ((current, 3), (previous, 3), (fast, 2), (heads, 2)):
        if np.asarray(value).shape != (n, width):
            raise ValueError("All probabilities and observations must align")
    side = gap_sides(raw)
    support, evidence = contact_support(raw), control_evidence(raw)
    visibility = raw.ball_seen.rolling(51, center=True, min_periods=1).mean().to_numpy()
    result = np.full(n, "X", dtype="<U1")
    for start, end in blocks:
        if not 0 <= start < end <= n or np.any(result[start:end] != "X"):
            raise ValueError("Invalid or overlapping decode blocks")
        for value in (current, previous, fast):
            block = value[start:end]
            if not np.isfinite(block).all() or (block < 0).any() or not np.allclose(block.sum(axis=1), 1):
                raise ValueError("Invalid block probabilities")
        if not np.isfinite(heads[start:end]).all() or (heads[start:end] < 0).any() or (heads[start:end] > 1).any():
            raise ValueError("Invalid D probabilities")
        sides = side[start:end]
        p = apply_context(current[start:end], sides, np.zeros(end-start), False)
        q = apply_context(previous[start:end], sides, np.zeros(end-start), False)
        live = apply_context(np.column_stack((fast[start:end], np.zeros(end-start))), sides,
                             np.zeros(end-start), False)[:, :2]
        live /= live.sum(axis=1, keepdims=True)
        emission = anchored_emissions(p, q, live, support[start:end], evidence[start:end])
        if augment:
            emission = stoppage_emissions(emission, heads[start:end], visibility[start:end], evidence[start:end].max(axis=1))
        result[start:end] = CLASSES[decode(emission)]
    return result
