"""Prospective ownership-only composition and learned source-extent shortening."""

import numpy as np

from boundary_refinement import runs
from control_refinement import smooth
from motion_dead_trim import decode_motion_dead_trim
from pair_candidate import codes
from stoppage_consensus import decode_stoppage_consensus
from transition_penalty import decode
from xgboost_baseline import CLASSES


def refine_source_extent(reference, baseline, probability, observed):
    reference, baseline = np.asarray(reference), np.asarray(baseline)
    probability, observed = np.asarray(probability), np.asarray(observed)
    if probability.shape != (len(reference), 2) or baseline.shape != reference.shape or observed.shape != reference.shape:
        raise ValueError('Source-extent inputs must align')
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1):
        raise ValueError('Invalid conditional graph probabilities')
    p = smooth(probability, 5)
    events = [(start, end, team) for start, end, team in runs(decode(p)) if 8 <= end-start <= 125]
    changed = (reference < 2) & (baseline < 2) & (reference != baseline)
    components = np.where(changed, reference, 2)
    result = reference.copy()
    for start, end, team in runs(components):
        if team == 2:
            continue
        candidates = []
        for left, right, owner in events:
            lo, hi = max(left, start), min(right, end)
            if owner != team or not 8 <= hi-lo <= 125 or not observed[lo:hi].all():
                continue
            confidence = p[lo:hi, team].mean()
            if confidence >= .65:
                candidates.append((hi-lo, confidence, lo, hi))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
            continue
        _, _, lo, hi = candidates[0]
        result[start:end] = baseline[start:end]
        result[lo:hi] = team
    if not np.array_equal(result == 2, reference == 2):
        raise AssertionError('Source extent changed D')
    if np.any((result != reference) & ~changed):
        raise AssertionError('Source extent changed an original ownership decision')
    return result


def decode_graph_conditional(arrays, probability, bc, motion_info, info, raw, blocks, mode):
    observed = (info.tg_seen.to_numpy() == 1) | (info.bg_seen.to_numpy() == 1)
    conditional = probability[:, :2]/probability[:, :2].sum(axis=1, keepdims=True)
    if mode == 'conditional_current':
        replacement = arrays[0].copy()
        replacement[observed, :2] = conditional[observed]*(1-arrays[0][observed, 2, None])
        return decode_motion_dead_trim(replacement, *arrays[1:], bc, motion_info, raw, blocks)
    if mode != 'source_extent':
        raise ValueError('Unknown graph conditional decoder')
    reference = decode_motion_dead_trim(*arrays, bc, motion_info, raw, blocks)
    baseline = decode_stoppage_consensus(*arrays[:4], raw, blocks)
    result = reference.copy()
    for start, end in blocks:
        value = refine_source_extent(codes(reference[start:end]), codes(baseline[start:end]),
                                     conditional[start:end], observed[start:end])
        result[start:end] = CLASSES[value]
    return result
