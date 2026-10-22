"""Shared seven-model reference and new pair-motion research inference."""

import numpy as np

from broadcast_refinement import decode_broadcast_closure
from pair_motion_inference import head_observed,live_motion,motion_events,motion_source
from stoppage_consensus import decode_stoppage_consensus
from xgboost_baseline import CLASSES


def codes(labels): return np.array([np.flatnonzero(CLASSES == value)[0] for value in labels])


def decode_pair_candidate(current,previous,fast,heads,bc_probability,new_probability,bc_features,info,raw,blocks,head,mode):
    observed = head_observed(bc_features,info,head)
    if mode == "replace_current":
        replacement = np.where(observed[:,None],new_probability,current)
        return decode_broadcast_closure(replacement,previous,fast,heads,bc_probability,bc_features,raw,blocks)
    reference = decode_broadcast_closure(current,previous,fast,heads,bc_probability,bc_features,raw,blocks)
    base = decode_stoppage_consensus(current,previous,fast,heads,raw,blocks)
    conditional = new_probability[:,:2]/new_probability[:,:2].sum(axis=1,keepdims=True)
    result = reference.copy()
    kernel,distance,source_observed = motion_source(info,head)
    for start,end in blocks:
        r = codes(reference[start:end])
        if mode == "live_replace": prediction = live_motion(r,conditional[start:end],observed[start:end])
        elif mode == "motion_events": prediction = motion_events(r,codes(base[start:end]),conditional[start:end],kernel[start:end],distance[start:end],source_observed[start:end])
        else: raise ValueError("Unknown pair-motion inference")
        result[start:end] = CLASSES[prediction]
    if not np.array_equal(result == "D",reference == "D"): raise AssertionError("Ownership motion changedD")
    return result
