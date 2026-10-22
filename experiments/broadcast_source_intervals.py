"""Conservative source-bounded ownership intervals, no possession target inputs."""

import numpy as np

from boundary_refinement import runs
from control_refinement import smooth
from transition_penalty import decode


def source_intervals(reference,learned,features,half_window,return_details=False):
    reference,learned = np.asarray(reference),np.asarray(learned)
    n = len(reference)
    if reference.shape != learned.shape or len(features) != n or reference.ndim != 1:
        raise ValueError("Source interval inputs must align")
    if not np.isin(reference,[0,1,2]).all() or not np.isin(learned,[0,1,2]).all() or half_window not in (2,5):
        raise ValueError("Invalid states or source averaging")
    observed = features.bc_ball_seen.to_numpy() > 0
    shares = features[[prefix+"bc_same_control_share" for prefix in ("h__","a__")]].to_numpy(dtype=float)
    distances = features[[prefix+"bc_same_distance_1" for prefix in ("h__","a__")]].to_numpy(dtype=float)
    observed &= np.isfinite(shares).all(axis=1) & np.isfinite(distances).all(axis=1)
    contacts = (distances < 2) & (distances[:,::-1]-distances > .5)
    result,details = reference.copy(),[]
    spans = runs(reference)
    edges = np.flatnonzero(np.diff(np.r_[False,observed,False]))
    for left,right in zip(edges[::2],edges[1::2]):
        probability = smooth(shares[left:right],half_window)
        for a,b,team in runs(decode(probability)):
            start,end = left+a,left+b
            reason = None
            if not 8 <= end-start <= 125 or start == 0 or end == n: reason = "duration_or_block_edge"
            elif probability[a:b,team].mean() < .65: reason = "source_confidence"
            elif contacts[start:end,team].sum() < 3: reason = "contacts"
            elif np.any(reference[start:end] == 2): reason = "reference_dead"
            elif np.mean(reference[max(0,start-13):start] == 1-team) <= .5 or np.mean(reference[end:min(n,end+13)] == 1-team) <= .5:
                reason = "opposite_flanks"
            elif np.mean(learned[start:end] == team) < .5: reason = "learned_agreement"
            elif any(owner == 1-team and 8 <= finish-begin <= 250 and begin <= start and end <= finish for begin,finish,owner in spans):
                reason = "protected_reference_island"
            if reason is None: result[start:end] = team
            details.append(dict(start=int(start),end_exclusive=int(end),team=int(team),accepted=reason is None,reason=reason))
    if not np.array_equal(result == 2,reference == 2): raise AssertionError("Source intervals changed D")
    return (result,details) if return_details else result
