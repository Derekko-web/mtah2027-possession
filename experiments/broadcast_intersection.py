"""Intersect physical-source and learned control; permit legitimate D closure."""

import numpy as np

from boundary_refinement import runs
from control_refinement import smooth
from transition_penalty import decode


def intersect_broadcast(reference,learned,features,intersection,return_details=False):
    reference,learned = np.asarray(reference),np.asarray(learned)
    n = len(reference)
    if reference.shape != learned.shape or reference.ndim != 1 or len(features) != n or not np.isin(reference,[0,1,2]).all() or not np.isin(learned,[0,1,2]).all():
        raise ValueError("Expected aligned reference, learned states and features")
    shares = features[[prefix+"bc_same_control_share" for prefix in ("h__","a__")]].to_numpy(dtype=float)
    distances = features[[prefix+"bc_same_distance_1" for prefix in ("h__","a__")]].to_numpy(dtype=float)
    observed = (features.bc_ball_seen.to_numpy() > 0) & np.isfinite(shares).all(axis=1) & np.isfinite(distances).all(axis=1)
    contacts = (distances < 2) & (distances[:,::-1]-distances > .5)
    result,details = reference.copy(),[]
    spans = runs(reference)
    edges = np.flatnonzero(np.diff(np.r_[False,observed,False]))
    for left,right in zip(edges[::2],edges[1::2]):
        p = smooth(shares[left:right],5)
        for a,b,team in runs(decode(p)):
            if intersection:
                agree = learned[left+a:left+b] == team
                changes = np.flatnonzero(np.diff(np.r_[False,agree,False]))
                proposals = [(left+a+c,left+a+d) for c,d in zip(changes[::2],changes[1::2])]
            else: proposals = [(left+a,left+b)]
            for start,end in proposals:
                reason = None
                if not 8 <= end-start <= 125 or start == 0 or end == n: reason = "duration_or_edge"
                elif p[start-left:end-left,team].mean() < .65: reason = "source_confidence"
                elif contacts[start:end,team].sum() < 3: reason = "contacts"
                elif np.any(reference[start:end] == 2): reason = "reference_dead"
                elif np.mean(learned[start:end] == team) < .5: reason = "learned_agreement"
                elif any(owner == 1-team and 8 <= finish-begin <= 250 and begin <= start and end <= finish for begin,finish,owner in spans):
                    reason = "protected_reference_island"
                else:
                    before,after = reference[max(0,start-13):start],reference[end:min(n,end+13)]
                    opposite = [np.mean(side == 1-team) > .5 for side in (before,after)]
                    closed = [any(np.mean(side == owner) > .5 for owner in (0,1,2)) for side in (before,after)]
                    if not any(opposite) or not all(closed): reason = "closure"
                if reason is None: result[start:end] = team
                details.append(dict(start=int(start),end_exclusive=int(end),team=int(team),accepted=reason is None,reason=reason))
    if not np.array_equal(result == 2,reference == 2): raise AssertionError("Intersection changed D")
    return (result,details) if return_details else result
