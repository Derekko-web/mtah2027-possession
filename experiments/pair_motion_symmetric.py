"""Team-symmetric event overlap resolution; retained v1 comparison is historical."""

import numpy as np

from boundary_refinement import runs


def symmetric_motion_events(reference,base,probability,kernel,distance,observed,return_details=False):
    reference,base = np.asarray(reference),np.asarray(base)
    n = len(reference)
    if base.shape != reference.shape or probability.shape != (n,2) or kernel.shape != (n,2) or distance.shape != (n,2) or len(observed) != n:
        raise ValueError("Motion event arrays differ")
    if not np.array_equal(reference == 2,base == 2): raise ValueError("Source baseline D differs")
    strong = (kernel >= .2)&(kernel-kernel[:,::-1] >= .1)&observed[:,None]
    contacts = (distance < 2)&(distance[:,::-1]-distance > .5)&observed[:,None]
    changed = []
    for start,end,team in runs(np.where(reference != base,reference,-1)):
        if team in (0,1): changed.append((start,end,team))
    accepted,details = [],[]
    episodes = np.flatnonzero(np.diff(np.r_[False,observed,False]))
    for team in (0,1):
        for left,right in zip(episodes[::2],episodes[1::2]):
            active = strong[left:right,team].copy()
            gaps = np.flatnonzero(np.diff(np.r_[True,active,True]))
            for a,b in zip(gaps[::2],gaps[1::2]):
                if 0 < a and b < len(active) and b-a <= 5 and not strong[left+a:left+b,1-team].any(): active[a:b] = True
            edges = np.flatnonzero(np.diff(np.r_[False,active,False]))
            for a,b in zip(edges[::2],edges[1::2]):
                core_start,core_end = left+a,left+b
                start,end = max(left,core_start-5),min(right,core_end+5)
                refinement = any(owner == team and max(begin,core_start) < min(finish,core_end) for begin,finish,owner in changed)
                gate_reference = base if refinement else reference
                spans = runs(gate_reference)
                reason = None
                if not 8 <= end-start <= 125 or start == 0 or end == n: reason = "duration_or_edge"
                elif contacts[start:end,team].sum() < 3: reason = "contacts"
                elif probability[start:end,team].mean() < .65: reason = "learned_confidence"
                elif (reference[start:end] == 2).any(): reason = "reference_dead"
                elif any(owner == 1-team and 8 <= finish-begin <= 250 and begin <= start and end <= finish for begin,finish,owner in spans): reason = "protected_island"
                else:
                    flanks = (gate_reference[max(0,start-13):start],gate_reference[end:min(n,end+13)])
                    opposite = [np.mean(side == 1-team) > .5 for side in flanks]
                    closed = [any(np.mean(side == owner) > .5 for owner in (0,1,2)) for side in flanks]
                    if not any(opposite) or not all(closed): reason = "closure"
                if reason is None: accepted.append((start,end,team,core_start,core_end))
                details.append(dict(start=int(start),end_exclusive=int(end),team=team,accepted=reason is None,reason=reason))
    result = reference.copy()
    # All eligibility reads the unchanged reference, and all restorations precede writes.
    for _,_,team,core_start,core_end in accepted:
        for begin,finish,owner in changed:
            if owner == team and max(begin,core_start) < min(finish,core_end): result[begin:finish] = base[begin:finish]
    scores = np.full((n,2),-np.inf)
    for start,end,team,_,_ in accepted:
        weight = probability[start:end,team]*np.nan_to_num(kernel[start:end,team],nan=0.)
        scores[start:end,team] = np.maximum(scores[start:end,team],weight)
    claimed = np.isfinite(scores).any(axis=1)
    result[claimed] = scores[claimed].argmax(axis=1)
    tied = np.isfinite(scores).all(axis=1)&np.isclose(scores[:,0],scores[:,1],atol=1e-12,rtol=0)
    result[tied] = reference[tied]
    if not np.array_equal(result == 2,reference == 2): raise AssertionError("Motion event alteredD")
    return (result,details) if return_details else result
