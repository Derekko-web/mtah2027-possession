"""Consensus-supported short ownership proposals with the retained dead/live model.

All candidate intervals are model predictions. No true labels, frame numbers or
hand-written boundaries enter decoding. Future inputs are allowed offline.
"""

import numpy as np

from control_refinement import compose, contact_support, control_evidence, smooth
from evaluate_physics_context import apply_context
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES

CONFIG = dict(threshold=.65,strength=.75)


def consensus_team(current, previous, fast, support, threshold=.65):
    total=current[:,:2].sum(axis=1,keepdims=True)
    reference=np.divide(current[:,:2],total,out=np.full((len(current),2),.5),where=total>0)
    team=reference.copy()
    proposal = decode(previous)
    edges = np.r_[0, np.flatnonzero(proposal[1:] != proposal[:-1])+1, len(proposal)]
    for start, end in zip(edges[:-1], edges[1:]):
        owner = proposal[start]
        if owner == 2 or not 8 <= end-start <= 125:
            continue
        if fast[start:end, owner].mean() < threshold or support[start:end, owner].sum() < 3:
            continue
        if start == 0 or end == len(team):
            continue
        before = reference[max(0,start-13):start].mean(axis=0)
        after = reference[end:min(len(team),end+13)].mean(axis=0)
        if before[1-owner] <= .5 or after[1-owner] <= .5:
            continue
        team[start:end] = fast[start:end]
    return team


def decode_consensus(current, previous, fast, raw, blocks, threshold=.65,strength=.75):
    n = len(raw)
    for value,columns in ((current,3),(previous,3),(fast,2)):
        if value.shape != (n,columns):
            raise ValueError("Probabilities and observations must align")
    if not 0.5 < threshold <= 1 or not 0 <= strength <= 1:
        raise ValueError("Invalid consensus thresholds")
    side = gap_sides(raw)
    support,evidence = contact_support(raw),control_evidence(raw)
    result = np.full(n,"X",dtype="<U1")
    for start,end in blocks:
        if not 0 <= start < end <= n or np.any(result[start:end]!="X"):
            raise ValueError("Invalid or overlapping blocks")
        for values in (current,previous,fast):
            block = values[start:end]
            if not np.isfinite(block).all() or (block<0).any() or not np.allclose(block.sum(axis=1),1):
                raise ValueError("Invalid block probabilities")
        sides = side[start:end]
        p = apply_context(current[start:end],sides,np.zeros(end-start),False)
        q = apply_context(previous[start:end],sides,np.zeros(end-start),False)
        live = apply_context(np.column_stack((fast[start:end],np.zeros(end-start))),sides,
                             np.zeros(end-start),False)[:,:2]
        live /= live.sum(axis=1,keepdims=True)
        p,q,live = smooth(p,25),smooth(q,25),smooth(live,5)
        team = consensus_team(p,q,live,support[start:end],threshold)
        dead = p[:,2]*(1-strength*evidence[start:end].max(axis=1))
        result[start:end] = CLASSES[decode(compose(dead,team))]
    return result
