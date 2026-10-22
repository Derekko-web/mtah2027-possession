"""Shared broadcast-head inference with confidence coverage and reference gates."""

import numpy as np
import pandas as pd

from boundary_refinement import runs
from control_refinement import contact_support,smooth
from stoppage_consensus import decode_stoppage_consensus
from transition_penalty import decode
from xgboost_baseline import CLASSES


def broadcast_contact_support(features):
    distances = pd.DataFrame({prefix+"distance_1":features[prefix+"bc_same_distance_1"] for prefix in ("h__","a__")})
    return contact_support(distances)


def short_broadcast_proposals(reference,probability,support,seen):
    reference,probability = np.asarray(reference),np.asarray(probability,dtype=float)
    n = len(reference)
    if probability.ndim != 2 or probability.shape not in ((n,2),(n,3)) or np.asarray(support).shape != (n,2) or np.asarray(seen).shape != (n,):
        raise ValueError("Broadcast proposal inputs must align")
    if not np.isin(reference,[0,1,2]).all() or not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(axis=1),1):
        raise ValueError("Invalid states or broadcast probabilities")
    total = probability[:,:2].sum(axis=1,keepdims=True)
    owner = np.divide(probability[:,:2],total,out=np.full((n,2),.5),where=total > 0)
    p = smooth(owner,5)
    result = reference.copy()
    for start,end,team in runs(decode(p)):
        if not 8 <= end-start <= 125 or start == 0 or end == n or np.any(reference[start:end] == 2): continue
        if p[start:end,team].mean() < .65 or np.mean(seen[start:end]) < .5 or support[start:end,team].sum() < 3: continue
        before,after = reference[max(0,start-13):start],reference[end:min(n,end+13)]
        if np.mean(before == 1-team) <= .5 or np.mean(after == 1-team) <= .5: continue
        result[start:end] = team
    return result


def decode_broadcast_consensus(current,previous,fast,heads,probability,features,raw,blocks,joint,mode):
    n = len(raw)
    if len(features) != n or np.asarray(probability).shape != (n,3 if joint else 2):
        raise ValueError("Broadcast inference arrays must align")
    seen = features.bc_ball_seen.to_numpy() > 0
    if mode == "replace_head":
        p = np.where(seen[:,None],probability,current) if joint else current
        f = fast if joint else np.where(seen[:,None],probability,fast)
        return decode_stoppage_consensus(p,previous,f,heads,raw,blocks)
    if mode != "short_proposals": raise ValueError("Unknown broadcast decoder")
    result = decode_stoppage_consensus(current,previous,fast,heads,raw,blocks)
    reference = result.copy()
    support = broadcast_contact_support(features)
    for start,end in blocks:
        states = np.array([np.flatnonzero(CLASSES == x)[0] for x in reference[start:end]])
        result[start:end] = CLASSES[short_broadcast_proposals(states,probability[start:end],support[start:end],seen[start:end])]
    if not np.array_equal(result == "D",reference == "D"):
        raise AssertionError("Broadcast short proposals changed D")
    return result
