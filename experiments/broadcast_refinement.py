"""Shared selected-reference plus source/model closure; unchanged D decisions."""

import numpy as np

from broadcast_consensus import decode_broadcast_consensus
from broadcast_intersection import intersect_broadcast
from stoppage_consensus import decode_stoppage_consensus
from xgboost_baseline import CLASSES


def decode_broadcast_closure(current,previous,fast,heads,broadcast_probability,features,raw,blocks):
    reference = decode_stoppage_consensus(current,previous,fast,heads,raw,blocks)
    learned = decode_broadcast_consensus(current,previous,fast,heads,broadcast_probability,features,raw,blocks,True,"replace_head")
    result = reference.copy()
    for start,end in blocks:
        r = np.array([np.flatnonzero(CLASSES == x)[0] for x in reference[start:end]])
        p = np.array([np.flatnonzero(CLASSES == x)[0] for x in learned[start:end]])
        result[start:end] = CLASSES[intersect_broadcast(r,p,features.iloc[start:end],False)]
    if not np.array_equal(result == "D",reference == "D"):
        raise AssertionError("Broadcast closure changed reference D")
    return result
