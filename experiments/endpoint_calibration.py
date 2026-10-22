"""Nested training cases and bounded learned endpoint calibration; no frame rules."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from boundary_refinement import runs


def contiguous_components(frames):
    frames = np.asarray(frames, dtype=int)
    if not len(frames):
        return []
    if not np.all(np.diff(frames) > 0):
        raise ValueError("Frames must be sorted and unique")
    cuts = np.r_[0, np.flatnonzero(np.diff(frames) != 1)+1, len(frames)]
    return [(int(frames[a]), int(frames[b-1])+1) for a,b in zip(cuts[:-1], cuts[1:])]


def nested_folds(outer_train):
    outer_train = np.asarray(outer_train, dtype=int)
    if not len(outer_train) or not np.all(np.diff(outer_train) > 0) or outer_train.min() < 11185 or outer_train.max() > 18684:
        raise ValueError("Outer training must contain sorted public indices only")
    result = []
    for block, (start, end) in enumerate(((11185,13685), (13685,16185), (16185,18685)), 1):
        validation = outer_train[(outer_train >= start) & (outer_train < end)]
        train = outer_train[(outer_train < start-250) | (outer_train >= end+250)]
        result.append((block, train, validation))
    return result


def eligible_islands(states):
    spans = runs(states)
    indices = [i for i,(start,end,owner) in enumerate(spans)
               if owner != 2 and 8 <= end-start <= 125 and 0 < i < len(spans)-1
               and spans[i-1][2] == 1-owner and spans[i+1][2] == 1-owner]
    return spans, indices


def timing_examples(predicted, truth, offset=0):
    """Match only within one contiguous eligible inner validation component."""
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if predicted.shape != truth.shape or predicted.ndim != 1 or not np.isin(predicted, [0,1,2]).all() or not np.isin(truth, [0,1,2]).all():
        raise ValueError("Expected aligned H/A/D states")
    spans, indices = eligible_islands(predicted)
    proposals = [spans[i] for i in indices]
    # Edge-touching true runs may be clipped by an outer/inner exclusion.
    targets = [span for span in runs(truth)[1:-1] if span[2] != 2]
    if not proposals or not targets:
        return []
    iou = np.zeros((len(proposals), len(targets)))
    for i,(start,end,owner) in enumerate(proposals):
        for j,(left,right,label) in enumerate(targets):
            intersection = max(0,min(end,right)-max(start,left))
            if owner == label and intersection:
                iou[i,j] = intersection/(end-start+right-left-intersection)
    rows, columns = linear_sum_assignment(-iou)
    result = []
    for i,j in zip(rows,columns):
        if iou[i,j] < .3:
            continue
        start,end,owner = proposals[i]
        left,right,_ = targets[j]
        result.append(dict(predicted_start=offset+start, predicted_end_exclusive=offset+end,
                           true_start=offset+left, true_end_exclusive=offset+right, team=owner,
                           iou=float(iou[i,j]), start_error=int(left-start), end_error=int(right-end)))
    return result


def estimate_correction(examples, minimum):
    if minimum not in (1,2):
        raise ValueError("Frozen minimum is one or two examples")
    if len(examples) < minimum:
        return dict(start=0., end=0., count=len(examples), sufficient=False)
    start = float(np.median([x["start_error"] for x in examples]))
    end = float(np.median([x["end_error"] for x in examples]))
    return dict(start=float(np.clip(start,-25,25)), end=float(np.clip(end,-25,25)),
                count=len(examples), sufficient=True, raw_start_median=start, raw_end_median=end)


def calibrate_endpoints(states, correction, both):
    """Jointly satisfy learned timing preferences without changing D or topology."""
    states = np.asarray(states)
    if states.ndim != 1 or not np.isin(states, [0,1,2]).all():
        raise ValueError("Expected H/A/D states")
    if not all(np.isfinite(correction[k]) and abs(correction[k]) <= 25 for k in ("start", "end")):
        raise ValueError("Timing correction must be finite and bounded")
    if not len(states) or (correction["end"] == 0 and (not both or correction["start"] == 0)):
        return states.copy()
    spans, indices = eligible_islands(states)
    eligible = set(indices)
    if not eligible:
        return states.copy()
    candidates, desired = [np.array([0])], [0.]
    for i,(_,end,_) in enumerate(spans[:-1],1):
        shifts = []
        if i-1 in eligible:
            shifts.append(correction["end"])
        if both and i in eligible:
            shifts.append(correction["start"])
        desired.append(end+float(np.mean(shifts)) if shifts else float(end))
        candidates.append(np.arange(max(1,end-25), min(len(states)-1,end+25)+1) if shifts else np.array([end]))
    candidates.append(np.array([len(states)]))
    desired.append(float(len(states)))
    costs, back = np.zeros(1), []
    for i,(_,original_end,_) in enumerate(spans):
        left,right = candidates[i], candidates[i+1]
        trial = np.broadcast_to(costs[:,None]+(right[None,:]-desired[i+1])**2, (len(left),len(right))).copy()
        trial += 1e-8*np.abs(right-original_end)[None,:]
        trial = np.where(right[None,:]-left[:,None] >= (8 if i in eligible else 1), trial, np.inf)
        previous = trial.argmin(axis=0)
        costs = trial[previous,np.arange(len(right))]
        back.append(previous)
    if not np.isfinite(costs[0]):
        raise ValueError("No feasible calibration path")
    cuts,index = [len(states)],0
    for i in range(len(spans)-1,-1,-1):
        index = back[i][index]
        cuts.append(int(candidates[i][index]))
    cuts.reverse()
    result = np.empty_like(states)
    for (start,end),(_,_,owner) in zip(zip(cuts[:-1],cuts[1:]),spans):
        result[start:end] = owner
    if not np.array_equal(result == 2, states == 2):
        raise AssertionError("Timing calibration changed D")
    return result
