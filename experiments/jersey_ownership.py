"""Confident local color-team geometry and bounded ownership-only inference."""

import numpy as np
import pandas as pd

from broadcast_intersection import intersect_broadcast
from contact_dynamics import geometry_features
from weak_ownership import ownership_prediction


def expand_team_probability(frames,probability,n):
    frames,probability = np.asarray(frames),np.asarray(probability)
    if probability.ndim != 3 or probability.shape[0] != len(frames) or probability.shape[2] != 2 or np.any(np.diff(frames) <= 0):
        raise ValueError("Expected ordered sampled player/team probabilities")
    result = np.full((n,probability.shape[1],2),np.nan)
    for index,frame in enumerate(frames):
        if 0 <= frame < n: result[frame] = probability[index]
        if index == len(frames)-1 or frames[index+1]-frame != 5: continue
        valid = np.isfinite(probability[index:index+2]).all(axis=(0,2))
        for step in range(1,5):
            if 0 <= frame+step < n:
                result[frame+step,valid] = probability[index,valid]*(1-step/5)+probability[index+1,valid]*(step/5)
    return result


def jersey_geometry(positions,speeds,ball,teams,sample_frames,probability):
    n,slots = positions.shape[:2]
    p = expand_team_probability(sample_frames,probability,n)
    safe = np.where(np.isfinite(p),p,-1)
    known = np.isfinite(p).all(axis=2) & (safe.max(axis=2) >= .8) & np.isfinite(positions).all(axis=2)
    owner = safe.argmax(axis=2)
    xy = np.concatenate([np.where((known & (owner == team))[:,:,None],positions,np.nan) for team in (0,1)],axis=1)
    sp = np.tile(speeds,(1,2))
    result = geometry_features(xy,sp,ball,ball,np.repeat([0,1],slots))
    result = result.rename(columns={name:name.replace("clean_","jc_") for name in result})
    result = result.drop(columns="jc_ball_correction")
    distance = result[[p+"jc_distance_1" for p in ("h__","a__")]].to_numpy()
    result["jc_ball_seen"] = np.isfinite(distance).all(axis=1).astype(float)
    observed = np.isfinite(positions).all(axis=2).sum(axis=1)
    result["jc_classified_fraction"] = known.sum(axis=1)/np.maximum(observed,1)
    result["jc_reassigned_fraction"] = (known & (owner != teams[None])).sum(axis=1)/np.maximum(known.sum(axis=1),1)
    extra = {}
    for prefix in ("h__","a__"):
        for column in ("jc_distance_1","jc_control_share"):
            for radius in (25,75,125):
                series = result[prefix+column]
                extra[prefix+column+f"_past{radius}"] = series.rolling(radius+1,min_periods=1).mean()
                extra[prefix+column+f"_future{radius}"] = series.iloc[::-1].rolling(radius+1,min_periods=1).mean().iloc[::-1]
    return result.join(pd.DataFrame(extra)).astype(np.float32)


def decode_jersey_ownership(reference,probability,features,mode):
    reference,probability = np.asarray(reference),np.asarray(probability,dtype=float)
    if probability.shape != (len(reference),2) or len(features) != len(reference): raise ValueError("Ownership arrays differ")
    observed = features.jc_ball_seen.to_numpy() > 0
    fallback = np.stack([reference == 0,reference == 1],axis=1).astype(float)
    fallback[reference == 2] = .5
    p = np.where(observed[:,None],probability,fallback)
    distance = features[[prefix+"jc_distance_1" for prefix in ("h__","a__")]].to_numpy()
    support = (distance < 2) & (distance[:,::-1]-distance > .5)
    replacement = ownership_prediction(reference,p,support,np.zeros_like(p),"replace_live")
    replacement[~observed] = reference[~observed]
    if mode == "replace_live": result = replacement
    elif mode == "short_proposals": result = ownership_prediction(reference,p,support,np.zeros_like(p),mode)
    elif mode == "source_closure":
        aliases = features.rename(columns={prefix+"jc_"+suffix:prefix+"bc_same_"+suffix
            for prefix in ("h__","a__") for suffix in ("distance_1","control_share")}).rename(columns={"jc_ball_seen":"bc_ball_seen"})
        result = intersect_broadcast(reference,replacement,aliases,False)
    else: raise ValueError("Unknown jersey decoder")
    result[~observed] = reference[~observed]
    if not np.array_equal(result == 2,reference == 2): raise AssertionError("Jersey inference changedD")
    return result
