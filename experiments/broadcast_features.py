"""Same/cross-camera possession observations on the tactical timeline, no labels."""

import numpy as np
import pandas as pd

from broadcast_alignment import transform
from contact_dynamics import geometry_features


def aligned_features(tactical,broadcast,mapping):
    n = len(tactical["positions"])
    frames = mapping["frames"]
    valid = frames >= 0
    source_frames = np.maximum(frames,0)
    bxy = broadcast["positions"][source_frames].copy()
    bball = broadcast["ball"][source_frames].copy()
    bspeed = broadcast["speeds"][source_frames].copy()
    for orientation in range(4):
        use = valid & (mapping["orientation"] == orientation)
        bxy[use] = transform(bxy[use],orientation)
        bball[use] = transform(bball[use],orientation)
    bxy[~valid],bball[~valid],bspeed[~valid] = np.nan,np.nan,np.nan
    result = {}
    for name,xy,speed,teams in (("same",bxy,bspeed,broadcast["teams"]),
                                ("cross",tactical["positions"],tactical["speeds"],tactical["teams"])):
        geometry = geometry_features(xy,speed,bball,bball,teams).drop(columns="clean_ball_correction")
        geometry.columns = [c.replace("clean_",f"bc_{name}_") for c in geometry.columns]
        result[name] = geometry
    quality = pd.DataFrame(dict(bc_matched=valid.astype(float),bc_ball_seen=np.isfinite(bball).all(axis=1).astype(float),
                                bc_formation_cost=mapping["cost"],bc_far_margin=np.clip(mapping["margin"],0,10)))
    summaries = {}
    for prefix in ("h__","a__"):
        for name in ("distance_1","control_share"):
            column = result["same"][prefix+"bc_same_"+name]
            for window in (25,75,125):
                summaries[prefix+f"bc_{name}_past_{window}"] = column.rolling(window,min_periods=1).mean()
                summaries[prefix+f"bc_{name}_future_{window}"] = column.iloc[::-1].rolling(window,min_periods=1).mean().iloc[::-1]
    summaries = pd.DataFrame(summaries)
    same = quality.join(result["same"]).astype(np.float32)
    combined = same.join(result["cross"]).join(summaries).astype(np.float32)
    return same,combined
