"""Team-invariant physical and automatic restart-timing inputs for dead/live heads.

Features read no labels. Restart timing is evidence rather than a copied answer.
The event kernels are zero farther than ten seconds from a phase boundary. Long
phase membership is an input supplied by the automatic tagger, as in rich_features.
"""

import numpy as np
import pandas as pd

from robust_motion import build_motion_inputs
from starter.load_smart_tagging import load as load_tags


def restart_features(n_frames, events):
    groups = {"all": []}
    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        kind = (event["tags"].get("Type") or "unknown").strip().lower().replace(" ", "_")
        groups.setdefault(kind, []).append(event)
        groups["all"].append(event)
    # Fixed feature schema on partial and full-match inputs.
    for kind in ("throw-in", "corner_kick", "goal_kick", "free_kick",
                 "kick_off_(after_goal)", "kick_off_(start)"):
        groups.setdefault(kind, [])
    result = {}
    for kind, pieces in sorted(groups.items()):
        prefix = "restart_"+kind
        active = np.zeros(n_frames)
        progress = np.zeros(n_frames)
        kernels = {name: np.zeros(n_frames) for name in
                   ("before_start", "after_start", "before_end", "after_end")}
        for event in pieces:
            start, end = int(event["start_frame"]), int(event["end_frame"])
            left, right = max(0, start), min(n_frames, end)
            if right > left:
                active[left:right] = 1
                progress[left:right] = np.maximum(progress[left:right],
                                                 (np.arange(left, right)-start)/max(end-start, 1))
            for boundary, frame in (("start", start), ("end", end)):
                for direction, sign in (("before", -1), ("after", 1)):
                    ix = frame+sign*np.arange(251)
                    distance = np.arange(251)/25
                    valid = (ix >= 0) & (ix < n_frames)
                    ix = ix[valid]
                    value = np.exp(-distance[valid]/3)
                    name = direction+"_"+boundary
                    kernels[name][ix] = np.maximum(kernels[name][ix], value)
        result[prefix+"_active"] = active
        result[prefix+"_progress"] = progress
        result.update({prefix+"_"+key: value for key, value in kernels.items()})
    return pd.DataFrame(result).astype(np.float32)


def physical_features(base, robust):
    shared = base[[c for c in base if not c.startswith(("h__", "a__"))]].copy()
    # Remove directional/team slots by forming min/max symmetric summaries.
    for column in base:
        if not column.startswith("h__"):
            continue
        name = column[3:]
        pair = base[["h__"+name, "a__"+name]]
        shared[name+"_minimum"] = pair.min(axis=1)
        shared[name+"_maximum"] = pair.max(axis=1)
    return shared.join(robust)


def build_stoppage_features(base):
    robust, _ = build_motion_inputs(len(base))
    physical = physical_features(base, robust)
    return physical, restart_features(len(base), load_tags())
