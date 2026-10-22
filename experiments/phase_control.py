"""Translate automatic phase semantics into symmetric soft control evidence."""

import numpy as np
import pandas as pd

from starter.load_smart_tagging import load as load_tags

ATTACKING = {"build up", "progression", "ball in final third", "ball in the box",
             "creating chances", "attacking transition", "long ball", "players in the box"}
DEFENDING = {"defending in defensive third", "defending in middle third",
             "defending in attacking third", "defensive transition"}


def phase_control(n_frames, events=None):
    events = load_tags() if events is None else events
    storage = {p+name: np.zeros(n_frames) for p in ("h__", "a__")
               for name in ("attack_phase", "opponent_defending_phase", "restart_phase")}
    for event in events:
        code = (event["code"] or "").strip().lower()
        team = (event["tags"].get("Team") or "").strip().lower()
        prefix = "h__" if team == "barcelona" else "a__" if team == "atletic club" else None
        if prefix is None:
            continue
        if code in ATTACKING:
            name = "attack_phase"
        elif code in DEFENDING:
            prefix = "a__" if prefix == "h__" else "h__"
            name = "opponent_defending_phase"
        elif code == "set pieces":
            name = "restart_phase"
        else:
            continue
        start, stop = max(0, int(event["start_frame"])), min(n_frames, int(event["end_frame"]))
        if stop > start:
            storage[prefix+name][start:stop] = 1
    result = pd.DataFrame(storage)
    for prefix in ("h__", "a__"):
        result[prefix+"phase_control_support"] = result[[prefix+"attack_phase", prefix+"opponent_defending_phase"]].mean(axis=1)
    return result.astype(np.float32)
