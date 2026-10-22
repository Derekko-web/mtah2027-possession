"""Collective activity persistence; no identities, ball visibility or labels."""

import numpy as np
import pandas as pd


def tempo_features(raw):
    speed = raw[["h__speed_median", "a__speed_median"]].mean(axis=1)
    moving = raw[["h__moving_fraction", "a__moving_fraction"]].mean(axis=1)
    base = pd.DataFrame({"collective_speed": speed, "collective_moving": moving,
                         "fastest_team_speed": raw[["h__speed_median", "a__speed_median"]].max(axis=1),
                         "slowest_team_speed": raw[["h__speed_median", "a__speed_median"]].min(axis=1)})
    result = {}
    for name, column in base.items():
        for seconds in (1, 3, 5):
            window = column.rolling(seconds*25+1, center=True, min_periods=seconds*25+1)
            result[f"{name}_mean_{seconds}s"] = window.mean()
            result[f"{name}_maximum_{seconds}s"] = window.max()
            result[f"{name}_minimum_{seconds}s"] = window.min()
    return base.join(pd.DataFrame(result)).astype(np.float32)
