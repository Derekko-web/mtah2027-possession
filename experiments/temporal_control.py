"""Short control memory and ordered observations, rather than long-window means."""

import numpy as np
import pandas as pd


def contact_memory(base):
    result = {}
    for own, other in (("h__", "a__"), ("a__", "h__")):
        contact = (base[own+"distance_1"] < 2) & (base[other+"distance_1"]-base[own+"distance_1"] > .5)
        for frames in (25, 75, 125):
            result[own+f"contact_fraction_past_{frames}"] = contact.astype(float).rolling(frames, min_periods=1).mean()
        known = np.where(contact, np.arange(len(base)), np.nan)
        last = pd.Series(known).ffill().to_numpy()
        age = np.arange(len(base))-last
        result[own+"recent_contact_decay"] = np.where(age <= 125, np.exp(-age/25), 0)
        following = pd.Series(known)[::-1].ffill()[::-1].to_numpy()
        wait = following-np.arange(len(base))
        result[own+"next_contact_decay"] = np.where(wait <= 25, np.exp(-wait/25), 0)
    return pd.DataFrame(result).astype(np.float32)


def temporal_taps(base):
    columns = ["ball_seen", "ball_displacement_1s", "h__distance_1", "a__distance_1",
               "h__control_share", "a__control_share", "h__nearest_speed", "a__nearest_speed"]
    result = {}
    for name in columns:
        for lag in (-50, -25, -5, 5, 25, 50):
            # Positive lag reads past observations; negative reads the future.
            result[name+f"_lag_{lag}"] = base[name].shift(lag)
    return pd.DataFrame(result).astype(np.float32)
