"""Bounded label-free contact chronology and receiver geometry."""

import numpy as np
import pandas as pd

from games import GAME, GAMES, require
from starter.baseline import team_of_each_track


def contact_chronology(base):
    result = {}
    frame = np.arange(len(base))
    for team, other in (("h__", "a__"), ("a__", "h__")):
        for radius in (1., 2.):
            touch = ((base[team+"distance_1"] < radius) &
                     (base[other+"distance_1"]-base[team+"distance_1"] > .5))
            known = pd.Series(np.where(touch, frame, np.nan))
            age = frame-known.ffill().to_numpy()
            wait = known.bfill().to_numpy()-frame
            suffix = str(int(radius))
            for horizon in (25, 75, 125):
                past = np.where(age <= horizon, np.exp(-age/25), 0)
                future = np.where(wait <= horizon, np.exp(-wait/25), 0)
                result[team+f"touch{suffix}_past_{horizon}"] = past
                result[team+f"touch{suffix}_future_{horizon}"] = future
                result[team+f"touch{suffix}_bracket_{horizon}"] = np.sqrt(past*future)
            for width in (13, 25, 75):
                result[team+f"touch{suffix}_past_fraction_{width}"] = touch.astype(float).rolling(width, min_periods=1).mean()
                result[team+f"touch{suffix}_future_fraction_{width}"] = touch.astype(float).iloc[::-1].rolling(
                    width, min_periods=1).mean().iloc[::-1]
    return pd.DataFrame(result).astype(np.float32)


def receiver_geometry(positions, ball, teams):
    """Distances to player sets, avoiding assumptions of persistent track slots."""
    n = len(ball)
    velocity = np.full_like(ball, np.nan)
    velocity[5:] = (ball[5:]-ball[:-5])*5
    velocity[np.linalg.norm(velocity, axis=1) > 50] = np.nan
    result = {}
    for code, prefix in ((0, "h__"), (1, "a__")):
        xy = positions[:, teams == code]
        offset = xy-ball[:, None, :]
        square_speed = np.sum(velocity**2, axis=1)
        time = np.clip(np.sum(offset*velocity[:, None, :], axis=2)/np.maximum(square_speed[:, None], .01), 0, 1)
        miss = np.linalg.norm(offset-time[:, :, None]*velocity[:, None, :], axis=2)
        result[prefix+"receiver_corridor_distance"] = finite_min(miss)
        for step in (5, 13, 25):
            projected = ball+velocity*(step/25)
            result[prefix+f"receiver_projected_distance_{step}"] = finite_min(np.linalg.norm(xy-projected[:, None, :], axis=2))
            future = np.full_like(ball, np.nan)
            future[:-step] = ball[step:]
            result[prefix+f"receiver_future_ball_distance_{step}"] = finite_min(np.linalg.norm(xy-future[:, None, :], axis=2))
    return pd.DataFrame(result).astype(np.float32)


def finite_min(values):
    minimum = np.min(np.where(np.isfinite(values), values, np.inf), axis=1)
    minimum[~np.isfinite(minimum)] = np.nan
    return minimum


def build_receiver_geometry(n_frames):
    metadata, tracking = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(metadata)
    pitch = np.array(GAMES[GAME]["pitch"])
    positions = np.full((n_frames, len(teams), 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    count = 0
    with open(tracking) as handle:
        for frame, line in enumerate(handle):
            if frame >= n_frames:
                break
            frame_id, players, ball_raw = line.strip().split(":")
            if int(frame_id) != frame+1:
                raise ValueError("Tracking frame mismatch")
            players = np.fromstring(players.replace(";", ","), sep=",").reshape(-1, 3)
            if len(players) != len(teams):
                raise ValueError("Tracking slot mismatch")
            positions[frame] = players[:, :2]*pitch
            ball[frame] = np.fromstring(ball_raw, sep=",")[:2]*pitch
            count += 1
    if count != n_frames:
        raise ValueError("Tracking ended early")
    return receiver_geometry(positions, ball, teams)
