"""Bounded motion inputs for two predeclared continuation experiments.

No labels, player identities or match clock are read. All coordinates are metres.
The largest observation radius is 50 frames (two seconds at the supplied 25 fps).
Missing observations remain missing; no trajectory is interpolated across a gap.
"""

import numpy as np
import pandas as pd

from games import GAME, GAMES, require
from starter.baseline import team_of_each_track

MAX_CONTEXT_FRAMES = 50


def stationarity_features(ball_xy, fps=25):
    ball_xy = np.asarray(ball_xy, dtype=float)
    if ball_xy.ndim != 2 or ball_xy.shape[1] != 2:
        raise ValueError("Expected ball positions with shape (frames, 2)")
    valid = np.isfinite(ball_xy).all(axis=1)
    xy = pd.DataFrame(np.where(valid[:, None], ball_xy, np.nan))
    result = {}
    for seconds in (1, 2):
        width = seconds * fps + 1
        # Coordinate interquartile spread resists isolated ball-tracking jumps.
        roll = xy.rolling(width, center=True, min_periods=(width + 1)//2)
        spread = roll.quantile(.75) - roll.quantile(.25)
        result[f"ball_iqr_{seconds}s"] = np.hypot(spread[0], spread[1])
        result[f"ball_observed_fraction_{seconds}s"] = pd.Series(valid.astype(float)).rolling(
            width, center=True, min_periods=width).mean()
    # Past/future endpoint displacement at half a second is less sensitive to
    # single-frame jitter; neither endpoint is filled when the ball is missing.
    step = fps // 2
    speed = np.full(len(xy), np.nan)
    speed[step:] = np.linalg.norm(xy.to_numpy()[step:] - xy.to_numpy()[:-step], axis=1) * fps / step
    for direction, values in (("past", pd.Series(speed)), ("future", pd.Series(speed)[::-1])):
        summary = values.rolling(fps, min_periods=(fps+1)//2).median()
        if direction == "future":
            # Future displacements must start at or after the current frame.
            summary = summary[::-1].shift(-step)
        result[f"ball_robust_speed_{direction}_1s"] = summary
    result["ball_stationary_step_fraction_1s"] = pd.Series(
        np.where(np.isfinite(speed), (speed < 1).astype(float), np.nan)
    ).rolling(2*fps+1, center=True, min_periods=fps+1).mean()
    return pd.DataFrame(result).astype(np.float32)


def set_relative_features(positions, ball_xy, teams, fps=25):
    """Match each current position to its closest same-team past position.

    This avoids assuming slot continuity. It can underestimate speed when two
    teammates cross; it is an experimental replacement, not corrected tracking.
    """
    positions = np.asarray(positions, dtype=float)
    ball_xy = np.asarray(ball_xy, dtype=float)
    teams = np.asarray(teams)
    n_frames = len(ball_xy)
    step = fps // 5
    dt = step / fps
    bv = np.full_like(ball_xy, np.nan)
    bv[step:] = (ball_xy[step:] - ball_xy[:-step]) / dt
    bv[np.linalg.norm(bv, axis=1) > 80] = np.nan
    result = {}
    for team, prefix in ((0, "h__"), (1, "a__")):
        xy = positions[:, teams == team]
        relative = np.full(xy.shape[:2], np.nan)
        for start in range(step, n_frames, 2000):
            stop = min(start+2000, n_frames)
            current, previous = xy[start:stop], xy[start-step:stop-step]
            delta = current[:, :, None, :] - previous[:, None, :, :]
            distances = np.linalg.norm(delta, axis=3)
            safe = np.where(np.isfinite(distances), distances, np.inf)
            matches = np.argmin(safe, axis=2)
            tied = np.isfinite(safe) & np.isclose(safe, safe.min(axis=2, keepdims=True), rtol=0, atol=1e-6)
            velocity = np.take_along_axis(delta, matches[:, :, None, None], axis=2)[:, :, 0] / dt
            # Slot order must not resolve ambiguous associations.
            velocity[tied.sum(axis=2) != 1] = np.nan
            velocity[np.linalg.norm(velocity, axis=2) > 15] = np.nan
            relative[start:stop] = np.linalg.norm(velocity-bv[start:stop, None, :], axis=2)
        distance = np.linalg.norm(xy-ball_xy[:, None, :], axis=2)
        safe = np.where(np.isfinite(distance), distance, np.inf)
        nearest = np.argmin(safe, axis=1)
        tied = np.isfinite(safe) & np.isclose(safe, safe.min(axis=1, keepdims=True), rtol=0, atol=1e-6)
        nearest_relative = relative[np.arange(n_frames), nearest].copy()
        nearest_relative[tied.sum(axis=1) != 1] = np.nan
        result[prefix+"nearest_set_relative_velocity"] = nearest_relative
        near = relative.copy()
        near[(distance > 3) | ~np.isfinite(distance)] = np.nan
        minimum = np.min(np.where(np.isfinite(near), near, np.inf), axis=1)
        minimum[~np.isfinite(minimum)] = np.nan
        result[prefix+"set_relative_velocity_within_3m"] = minimum
    return pd.DataFrame(result).astype(np.float32)


def build_motion_inputs(n_frames):
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
            frame_id, player_raw, ball_raw = line.strip().split(":")
            if int(frame_id) != frame + 1:
                raise ValueError("Tracking frame alignment failed")
            players = np.fromstring(player_raw.replace(";", ","), sep=",").reshape(-1, 3)
            if len(players) != len(teams):
                raise ValueError("Tracking slots do not match metadata")
            positions[frame] = players[:, :2]*pitch
            ball[frame] = np.fromstring(ball_raw, sep=",")[:2]*pitch
            count += 1
    if count != n_frames:
        raise ValueError("Tracking ended early")
    return stationarity_features(ball), set_relative_features(positions, ball, teams)
