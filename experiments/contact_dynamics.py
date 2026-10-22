"""Bounded trajectory repair and touch dynamics without labels or player identity."""

import numpy as np
import pandas as pd

from games import GAME, GAMES, require
from starter.baseline import team_of_each_track


def load_observations(n_frames):
    metadata, tracking = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(metadata)
    positions = np.full((n_frames, len(teams), 2), np.nan)
    speeds = np.full((n_frames, len(teams)), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    pitch = np.array(GAMES[GAME]["pitch"])
    count = 0
    with open(tracking) as handle:
        for frame, line in enumerate(handle):
            if frame >= n_frames:
                break
            frame_id, player_raw, ball_raw = line.strip().split(":")
            if int(frame_id) != frame+1:
                raise ValueError("Tracking frame alignment failed")
            players = np.fromstring(player_raw.replace(";", ","), sep=",").reshape(-1, 3)
            if len(players) != len(teams):
                raise ValueError("Tracking slots differ from metadata")
            positions[frame], speeds[frame] = players[:, :2]*pitch, players[:, 2]
            ball[frame] = np.fromstring(ball_raw, sep=",")[:2]*pitch
            count += 1
    if count != n_frames:
        raise ValueError("Tracking ended early")
    return positions, speeds, ball, teams


def bounded_fill(ball, maximum):
    ball = np.asarray(ball, dtype=float)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("Expected frames by two coordinates")
    result = ball.copy()
    valid = np.isfinite(ball).all(axis=1)
    result[~valid] = np.nan
    edges = np.flatnonzero(np.diff(np.r_[False, ~valid, False]))
    for start, end in zip(edges[::2], edges[1::2]):
        if start == 0 or end == len(ball) or end-start > maximum:
            continue
        fraction = (np.arange(start, end)-start+1)/(end-start+1)
        result[start:end] = ball[start-1]+fraction[:, None]*(ball[end]-ball[start-1])
    return result


def repair_ball(ball, method):
    ball = np.asarray(ball, dtype=float)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("Expected frames by two coordinates")
    valid = np.isfinite(ball).all(axis=1)
    clean = np.where(valid[:, None], ball, np.nan)
    if method == "median":
        filled = bounded_fill(clean, 5)
        fitted = pd.DataFrame(filled).rolling(7, center=True, min_periods=3).median().to_numpy()
        allowed = np.isfinite(filled).all(axis=1)
    elif method == "robust":
        padded = np.pad(clean, ((12, 12), (0, 0)), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, 25, axis=0).transpose(0, 2, 1)
        observed = np.isfinite(windows).all(axis=2)
        values = np.where(observed[:, :, None], windows, 0.)
        t = np.arange(-12, 13, dtype=float)
        temporal = np.exp(-.5*(t/6)**2)
        weights = observed*temporal
        fitted = np.full_like(clean, np.nan)
        for iteration in range(4):
            mass = weights.sum(axis=1)
            first = (weights*t).sum(axis=1)
            second = (weights*t*t).sum(axis=1)
            response = np.einsum("nw,nwc->nc", weights, values)
            response_t = np.einsum("nw,w,nwc->nc", weights, t, values)
            denominator = mass*second-first*first
            ok = (observed.sum(axis=1) >= 5) & (denominator > 1e-10)
            intercept = np.divide(second[:, None]*response-first[:, None]*response_t,
                                  denominator[:, None], out=np.zeros_like(response), where=ok[:, None])
            slope = np.divide(mass[:, None]*response_t-first[:, None]*response,
                              denominator[:, None], out=np.zeros_like(response), where=ok[:, None])
            fitted[ok] = intercept[ok]
            if iteration != 3:
                residual = np.linalg.norm(values-intercept[:, None, :]-t[None, :, None]*slope[:, None, :], axis=2)
                weights = observed*temporal*np.minimum(1., 2/np.maximum(residual, 1e-12))
        allowed = np.isfinite(bounded_fill(clean, 13)).all(axis=1)
    else:
        raise ValueError("Unknown trajectory repair")
    result = np.where(np.isfinite(fitted).all(axis=1)[:, None], fitted, clean)
    result[~allowed] = np.nan
    return result


def geometry_features(positions, speeds, ball, original, teams):
    n = len(ball)
    if positions.shape[:2] != speeds.shape or positions.shape != (n, len(teams), 2) or original.shape != ball.shape:
        raise ValueError("Geometry arrays must align")
    result = {"clean_ball_seen": np.isfinite(ball).all(axis=1).astype(float),
              "clean_ball_correction": np.linalg.norm(ball-original, axis=1)}
    before = np.full_like(ball, np.nan)
    before[5:] = (ball[5:]-ball[:-5])*5
    before[np.linalg.norm(before, axis=1) > 80] = np.nan
    result["clean_ball_speed5"] = np.linalg.norm(before, axis=1)
    for code, prefix in ((0, "h__"), (1, "a__")):
        xy, speed = positions[:, teams == code], speeds[:, teams == code]
        distance = np.linalg.norm(xy-ball[:, None, :], axis=2)
        safe = np.where(np.isfinite(distance), distance, np.inf)
        sorted_distance = np.sort(safe, axis=1)
        for rank in range(3):
            value = sorted_distance[:, rank].copy() if rank < sorted_distance.shape[1] else np.full(n, np.nan)
            value[~np.isfinite(value)] = np.nan
            result[prefix+f"clean_distance_{rank+1}"] = value
        nearest = safe.argmin(axis=1)
        nearest_speed = speed[np.arange(n), nearest].copy()
        tied = np.isclose(safe, safe.min(axis=1, keepdims=True), atol=1e-6, rtol=0).sum(axis=1) != 1
        nearest_speed[tied | ~np.isfinite(safe.min(axis=1)) | (nearest_speed < 0)] = np.nan
        result[prefix+"clean_nearest_speed"] = nearest_speed
        for radius in (1, 3, 5):
            count = (distance <= radius).sum(axis=1).astype(float)
            count[~np.isfinite(distance).any(axis=1)] = np.nan
            result[prefix+f"clean_within_{radius}m"] = count
    dh, da = result["h__clean_distance_1"], result["a__clean_distance_1"]
    for prefix, own, other in (("h__", dh, da), ("a__", da, dh)):
        result[prefix+"clean_advantage"] = other-own
        result[prefix+"clean_control_share"] = (other+.1)/(own+other+.2)
    return pd.DataFrame(result).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def impulse_features(ball, geometry):
    n = len(ball)
    before, after = np.full_like(ball, np.nan), np.full_like(ball, np.nan)
    before[5:] = (ball[5:]-ball[:-5])*5
    after[:-5] = (ball[5:]-ball[:-5])*5
    for velocity in (before, after):
        velocity[np.linalg.norm(velocity, axis=1) > 80] = np.nan
    impulse = np.linalg.norm(after-before, axis=1)
    speed_before, speed_after = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    result = dict(clean_impulse=impulse, clean_speed_before=speed_before, clean_speed_after=speed_after,
                  clean_turn_cosine=np.sum(before*after, axis=1)/np.maximum(speed_before*speed_after, .01))
    for prefix in ("h__", "a__"):
        strength = np.exp(-.5*(geometry[prefix+"clean_distance_1"].to_numpy()/2)**2)*np.clip(impulse/10, 0, 1)
        result[prefix+"impulse_strength"] = strength
        for window in (25, 75):
            for direction, series in (("past", pd.Series(strength)), ("future", pd.Series(strength)[::-1])):
                summary = series.rolling(window, min_periods=1).max()
                result[prefix+f"impulse_{direction}_max{window}"] = summary.to_numpy() if direction == "past" else summary[::-1].to_numpy()
        for window in (13, 25):
            for direction, series in (("past", pd.Series(strength)), ("future", pd.Series(strength)[::-1])):
                summary = series.rolling(window, min_periods=1).mean()
                result[prefix+f"impulse_{direction}_mean{window}"] = summary.to_numpy() if direction == "past" else summary[::-1].to_numpy()
    return pd.DataFrame(result, index=np.arange(n)).replace([np.inf, -np.inf], np.nan).astype(np.float32)
