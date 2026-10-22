"""Label-free, team-symmetric tracking and tactical features for offline inference.

Largest temporal input radius is five seconds, inside the ten-second CV buffers.
No frame index, match clock, answer key or hand-written possession boundaries
are used as model inputs. Raw coordinates are converted to metres; shared pitch
location features are invariant to switching ends at halftime.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, require, source_frames
from starter.baseline import team_of_each_track
from starter.load_smart_tagging import load as load_tags

MAX_CONTEXT_SECONDS = 5.04  # Five-second window plus one frame for a speed difference.


def build_features(n_frames=None):
    n_frames = n_frames or source_frames()
    metadata, tracking = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(metadata)
    pitch = np.array(GAMES[GAME]["pitch"])
    fps = GAMES[GAME]["fps"]
    ball_xy = np.full((n_frames, 2), np.nan)
    storage = {}
    def put(name, frame, value):
        if name not in storage:
            storage[name] = np.full(n_frames, np.nan, dtype=np.float32)
        storage[name][frame] = value

    with open(tracking) as handle:
        for frame, line in enumerate(handle):
            if frame >= n_frames:
                break
            frame_id, raw, ball_raw = line.strip().split(":")
            if int(frame_id) != frame + 1:
                raise ValueError("Tracking frame alignment failed")
            players = np.fromstring(raw.replace(";", ","), sep=",").reshape(-1, 3)
            if len(players) != len(teams):
                raise ValueError("Tracking slots do not match metadata")
            positions = players[:, :2] * pitch
            ball = np.fromstring(ball_raw, sep=",")[:2] * pitch
            if np.isfinite(ball).all():
                ball_xy[frame] = ball
            put("ball_seen", frame, float(np.isfinite(ball).all()))
            valid = np.isfinite(positions).all(axis=1)
            for team, prefix in [(0, "h__"), (1, "a__")]:
                mask = valid & (teams == team)
                xy = positions[mask]
                speeds = players[mask, 2]
                finite_speed = speeds[np.isfinite(speeds) & (speeds >= 0)]
                put(prefix + "n_tracks", frame, len(xy))
                if len(finite_speed):
                    put(prefix + "speed_median", frame, np.median(finite_speed))
                    put(prefix + "speed_p90", frame, np.quantile(finite_speed, .9))
                    put(prefix + "moving_fraction", frame, np.mean(finite_speed > 1))
                if not len(xy):
                    continue
                center = np.median(xy, axis=0)
                put(prefix + "spread_x", frame, np.std(xy[:, 0]))
                put(prefix + "spread_y", frame, np.std(xy[:, 1]))
                if not np.isfinite(ball).all():
                    continue
                distance = np.linalg.norm(xy - ball, axis=1)
                order = np.argsort(distance)
                for rank in range(min(3, len(order))):
                    put(prefix + f"distance_{rank+1}", frame, distance[order[rank]])
                put(prefix + "nearest_speed", frame, speeds[order[0]])
                put(prefix + "distance_median", frame, np.median(distance))
                put(prefix + "center_ball_distance", frame, np.linalg.norm(center - ball))
                for radius in (1, 3, 5, 10):
                    put(prefix + f"within_{radius}m", frame, np.sum(distance <= radius))
        if frame < n_frames - 1:
            raise ValueError("Tracking ended early")

    base = pd.DataFrame(storage)
    ball_speed = np.r_[np.nan, np.linalg.norm(np.diff(ball_xy, axis=0), axis=1) * fps]
    base["ball_speed"] = ball_speed
    base["ball_goal_line_distance"] = np.minimum(ball_xy[:, 0], pitch[0] - ball_xy[:, 0])
    base["ball_touchline_distance"] = np.minimum(ball_xy[:, 1], pitch[1] - ball_xy[:, 1])
    for seconds in (.2, 1):
        step = int(seconds * fps)
        displacement = np.full(n_frames, np.nan)
        displacement[step:] = np.linalg.norm(ball_xy[step:] - ball_xy[:-step], axis=1) / seconds
        base[f"ball_displacement_{seconds}s"] = displacement
    dh, da = base["h__distance_1"], base["a__distance_1"]
    for prefix, own, other in [("h__", dh, da), ("a__", da, dh)]:
        base[prefix + "advantage"] = other - own
        base[prefix + "control_share"] = (other + .1) / (own + other + .2)
        base[prefix + "is_closest"] = np.where(dh.notna() & da.notna(), own <= other, np.nan)

    # Recent and forthcoming evidence carries team information through tracking gaps.
    extra = {}
    temporal = [name for name in base if any(token in name for token in
                ("distance_1", "advantage", "control_share", "is_closest",
                 "speed_median", "moving_fraction", "ball_speed", "ball_seen"))]
    for seconds in (1, 3, 5):
        width = seconds * fps
        for name in temporal:
            column = base[name]
            extra[f"{name}__past_{seconds}s"] = column.rolling(width, min_periods=1).mean()
            extra[f"{name}__future_{seconds}s"] = column.iloc[::-1].rolling(
                width, min_periods=1).mean().iloc[::-1]
            if name in ["h__distance_1", "a__distance_1", "ball_speed"]:
                extra[f"{name}__minimum_{seconds}s"] = column.rolling(
                    2 * width + 1, center=True, min_periods=1).min()
    features = pd.concat([base, pd.DataFrame(extra)], axis=1)

    tag_data = {}
    for event in load_tags():
        start, end = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        if end <= start:
            continue
        team = (event["tags"].get("Team") or "").lower()
        prefix = "h__" if team == "barcelona" else "a__" if team == "atletic club" else ""
        name = prefix + "tag_" + event["code"].lower().replace(" ", "_")
        if name not in tag_data:
            tag_data[name] = np.zeros(n_frames, dtype=np.float32)
        tag_data[name][start:end] = 1
        if event["code"].lower() == "set pieces":
            kind = (event["tags"].get("Type") or "unknown").lower().replace(" ", "_")
            key = prefix + "tag_restart_" + kind
            if key not in tag_data:
                tag_data[key] = np.zeros(n_frames, dtype=np.float32)
            tag_data[key][start:end] = 1
    # Every team-specific input must have its opposite-team counterpart for augmentation.
    for name in list(tag_data):
        if name.startswith(("h__", "a__")):
            other = ("a__" if name.startswith("h__") else "h__") + name[3:]
            tag_data.setdefault(other, np.zeros(n_frames, dtype=np.float32))
    features = pd.concat([features, pd.DataFrame(tag_data)], axis=1)
    if not features.columns.is_unique:
        raise ValueError("Feature names must be unique")
    return features.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def swap_teams(features):
    """Swap team-relative inputs, leaving shared physical inputs unchanged."""
    sources = []
    for name in features.columns:
        other = ("a__" if name.startswith("h__") else "h__") + name[3:]
        sources.append(other if name.startswith(("h__", "a__")) else name)
    swapped = features.loc[:, sources].copy()
    swapped.columns = features.columns
    return swapped
