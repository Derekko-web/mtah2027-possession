"""Input-only ball acceleration/coherence features, with <=2 s observation context."""

import numpy as np
import pandas as pd

from games import GAME, GAMES, require


def build_kinematics(n_frames):
    tracking = require(["atd_raw"])[0]
    xy = np.full((n_frames, 2), np.nan)
    with open(tracking) as handle:
        for frame, line in enumerate(handle):
            if frame >= n_frames:
                break
            frame_id, _, ball = line.strip().split(":")
            if int(frame_id) != frame + 1:
                raise ValueError("Unexpected frame index")
            xy[frame] = np.fromstring(ball, sep=",")[:2] * np.array(GAMES[GAME]["pitch"])
    fps = GAMES[GAME]["fps"]
    velocity = np.vstack(([np.nan, np.nan], np.diff(xy, axis=0))) * fps
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.r_[np.nan, np.linalg.norm(np.diff(velocity, axis=0), axis=1) * fps]
    displacement = np.full(n_frames, np.nan)
    displacement[fps:] = np.linalg.norm(xy[fps:] - xy[:-fps], axis=1)
    path_length = pd.Series(speed / fps).rolling(fps, min_periods=fps).sum()
    features = pd.DataFrame({
        "ball_acceleration": acceleration,
        "ball_acceleration_past_1s": pd.Series(acceleration).rolling(fps, min_periods=13).median(),
        "ball_acceleration_future_1s": pd.Series(acceleration)[::-1].rolling(fps,min_periods=13).median()[::-1],
        "ball_speed_std_past_1s": pd.Series(speed).rolling(fps,min_periods=13).std(),
        "ball_speed_std_future_1s": pd.Series(speed)[::-1].rolling(fps,min_periods=13).std()[::-1],
        "ball_direction_coherence": displacement / np.maximum(path_length, .01),
    })
    return features.replace([np.inf, -np.inf], np.nan).astype(np.float32)
