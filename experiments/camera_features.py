"""Camera motion from supplied homography, with no video/model uploads or labels."""

import numpy as np
import pandas as pd

from games import GAME, GAMES
from starter.load_homography import load


def camera_features(n_frames):
    matrices, error, valid = load()
    matrices = matrices[:n_frames]
    valid = valid[:n_frames]
    if len(matrices) != n_frames:
        raise ValueError("Homography frame count mismatch")
    centers = np.full((n_frames, 2), np.nan)
    scale = np.full(n_frames, np.nan)
    if valid.any():
        inverse = np.linalg.inv(matrices[valid])
        projected = inverse@np.array([960., 540., 1.])
        centers[valid] = projected[:, :2]/projected[:, 2, None]*np.array(GAMES[GAME]["pitch"])
        # Area scale at image center, in square metres per image pixel.
        dx = inverse@np.array([961., 540., 1.])
        dy = inverse@np.array([960., 541., 1.])
        dx = dx[:, :2]/dx[:, 2, None]*np.array(GAMES[GAME]["pitch"])-centers[valid]
        dy = dy[:, :2]/dy[:, 2, None]*np.array(GAMES[GAME]["pitch"])-centers[valid]
        scale[valid] = np.abs(dx[:, 0]*dy[:, 1]-dx[:, 1]*dy[:, 0])
    displacement = np.full(n_frames, np.nan)
    displacement[5:] = np.linalg.norm(centers[5:]-centers[:-5], axis=1)/.2
    displacement[displacement > 40] = np.nan
    log_scale = np.log(np.maximum(scale, 1e-12))
    zoom = np.r_[np.full(5, np.nan), np.abs(log_scale[5:]-log_scale[:-5])/.2]
    zoom[zoom > 5] = np.nan
    base = pd.DataFrame(dict(camera_pan_speed=displacement, camera_zoom_rate=zoom,
                             camera_log_area_scale=log_scale, camera_fit_error=error[:n_frames]))
    extra = {}
    for name in ("camera_pan_speed", "camera_zoom_rate"):
        for seconds in (1, 3):
            window = base[name].rolling(seconds*25+1, center=True, min_periods=seconds*25+1)
            extra[f"{name}_mean_{seconds}s"] = window.mean()
            extra[f"{name}_max_{seconds}s"] = window.max()
    return base.join(pd.DataFrame(extra)).replace([np.inf, -np.inf], np.nan).astype(np.float32)
