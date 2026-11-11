"""Check and correct a small second-camera offset from joint ball observations."""

import numpy as np

from rule_based_decoder import runs


def calibrate(ball, donor, aligned, *, fps=25):
    calibrated = np.full_like(donor, np.nan)
    max_lag = max(1, round(.16 * fps))
    offsets = np.arange(-max_lag, max_lag + 1)
    window = round(8 * fps)
    context = round(2 * fps)
    ball_seen = np.isfinite(ball).all(axis=1)
    donor_seen = np.isfinite(donor).all(axis=1)

    for start, end in runs(aligned):
        for left in range(start, end, window):
            right = min(end, left + window)
            overlap = np.arange(max(start + max_lag, left - context),
                                min(end - max_lag, right + context))
            candidates = overlap[:, None] + offsets
            joint = ball_seen[overlap] & donor_seen[candidates].all(axis=1)
            if joint.sum() < max(6, round(.48 * fps)):
                continue

            # Use the same observed frames for every offset so gaps cannot favor one.
            residual = ball[overlap[joint], None, :] - donor[candidates[joint]]
            bias = np.median(residual, axis=0)
            errors = np.linalg.norm(residual - bias, axis=2)
            cost = np.median(errors, axis=0)
            acceptable = ((np.linalg.norm(bias, axis=1) <= 1.5) & (cost <= 1.)
                          & (np.quantile(errors, .80, axis=0) <= 2.))
            if not acceptable.any():
                continue
            best = int(np.argmin(np.where(acceptable, cost + .02 * np.abs(offsets), np.inf)))
            if acceptable[max_lag] and cost[max_lag] - cost[best] < .08:
                best = max_lag

            frames = np.arange(left, right)
            indices = frames + offsets[best]
            valid = (indices >= start) & (indices < end)
            calibrated[frames[valid]] = donor[indices[valid]] + bias[best]
    return calibrated
