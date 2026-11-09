"""Full-match extension of the frozen 5 Hz representation interpolation.

Uses the same two-endpoint float64 arithmetic and float32 output as R37.
Frames without two supported endpoints inside their own half stay unsupported.
The original public-only implementation remains unchanged.
"""
import numpy as np


def interpolate_halves(frames, hidden, supported, n, blocks):
    frames = np.asarray(frames)
    hidden = np.asarray(hidden)
    supported = np.asarray(supported, bool)
    if (frames.ndim != 1 or not len(frames) or hidden.shape != (len(frames), 64)
            or supported.shape != frames.shape or not np.issubdtype(frames.dtype, np.integer)
            or np.any(np.diff(frames) <= 0) or np.any(frames % 5)
            or np.any(frames < 0) or np.any(frames >= n)):
        raise ValueError("Invalid full 5 Hz representation grid")
    features = np.full((n, 64), np.nan, np.float32)
    eligible = np.zeros(n, bool)
    last = 0
    for start, end in blocks:
        if not 0 <= last <= start < end <= n:
            raise ValueError("Invalid or overlapping half blocks")
        last = end
        indices = np.flatnonzero((frames >= start) & (frames < end))
        if not len(indices):
            raise ValueError("Half has no representation centers")
        grid = frames[indices]
        if len(grid) > 1 and not np.all(np.diff(grid) == 5):
            raise ValueError("Incomplete within-half 5 Hz grid")
        query = np.arange(start, end)
        left = np.searchsorted(grid, query, side="right") - 1
        valid = left >= 0
        left = np.maximum(left, 0)
        exact = grid[left] == query
        right = left + 1
        right[exact] = left[exact]
        valid &= right < len(grid)
        right = np.minimum(right, len(grid) - 1)
        li, ri = indices[left], indices[right]
        valid &= supported[li] & supported[ri]
        weight = ((query - grid[left]) / 5).astype(np.float64)
        value = (hidden[li].astype(np.float64) * (1 - weight[:, None])
                 + hidden[ri].astype(np.float64) * weight[:, None]).astype(np.float32)
        features[query[valid]] = value[valid]
        eligible[query] = valid
    return features, eligible


def full_context_frames(labels, public_start=11185, public_end=18685):
    """Same global phase, live centers and complete +/-79 public footprint."""
    train = np.flatnonzero(np.asarray(labels) != "X")
    np.testing.assert_array_equal(train, np.arange(public_start, public_end))
    observed = train[(train % 5 == 0) & (train - 79 >= public_start)
                     & (train + 79 < public_end)]
    centers = observed[np.isin(np.asarray(labels)[observed], ("H", "A"))]
    if not len(centers) or 2 * len(centers) > 3000:
        raise ValueError("Full paired live context exceeds the frozen native interface")
    return centers, observed
