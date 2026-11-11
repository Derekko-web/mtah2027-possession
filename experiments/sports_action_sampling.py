"""Fixed native-frame action windows with sealed-image and half exclusions."""
import numpy as np


def action_centers(n, blocks, reserved, held):
    result = []
    for center in range(11000, 18951, 50):
        start, end = center-100, center+98
        if start < 0 or end >= n or not any(left <= start and end <= right for _, left, right in blocks):
            continue
        if any(start <= row['end']+250 and end >= row['start']-250 for row in reserved):
            continue
        if np.any((np.asarray(held)+250 >= start) & (np.asarray(held)-250 <= end)):
            continue
        result.append(center)
    return np.array(result, dtype=int)


def clip_frames(center):
    return int(center)-100+2*np.arange(100)


def contiguous_ranges(frames):
    frames = np.asarray(frames, dtype=int)
    if not len(frames):
        return []
    edges = np.r_[0, np.flatnonzero(np.diff(frames) != 2)+1, len(frames)]
    return [(int(frames[a]), int(frames[b-1])) for a,b in zip(edges[:-1], edges[1:])]
