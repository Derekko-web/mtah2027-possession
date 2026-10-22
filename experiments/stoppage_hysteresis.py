"""Bounded interval growth from independently supported model-derived cores."""

import numpy as np


def hysteresis_dead(head, visibility, moving, low, extension=125):
    head, visibility, moving = map(np.asarray, (head, visibility, moving))
    if not (head.ndim == 1 and head.shape == visibility.shape == moving.shape):
        raise ValueError("Hysteresis inputs must be aligned vectors")
    if not 0 < low < .8 or not isinstance(extension, int) or extension < 0:
        raise ValueError("Invalid hysteresis settings")
    if not all(np.isfinite(x).all() for x in (head, visibility, moving)):
        raise ValueError("Non-finite hysteresis evidence")
    qualifies = (visibility >= .5) & (moving <= .2)
    strong = (head > .8) & qualifies
    weak = (head >= low) & qualifies
    edges = np.flatnonzero(np.diff(np.r_[False, strong, False]))
    keep = np.zeros(len(head), bool)
    for start, end in zip(edges[::2], edges[1::2]):
        if end-start < 50:
            continue
        left, right = start, end
        while left > max(0, start-extension) and weak[left-1]:
            left -= 1
        while right < min(len(head), end+extension) and weak[right]:
            right += 1
        keep[left:right] = True
    return np.where(keep, head, 0.)
