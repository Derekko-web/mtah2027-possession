"""Extend an existing stoppage where the ball disappears at its edge."""

import numpy as np

from rule_based_decoder import runs


def extend_stoppages(reference, ball, seconds, *, fps=25):
    labels = reference.copy()
    if seconds == 0:
        return labels
    missing = ~np.isfinite(ball).all(axis=1)
    max_extension = round(seconds * fps)
    for start, end in runs(missing & (reference != 'D')):
        previous_dead = start > 0 and reference[start - 1] == 'D'
        following_dead = end < len(ball) and reference[end] == 'D'
        if previous_dead and following_dead and end - start <= 2 * max_extension:
            labels[start:end] = 'D'
        elif previous_dead != following_dead:
            if previous_dead:
                labels[start:min(end, start + max_extension)] = 'D'
            else:
                labels[max(start, end - max_extension):end] = 'D'
    return labels
