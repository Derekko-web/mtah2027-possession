"""Physical release evidence limited to model-derived brief ownership islands."""

import numpy as np
import pandas as pd

from boundary_refinement import runs
from control_refinement import compose, smooth
from transition_penalty import decode


def exclusive_contacts(raw):
    result = []
    for owner, other in (("h__", "a__"), ("a__", "h__")):
        result.append((raw[owner+"distance_1"].to_numpy() < 2) &
                      (raw[other+"distance_1"].to_numpy()-raw[owner+"distance_1"].to_numpy() > .5))
    return np.column_stack(result)


def trim_islands(emissions, current, contacts, window, both):
    states = decode(emissions)
    slow = smooth(current, 25)
    total = slow[:, :2].sum(axis=1, keepdims=True)
    current_owner = np.divide(slow[:, :2], total, out=np.full((len(states), 2), .5), where=total > 0)
    result = emissions.copy()
    counts = pd.DataFrame(contacts.astype(float)).rolling(window, center=True, min_periods=1).sum().to_numpy()
    for start, end, owner in runs(states):
        if owner == 2 or not 8 <= end-start <= 125 or start == 0 or end == len(states):
            continue
        if states[start-1] != 1-owner or states[end] != 1-owner:
            continue
        contrary = (counts[:, 1-owner] >= 3) & (counts[:, owner] == 0) & (current_owner[:, 1-owner] > .6)
        cuts = np.flatnonzero(contrary[max(start+8, end-25):end])+max(start+8, end-25)
        right = int(cuts[0]) if len(cuts) else end
        left = start
        if both:
            cuts = np.flatnonzero(contrary[start:min(start+25, right-8)])+start
            if len(cuts):
                left = int(cuts[-1]+1)
        for a, b in ((start, left), (right, end)):
            if b > a:
                result[a:b] = compose(emissions[a:b, 2], current_owner[a:b])
    return result
