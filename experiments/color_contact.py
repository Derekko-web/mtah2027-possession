"""Independent team-color contact evidence and bounded ownership proposals."""
import numpy as np
from boundary_refinement import runs
from control_refinement import smooth
from transition_penalty import decode


def contact_scores(point, positions, probability):
    valid = np.isfinite(positions).all(axis=1) & np.isfinite(probability).all(axis=1) & (probability.max(axis=1) >= .8)
    if not valid.any() or not np.isfinite(point).all():
        return np.full(2, np.nan)
    distance = np.linalg.norm(positions[valid]-point, axis=1)
    return (probability[valid]*np.exp(-.5*(distance/2)**2)[:, None]).max(axis=0)


def contact_posterior(reference, scores):
    reference, scores = np.asarray(reference), np.asarray(scores, float)
    if scores.shape != (len(reference), 2) or not np.isin(reference, [0, 1, 2]).all():
        raise ValueError('Contact arrays must align with a validation block')
    posterior = np.full((len(reference), 2), .5)
    for team in (0, 1):
        posterior[reference == team] = [.9, .1] if team == 0 else [.1, .9]
    finite = np.isfinite(scores).all(axis=1)
    strong = finite & (np.nan_to_num(scores, nan=0).max(axis=1) >= .5) & (np.abs(scores[:, 0]-scores[:, 1]) >= .2)
    support = np.zeros((len(reference), 2), bool)
    owner = np.nan_to_num(scores, nan=0).argmax(axis=1)
    support[np.flatnonzero(strong), owner[strong]] = True
    edges = np.flatnonzero(np.diff(np.r_[False, reference != 2, False]))
    for start, end in zip(edges[::2], edges[1::2]):
        last = None
        for frame in range(start, end):
            if strong[frame]:
                last = frame
            if last is not None and frame-last <= 75:
                posterior[frame] = scores[last]/scores[last].sum()
    return posterior, support


def predict_contact_block(reference, scores):
    reference = np.asarray(reference)
    posterior, support = contact_posterior(reference, scores)
    probability = smooth(posterior, 5)
    result = reference.copy()
    islands = [(start, end, team) for start, end, team in runs(reference) if team != 2 and 8 <= end-start <= 125]
    for start, end, team in runs(decode(probability)):
        if not 8 <= end-start <= 125 or start == 0 or end == len(reference) or (reference[start:end] == 2).any():
            continue
        if probability[start:end, team].mean() < .65 or support[start:end, team].sum() < 3:
            continue
        before, after = reference[max(0, start-13):start], reference[end:min(len(reference), end+13)]
        if (before == 1-team).mean() <= .5 or (after == 1-team).mean() <= .5:
            continue
        if any(team != old_team and start <= left and end >= right for left, right, old_team in islands):
            continue
        result[start:end] = team
    if not np.array_equal(result == 2, reference == 2):
        raise AssertionError('Color-contact proposals changed D')
    return result
