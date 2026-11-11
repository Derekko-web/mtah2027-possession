"""Match camera timelines by anonymous, same-team formations."""

import numpy as np
from scipy.spatial import cKDTree


REFLECTIONS = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
ANCHOR_STEP = 25
TEMPLATE = np.array([-25, 0, 25])


def signatures(positions, teams, frames, minimum):
    output = []
    for team in (0, 1):
        xy = positions[frames][:, :, teams == team]
        valid = np.isfinite(xy).all(axis=-1)
        n = valid.sum(axis=-1, keepdims=True)
        center = np.where(valid[..., None], xy, 0).sum(axis=-2) / np.maximum(n, 1)
        spread = np.sqrt(np.where(valid[..., None], (xy - center[..., None, :]) ** 2, 0).sum(axis=-2)
                         / np.maximum(n, 1))
        value = np.concatenate([center, .5 * spread], axis=-1)
        value[n[..., 0] < minimum] = np.nan
        output.append(value)
    return np.concatenate(output, axis=-1).reshape(len(frames), -1)


def formation_error(primary, secondary, teams, other_teams):
    """Evaluate candidate secondary frames against the primary team sets."""
    total = np.zeros(len(secondary))
    for team in (0, 1):
        template = primary[:, teams == team]
        candidates = secondary[:, :, other_teams == team]
        valid = np.isfinite(candidates).all(axis=-1)
        distance = np.linalg.norm(candidates[..., None, :] - template[None, :, None, :, :], axis=-1)
        best = np.where(np.isfinite(distance), distance, np.inf).min(axis=-1)
        costs = np.where(valid, np.minimum(best, 14) ** 2, 0).sum(axis=(1, 2))
        n = valid.sum(axis=(1, 2))
        total += np.divide(costs, n, out=np.full(len(n), np.inf), where=n > 0)
    return np.sqrt(total / 2)


def camera_map(primary, secondary, teams, other_teams, *, max_error, min_margin):
    """Return a frame index and reflection only where adjacent anchors agree on the clock."""
    centers = np.arange(25, len(primary) - 25, ANCHOR_STEP)
    other = np.arange(28, len(secondary) - 28, 5)
    query = signatures(primary, teams, centers[:, None] + TEMPLATE, 4)
    reference = signatures(secondary, other_teams, other[:, None] + TEMPLATE, 3)
    valid = np.isfinite(reference).all(axis=1)
    reference, other = reference[valid], other[valid]
    mapped = np.full(len(primary), -1, dtype=int)
    orientation = np.full(len(primary), -1, dtype=int)
    if len(other) < 12:
        return mapped, orientation
    transformed = np.concatenate([reference * np.tile([*signs, 1, 1], 6) for signs in REFLECTIONS])
    tree = cKDTree(transformed)
    query_valid = np.isfinite(query).all(axis=1)
    _, candidates = tree.query(query[query_valid], k=28)
    accepted = np.full((len(centers), 2), -1, dtype=int)
    for row, found in zip(np.flatnonzero(query_valid), candidates):
        point = centers[row]
        offsets = other[found % len(other)]
        signs = found // len(other)
        reference_points = secondary[offsets[:, None] + TEMPLATE] * REFLECTIONS[signs, None, None]
        error = formation_error(primary[point + TEMPLATE], reference_points, teams, other_teams)
        index = int(np.argmin(error))
        frame, direction = int(offsets[index]), int(signs[index])
        distant = (np.abs(offsets - frame) > 75) | (signs != direction)
        if error[index] > max_error or (distant.any() and error[distant].min() - error[index] < min_margin):
            continue
        fine = np.arange(frame - 3, frame + 4)
        probes = secondary[fine[:, None] + TEMPLATE] * REFLECTIONS[direction]
        fine_error = formation_error(primary[point + TEMPLATE], probes, teams, other_teams)
        accepted[row] = fine[fine_error.argmin()], direction
    for i in range(len(centers) - 1):
        (left, sign), (right, next_sign) = accepted[i:i + 2]
        if left < 0 or right < 0 or sign != next_sign or abs(right - left - ANCHOR_STEP) > 4:
            continue
        start = centers[i]
        sl = slice(start, start + ANCHOR_STEP + 1)
        mapped[sl] = np.rint(np.linspace(left, right, ANCHOR_STEP + 1)).astype(int)
        orientation[sl] = sign
    return mapped, orientation


def fill_ball(ball, secondary_ball, mapping, orientation):
    output = ball.copy()
    missing = (mapping >= 0) & ~np.isfinite(ball).all(axis=1)
    output[missing] = secondary_ball[mapping[missing]] * REFLECTIONS[orientation[missing]]
    return output
