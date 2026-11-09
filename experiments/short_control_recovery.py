"""Recover brief, corroborated handovers that slow ownership smoothing erased."""

import numpy as np

from possession_boundary_precision import spans
from transition_penalty import decode


TEAMS = np.array(['H', 'A'])
CONFIDENCE = .65
FLANK = 13


def recover(reference, owner, fast, eligible, raw, broadcast, blocks):
    """Decode unsmoothed ownership and use it only inside established live runs.

    A short proposal needs support from both cameras and the fast head. Read
    every proposal against the same reference and keep dead and unknown labels.
    """
    reference = np.asarray(reference)
    owner, fast = np.asarray(owner), np.asarray(fast)
    eligible = np.asarray(eligible, dtype=bool)
    n = len(reference)
    if (reference.shape != (n,) or owner.shape != (n, 2) or fast.shape != (n, 2)
            or eligible.shape != (n,) or len(raw) != n or len(broadcast) != n):
        raise ValueError('Recovery requires aligned labels, probabilities and observations')

    tactical = raw[['h__distance_1', 'a__distance_1']].to_numpy()
    camera = broadcast[['h__bc_same_distance_1', 'a__bc_same_distance_1']].to_numpy()
    observed = (eligible & (raw.ball_seen.to_numpy() > 0)
                & (broadcast.bc_matched.to_numpy() == 1)
                & (broadcast.bc_same_ball_seen.to_numpy() == 1)
                & np.isfinite(tactical).all(axis=1) & np.isfinite(camera).all(axis=1)
                & (raw.ball_goal_line_distance.to_numpy() >= 0)
                & (raw.ball_touchline_distance.to_numpy() >= 0))
    bilateral_contact = ((tactical < 2) & (tactical[:, ::-1] - tactical > .5)
                         & (camera < 2) & (camera[:, ::-1] - camera > .5))
    core_support = bilateral_contact & (fast > CONFIDENCE)
    # Allow one frame of boundary uncertainty when the fast head favors the
    # handover and either TabPFN is uncertain or both cameras agree.
    edge_support = (observed[:, None] & (fast > .5)
                    & ((owner > 1 - CONFIDENCE) | bilateral_contact))

    result = reference.copy()
    events = []
    covered = np.zeros(n, dtype=bool)
    for left, right in blocks:
        if not 0 <= left < right <= n or covered[left:right].any():
            raise ValueError('Invalid or overlapping decode blocks')
        covered[left:right] = True
        for begin, finish, incumbent in spans(reference[left:right]):
            if incumbent not in ('H', 'A') or finish - begin <= 250:
                continue
            start, end = left + begin, left + finish
            short_owner = TEAMS[decode(np.where(eligible[start:end, None], owner[start:end], .5))]
            for a, b, team in spans(short_owner):
                if (team == incumbent or not 8 <= b - a <= 50
                        or a < FLANK or len(short_owner) - b < FLANK):
                    continue
                if (not np.all(short_owner[a-FLANK:a] == incumbent)
                        or not np.all(short_owner[b:b+FLANK] == incumbent)):
                    continue
                first, last = start + a, start + b
                k = int(team == 'A')
                core = slice(first, last)
                if not (observed[core].all() and core_support[core, k].sum() >= 3
                        and owner[core, k].mean() >= CONFIDENCE and fast[core, k].mean() >= CONFIDENCE):
                    continue
                first -= int(edge_support[first-1, k])
                last += int(edge_support[last, k])
                result[first:last] = team
                events.append(dict(start=first, end_exclusive=last, team=team))
    return result, events
