"""One temporal global formation hypothesis; metadata only, no images or targets."""
import numpy as np
import pandas as pd
from broadcast_alignment import formation_cost, frame_usable, map_anchors
from subgrid_alignment import FIELDS, STRICT, equal, safe_interval, extension, metadata_gate
from sports_action_sampling import clip_frames

OFFSETS = (-25, 0, 25)
MAPPING_CONTEXT = 75
INVERSE_RADIUS = 100
DECISION_RADIUS = 175


def temporal_costs(frame, candidates, orientation, tactical, broadcast, ut, ub):
    """All triplets must be usable; no threshold filters the search universe."""
    candidates = np.asarray(candidates, dtype=int)
    scores = np.full((len(candidates), 3), np.inf)
    if not len(candidates):
        return np.full(0, np.inf), scores
    if frame < 25 or frame+25 >= len(ut) or not ut[frame+np.array(OFFSETS)].all():
        return np.full(len(candidates), np.inf), scores
    valid = (candidates >= 25) & (candidates+25 < len(ub))
    for d in OFFSETS:
        indices = np.clip(candidates+d, 0, len(ub)-1)
        valid &= ub[indices]
    for j, d in enumerate(OFFSETS):
        if valid.any():
            scores[valid, j] = formation_cost(tactical['positions'][frame+d],
                broadcast['positions'][candidates[valid]+d], tactical['teams'],
                broadcast['teams'], int(orientation))
    return np.sqrt(np.mean(scores**2, axis=1)), scores


def temporal_search(frame, tactical, broadcast, bf, ut, ub):
    values = [temporal_costs(frame, bf, o, tactical, broadcast, ut, ub) for o in range(4)]
    costs = np.stack([v[0] for v in values])
    o, j = np.unravel_index(int(costs.argmin()), costs.shape)
    if not np.isfinite(costs[o, j]):
        return np.full(4, np.nan), np.array([np.nan, np.nan, np.inf]), np.full(3, np.inf), int(np.isfinite(costs).sum())
    far = np.flatnonzero(np.abs(bf-bf[j]) > 250)
    runner = np.array([np.nan, np.nan, np.inf])
    if len(far):
        ro, rj = np.unravel_index(int(costs[:, far].argmin()), (4, len(far)))
        if np.isfinite(costs[ro, far[rj]]):
            runner = np.array([bf[far[rj]], ro, costs[ro, far[rj]]])
    return (np.array([bf[j], o, costs[o, j], runner[2]-costs[o, j]]),
            runner, values[o][1][j], int(np.isfinite(costs).sum()))


def neighborhood_reason(coarse, individual):
    if not np.isfinite(coarse[:, :3]).all(): return 'unavailable_neighbor'
    if (coarse[:, 2] > 2.5).any(): return 'neighbor_temporal_cost'
    if not (individual <= 2.5).all(): return 'neighbor_individual_cost'
    if not (coarse[:, 3] >= .5).all(): return 'neighbor_temporal_margin'
    if not (coarse[:, 1] == coarse[1, 1]).all(): return 'orientation'
    if (np.abs(np.diff(coarse[:, 0])-25) > 25).any(): return 'coarse_increment'
    return 'pass'


def temporal_anchors(tactical, broadcast, blocks, reserved, held, progress=None):
    ut, ub = frame_usable(tactical), frame_usable(broadcast)
    tf = np.arange(25, len(ut)-25, 25)
    bf = np.arange(25, len(ub)-25, 25)
    bf = bf[ub[bf-25] & ub[bf] & ub[bf+25]]
    if not len(bf): raise ValueError('No eligible temporal global candidates')
    coarse, fine = np.full((len(tf), 4), np.nan), np.full((len(tf), 3), np.nan)
    trace = dict(safe=np.array([safe_interval(int(t), 25, len(ut), blocks, reserved, held) for t in tf]),
        runner=np.full((len(tf), 3), np.nan), individual_costs=np.full((len(tf), 3), np.inf),
        candidate_counts=np.zeros(len(tf), int), fine_proposal=np.full((len(tf), 3), np.nan),
        fine_individual=np.full((len(tf), 3), np.inf), extra_costs=np.full((len(tf), 2), np.nan),
        reason=np.full(len(tf), 'unsafe_context', dtype='<U36'))
    for i, t in enumerate(tf):
        if not trace['safe'][i]: continue
        trace['reason'][i] = 'tactical_triplet_unusable'
        if not ut[t+np.array(OFFSETS)].all(): continue
        coarse[i], trace['runner'][i], trace['individual_costs'][i], trace['candidate_counts'][i] = temporal_search(t, tactical, broadcast, bf, ut, ub)
        trace['reason'][i] = 'grid_edge'
        if progress is not None and i % 50 == 0: progress(i, len(tf))
    for i in range(1, len(tf)-1):
        if not trace['safe'][i] or trace['reason'][i] == 'tactical_triplet_unusable': continue
        reason = neighborhood_reason(coarse[i-1:i+2], trace['individual_costs'][i-1:i+2])
        trace['reason'][i] = reason
        if reason != 'pass': continue
        t, o = tf[i], int(coarse[i, 1])
        candidates = np.arange(int(coarse[i, 0])-25, int(coarse[i, 0])+26)
        candidates = candidates[(candidates >= 25) & (candidates < len(ub)-25)]
        costs, individual = temporal_costs(t, candidates, o, tactical, broadcast, ut, ub)
        j = int(costs.argmin())
        if not np.isfinite(costs[j]):
            trace['reason'][i] = 'fine_unavailable'; continue
        possible = candidates[costs <= costs[j]+.05]
        proposal = np.array([candidates[j], costs[j], possible[-1]-possible[0]])
        trace['fine_proposal'][i], trace['fine_individual'][i] = proposal, individual[j]
        reason = ('fine_temporal_cost' if proposal[1] > 2.5 else
                  'fine_individual_cost' if (individual[j] > 2.5).any() else
                  'fine_span' if proposal[2] > 10 else 'extra_unavailable')
        if reason == 'extra_unavailable':
            b = int(proposal[0])
            for j, d in enumerate((-12, 12)):
                if ut[t+d] and ub[b+d]:
                    trace['extra_costs'][i, j] = formation_cost(tactical['positions'][t+d],
                        broadcast['positions'][b+d:b+d+1], tactical['teams'], broadcast['teams'], o)[0]
            if np.isfinite(trace['extra_costs'][i]).all():
                reason = 'extra_cost' if (trace['extra_costs'][i] > 2.5).any() else 'allowed'
        trace['reason'][i] = reason
        if reason == 'allowed': fine[i] = proposal
    anchors = dict(tactical_frames=tf, coarse=coarse, fine=fine)
    mapping = map_anchors(anchors, tactical, broadcast, STRICT)
    equal(mapping['anchor_allowed'], trace['reason'] == 'allowed', 'temporal allowed trace')
    return anchors, trace, mapping


def inverse_schedule(n, mapping, blocks, reserved, held, broadcast_usable):
    """Every success and failure reads at most c±100 mapping rows, context ±75."""
    rows = []
    counts = dict(grid=0, boundary_or_prefix=0, sealed=0, center_unmapped=0,
                  source_bounds=0, source_unusable=0, inverse_component=0, eligible=0)
    for center in range(11000, 18951, 25):
        counts['grid'] += 1
        left, right = center-DECISION_RADIUS, center+DECISION_RADIUS
        if left < 0 or right >= n or not any(a <= left and right <= b for _, a, b in blocks):
            counts['boundary_or_prefix'] += 1; continue
        if not safe_interval(center, DECISION_RADIUS, n, blocks, reserved, held):
            counts['sealed'] += 1; continue
        b = int(mapping['frames'][center])
        if b < 0 or mapping['orientation'][center] < 0:
            counts['center_unmapped'] += 1; continue
        frames = clip_frames(b)
        if frames[0] < 0 or frames[-1] >= len(broadcast_usable):
            counts['source_bounds'] += 1; continue
        if not broadcast_usable[frames[0]:frames[-1]+1].all():
            counts['source_unusable'] += 1; continue
        scan = center-INVERSE_RADIUS
        mapped = mapping['frames'][scan:center+INVERSE_RADIUS+1]
        orientation = mapping['orientation'][scan:center+INVERSE_RADIUS+1]
        valid = (mapped >= 0) & (mapped < len(broadcast_usable)) & (orientation >= 0)
        edges = np.r_[0, np.flatnonzero(~valid[:-1] | ~valid[1:] |
            (orientation[1:] != orientation[:-1]) | (np.diff(mapped) < 0) | (np.diff(mapped) > 2))+1, len(mapped)]
        a, z = int(edges[edges <= INVERSE_RADIUS][-1]), int(edges[edges > INVERSE_RADIUS][0])
        values = mapped[a:z]
        if not len(values) or frames[0] < values[0] or frames[-1] > values[-1] or np.any(np.min(np.abs(frames[:, None]-values), axis=1) > 1):
            counts['inverse_component'] += 1; continue
        first = int(np.searchsorted(values, frames[0], side='right')-1)
        last = int(np.searchsorted(values, frames[-1], side='left'))
        rows.append(dict(tactical_center=center, broadcast_center=b,
            component_left=scan+a, component_right=scan+z-1, image_left=scan+a+first,
            image_right=scan+a+last, decision_left=left, decision_right=right))
        counts['eligible'] += 1
    return rows, counts


def checkpoints(mappings, tactical, broadcast, blocks, reserved, held):
    ut, ub = frame_usable(tactical), frame_usable(broadcast)
    rows = []
    for f in range(11185, 18685):
        if f % 25 not in (6, 19) or not safe_interval(f, 75, len(ut), blocks, reserved, held): continue
        for name, mapping in mappings.items():
            b, o = int(mapping['frames'][f]), int(mapping['orientation'][f])
            mapped = b >= 0 and o >= 0
            usable = bool(mapped and ut[f] and b < len(ub) and ub[b])
            cost = formation_cost(tactical['positions'][f], broadcast['positions'][b:b+1],
                tactical['teams'], broadcast['teams'], o)[0] if usable else np.nan
            rows.append(dict(variant=name, tactical_frame=f, mapped=mapped, usable=usable, cost=cost,
                new_only=bool(name == 'extension' and mapping['origin'][f] == 2)))
    table = pd.DataFrame(rows)
    summary = {}
    for name in (*mappings, 'new_only'):
        group = table[table.new_only] if name == 'new_only' else table[table.variant == name]
        cost = group.loc[group.usable, 'cost'].to_numpy(); mapped = int(group.mapped.sum())
        summary[name] = dict(eligible=len(group), mapped=mapped, usable=len(cost),
            unavailable_mapped=mapped-len(cost), source_availability=float(len(cost)/mapped) if mapped else None,
            above_2_5=int((cost > 2.5).sum()), bad_fraction=float((cost > 2.5).mean()) if len(cost) else None,
            median_cost=float(np.median(cost)) if len(cost) else None)
    return table, summary
