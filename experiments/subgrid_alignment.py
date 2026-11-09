"""Frozen metadata-only half-cell refinement; never reads images or targets."""
import numpy as np
import pandas as pd
from broadcast_alignment import formation_cost, frame_usable, map_anchors

FIELDS = ('frames', 'orientation', 'cost', 'margin')
STRICT = dict(maximum_cost=2.5, minimum_margin=.5, maximum_fine_span=10)


def equal(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)
    if not same:
        raise ValueError('Subgrid parity failed: ' + label)


def safe_interval(frame, radius, n, blocks, reserved, held):
    left, right = frame-radius, frame+radius
    return (0 <= left and right < n and any(a <= left and right <= b for _, a, b in blocks)
            and not any(left <= r['end']+250 and right >= r['start']-250 for r in reserved)
            and not np.any((held+250 >= left) & (held-250 <= right)))


def coarse_search(frame, tactical, broadcast, bf):
    costs = np.stack([formation_cost(tactical['positions'][frame], broadcast['positions'][bf],
                                    tactical['teams'], broadcast['teams'], o) for o in range(4)])
    o, j = np.unravel_index(int(costs.argmin()), costs.shape)
    best = costs[o, j]
    far = np.flatnonzero(np.abs(bf-bf[j]) > 250)
    runner = np.array([np.nan, np.nan, np.inf])
    if len(far):
        ro, rj = np.unravel_index(int(costs[:, far].argmin()), (4, len(far)))
        runner = np.array([bf[far[rj]], ro, costs[ro, far[rj]]])
    return np.array([bf[j], o, best, runner[2]-best]), runner


def local_refinement(frame, center, orientation, tactical, broadcast, usable_b):
    candidates = np.arange(int(center)-12, int(center)+13)
    candidates = candidates[(candidates >= 25) & (candidates < len(usable_b)-25)]
    count = len(candidates)
    censored = count < 25
    candidates = candidates[usable_b[candidates]]
    if not len(candidates):
        return np.array([np.nan, np.inf, np.nan]), (count, 0), censored
    score = formation_cost(tactical['positions'][frame], broadcast['positions'][candidates],
                           tactical['teams'], broadcast['teams'], int(orientation))
    best = int(score.argmin())
    possible = candidates[score <= score[best]+.05]
    return np.array([candidates[best], score[best], possible[-1]-possible[0]]), (count, len(candidates)), censored


def neighborhood_reason(coarse, spans):
    if not np.isfinite(coarse[:, :3]).all(): return 'unavailable_neighbor'
    if (coarse[:, 2] > 2.5).any(): return 'neighbor_cost'
    if not (coarse[:, 3] >= .5).all(): return 'neighbor_margin'
    if not (spans <= 10).all(): return 'neighbor_halfcell_span'
    if not (coarse[:, 1] == coarse[1, 1]).all(): return 'orientation'
    if (np.abs(np.diff(coarse[:, 0])-25) > 25).any(): return 'coarse_increment'
    return 'pass'


def fine_search(frame, center, orientation, tactical, broadcast, usable_t, usable_b):
    candidates = np.arange(int(center)-25, int(center)+26)
    candidates = candidates[(candidates >= 25) & (candidates < len(usable_b)-25)]
    values = []
    for d in (-25, 0, 25):
        v = formation_cost(tactical['positions'][frame+d], broadcast['positions'][candidates+d],
                           tactical['teams'], broadcast['teams'], int(orientation))
        v[~usable_b[candidates+d] | ~usable_t[frame+d]] = np.inf
        values.append(v**2)
    score = np.sqrt(np.mean(values, axis=0))
    best = int(score.argmin())
    if not np.isfinite(score[best]): return np.array([np.nan, np.nan, np.nan])
    possible = candidates[score <= score[best]+.05]
    return np.array([candidates[best], score[best], possible[-1]-possible[0]])


def refine_anchors(tactical, broadcast, original, blocks, reserved, held):
    tf = np.arange(25, len(tactical['positions'])-25, 25)
    equal(tf, original['tactical_frames'], 'original prefix anchor grid')
    ut, ub = frame_usable(tactical), frame_usable(broadcast)
    bf = np.arange(25, len(ub)-25, 25); bf = bf[ub[bf]]
    if not len(bf): raise ValueError('No eligible global coarse candidates')
    size = len(tf)
    coarse, fine = np.full((size, 4), np.nan), np.full((size, 3), np.nan)
    trace = dict(safe=np.array([safe_interval(int(t), 25, len(ut), blocks, reserved, held) for t in tf]),
                 runner=np.full((size, 3), np.nan), winner_refined=np.full((size, 3), np.nan),
                 runner_refined=np.full((size, 3), np.nan), candidate_counts=np.zeros((size, 4), int),
                 boundary_censored=np.zeros((size, 2), bool), fine_proposal=np.full((size, 3), np.nan),
                 extra_costs=np.full((size, 2), np.nan), reason=np.full(size, 'unsafe_context', dtype='<U32'))
    for i, t in enumerate(tf):
        if not trace['safe'][i]: continue
        trace['reason'][i] = 'tactical_unusable'
        if not ut[t]: continue
        old, runner = coarse_search(t, tactical, broadcast, bf)
        equal(old, original['coarse'][i], 'safe original global coarse replay')
        trace['runner'][i] = runner
        win, wc, censored = local_refinement(t, old[0], old[1], tactical, broadcast, ub)
        trace['winner_refined'][i] = win
        trace['candidate_counts'][i, :2] = wc
        trace['boundary_censored'][i, 0] = censored
        if np.isfinite(runner[0]):
            alt, ac, rc = local_refinement(t, runner[0], runner[1], tactical, broadcast, ub)
            trace['candidate_counts'][i, 2:] = ac
            trace['boundary_censored'][i, 1] = rc
        else:
            alt = np.array([np.nan, np.inf, np.nan])
        trace['runner_refined'][i] = alt
        if win[1] > old[2]+1e-12: raise ValueError('Half-cell omitted original winner')
        coarse[i] = [win[0], old[1], win[1], min(old[3], alt[1]-win[1])]
        trace['reason'][i] = 'grid_edge'
    for i in range(1, size-1):
        if not trace['safe'][i] or not ut[tf[i]]: continue
        reason = neighborhood_reason(coarse[i-1:i+2], trace['winner_refined'][i-1:i+2, 2])
        trace['reason'][i] = reason
        if reason != 'pass': continue
        t, o = tf[i], int(coarse[i, 1])
        proposal = fine_search(t, coarse[i, 0], o, tactical, broadcast, ut, ub)
        trace['fine_proposal'][i] = proposal
        reason = 'fine_unavailable'
        if np.isfinite(proposal).all():
            reason = 'fine_cost' if proposal[1] > 2.5 else 'fine_span' if proposal[2] > 10 else 'extra_unavailable'
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
    equal(mapping['anchor_allowed'], trace['reason'] == 'allowed', 'candidate allowed trace')
    return anchors, trace, mapping


def extension(original, candidate):
    old = original['frames'] >= 0
    result = {k:np.where(old, original[k], candidate[k]) for k in FIELDS}
    result['origin'] = np.where(old, 1, np.where(candidate['frames'] >= 0, 2, 0))
    for k in FIELDS: equal(result[k][old], original[k][old], 'preserved original '+k)
    return result


def checkpoints(mappings, tactical, broadcast, blocks, reserved, held):
    ut, ub = frame_usable(tactical), frame_usable(broadcast)
    rows = []
    for f in range(11185, 18685):
        if f % 25 not in (6, 19) or not safe_interval(f, 50, len(ut), blocks, reserved, held): continue
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
    for name in ('baseline', 'candidate', 'extension', 'new_only'):
        group = table[table.new_only] if name == 'new_only' else table[table.variant == name]
        cost = group.loc[group.usable, 'cost'].to_numpy()
        mapped = int(group.mapped.sum())
        summary[name] = dict(eligible=len(group), mapped=mapped, usable=len(cost),
            unavailable_mapped=mapped-len(cost), source_availability=float(len(cost)/mapped) if mapped else None,
            above_2_5=int((cost > 2.5).sum()), bad_fraction=float((cost > 2.5).mean()) if len(cost) else None,
            median_cost=float(np.median(cost)) if len(cost) else None)
    return table, summary


def metadata_gate(coverage, consistency, rule, checkpoint_rule, safety):
    base, new = consistency['baseline'], consistency['new_only']
    defined = base['usable'] > 0 and new['usable'] > 0
    checks = dict(total=coverage['observed_public_frames'] >= rule['min_observed_frames'],
        net_added=coverage['net_additional_public_frames'] >= rule['min_net_additional_frames'],
        retained=coverage['original_coverage_retention'] >= rule['min_original_coverage_retention'],
        checkpoint_count=new['usable'] >= checkpoint_rule['minimum_new_usable'],
        checkpoint_availability=bool(new['source_availability'] is not None and new['source_availability'] >= checkpoint_rule['minimum_new_source_availability']),
        checkpoint_geometry=bool(defined and new['bad_fraction'] <= base['bad_fraction']+checkpoint_rule['maximum_excess_bad_fraction']),
        preservation_safety_prefix_replay=bool(safety))
    return dict(checks=checks, passed=all(checks.values()))
