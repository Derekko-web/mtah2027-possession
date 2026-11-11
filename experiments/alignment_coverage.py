"""Bounded inverse clip scheduling; strict correspondence and possession labels unchanged."""
import numpy as np
import pandas as pd
from broadcast_action_sampling import mapping_components
from broadcast_action_features import feature_view
from sports_action_features import PROBABILITY
from sports_action_sampling import clip_frames


def inverse_schedule(n, mapping, blocks, reserved, held, broadcast_usable, grid):
    if grid not in (25, 50):
        raise ValueError('Unfrozen center grid')
    rows = []
    counts = dict(grid=0, boundary_or_prefix=0, sealed=0, center_unmapped=0,
                  source_bounds=0, source_unusable=0, inverse_component=0, eligible=0)
    for center in range(11000, 18951, grid):
        counts['grid'] += 1
        decision_left, decision_right = center-175, center+175
        if decision_left < 0 or decision_right >= n or not any(a <= decision_left and decision_right <= b for _, a, b in blocks):
            counts['boundary_or_prefix'] += 1
            continue
        if any(decision_left <= r['end']+250 and decision_right >= r['start']-250 for r in reserved) or np.any((np.asarray(held)+250 >= decision_left) & (np.asarray(held)-250 <= decision_right)):
            counts['sealed'] += 1
            continue
        source_center = int(mapping['frames'][center])
        if source_center < 0 or mapping['orientation'][center] < 0:
            counts['center_unmapped'] += 1
            continue
        frames = clip_frames(source_center)
        if frames[0] < 0 or frames[-1] >= len(broadcast_usable):
            counts['source_bounds'] += 1
            continue
        if not np.all(broadcast_usable[frames[0]:frames[-1]+1]):
            counts['source_unusable'] += 1
            continue
        # Never search the entire component: all absence/eligibility decisions
        # depend on at most this fixed c +/-125 mapping interval.
        scan_left, scan_right = center-125, center+125
        mapped = mapping['frames'][scan_left:scan_right+1]
        orientation = mapping['orientation'][scan_left:scan_right+1]
        valid = (mapped >= 0) & (mapped < len(broadcast_usable)) & (orientation >= 0)
        edges = np.r_[0, np.flatnonzero((~valid[:-1]) | (~valid[1:]) |
                          (orientation[1:] != orientation[:-1]) | (np.diff(mapped) < 0) |
                          (np.diff(mapped) > 2))+1, len(mapped)]
        local_center = 125
        left = int(edges[edges <= local_center][-1])
        right = int(edges[edges > local_center][0])
        values = mapped[left:right]
        if not len(values) or frames[0] < values[0] or frames[-1] > values[-1]:
            counts['inverse_component'] += 1
            continue
        if np.any(np.min(np.abs(frames[:, None]-values), axis=1) > 1):
            counts['inverse_component'] += 1
            continue
        first = int(np.searchsorted(values, frames[0], side='right')-1)
        last = int(np.searchsorted(values, frames[-1], side='left'))
        image_left, image_right = scan_left+left+first, scan_left+left+last
        if not (center-125 <= image_left <= center <= image_right <= center+125):
            raise AssertionError('Inverse footprint outside frozen bounds')
        rows.append(dict(tactical_center=center, broadcast_center=source_center,
                         component_left=scan_left+left, component_right=scan_left+right-1,
                         image_left=image_left, image_right=image_right,
                         decision_left=decision_left, decision_right=decision_right))
        counts['eligible'] += 1
    return rows, counts


def query_plan(rows, mapping, n):
    """Closest usable same-component candidate, with deterministic earlier ties."""
    center = np.full(n, -1, int)
    source = np.full(n, -1, int)
    distance = np.full(n, 26, int)
    for row in sorted(rows, key=lambda r:r['tactical_center']):
        c, b = row['tactical_center'], row['broadcast_center']
        use = np.arange(max(0, c-25, row['component_left']), min(n, c+26, row['component_right']+1))
        mapped = mapping['frames'][use]
        valid = (mapped >= b-100) & (mapped <= b+98)
        use = use[valid]
        take = np.abs(use-c) < distance[use]  # strict, preserves earlier tie.
        use = use[take]
        center[use], source[use], distance[use] = c, b, np.abs(use-c)
    return dict(tactical_center=center, broadcast_center=source,
                broadcast_frame=np.where(center >= 0, mapping['frames'][:n], -1), seen=center >= 0)


def reference_query_plan(rows, mapping, n, blocks):
    """Exact original Round22 nearest-center metadata, without reading heads."""
    centers = np.array(sorted(r['tactical_center'] for r in rows))
    by_center = {r['tactical_center']:r['broadcast_center'] for r in rows}
    frame = np.arange(n)
    right = np.minimum(np.searchsorted(centers, frame), len(centers)-1)
    left = np.maximum(right-1, 0)
    chosen = np.where(np.abs(frame-centers[left]) <= np.abs(frame-centers[right]), left, right)
    component = mapping_components(mapping, n, blocks)
    seen = (np.abs(frame-centers[chosen]) <= 25) & (component >= 0)
    seen &= component == component[centers[chosen]]
    tactical = np.where(seen, centers[chosen], -1)
    source = np.array([by_center.get(int(c), -1) for c in tactical])
    return dict(tactical_center=tactical, broadcast_center=source,
                broadcast_frame=np.where(seen, mapping['frames'][:n], -1), seen=seen)


def synthetic_feature_view(queries, mapping, n, blocks):
    """Only synthetic model outputs; no actual image/action cache is consumed."""
    columns = PROBABILITY+['sa_event_max', 'sa_entropy', 'sa_displacement']
    values = {key:np.full(n, np.nan) for key in columns}
    values['sa_seen'] = queries['seen'].astype(float)
    seen = queries['seen']
    first = queries['broadcast_center'][seen]-100
    position = (queries['broadcast_frame'][seen]-first)/2
    lo, hi = np.floor(position).astype(int), np.ceil(position).astype(int)
    alpha = position-lo
    # Same fixed source-frame function regardless of clip/target prefix.
    def probability(index):
        frame = first+2*index
        logits = np.sin(frame[:, None]/37 + np.arange(13)[None, :]*.7)
        logits += np.cos(frame[:, None]/19-np.arange(13)[None, :]*.2)
        p = np.exp(logits-logits.max(axis=1, keepdims=True))
        return p/p.sum(axis=1, keepdims=True)
    p = probability(lo)*(1-alpha[:, None])+probability(hi)*alpha[:, None]
    for i, key in enumerate(PROBABILITY):
        values[key][seen] = p[:, i]
    values['sa_event_max'][seen] = p[:, 1:].max(axis=1)
    values['sa_entropy'][seen] = -(p*np.log(np.maximum(p, 1e-30))).sum(axis=1)
    values['sa_displacement'][seen] = np.sin((first+2*lo)/11)*(1-alpha)+np.sin((first+2*hi)/11)*alpha
    dense = pd.DataFrame(values).astype(np.float32)
    return feature_view(dense, mapping_components(mapping, n, blocks))


def coverage_diagnostics(queries, original, rows, counts, public_start=11185, public_stop=18685):
    mask = queries['seen'][public_start:public_stop]
    before = original['seen'][public_start:public_stop]
    windows = []
    for length in (25, 50, 75, 125):
        fractions = np.array([mask[start:start+length].mean() for start in range(0, len(mask)-length+1, 25)])
        windows.append(dict(frames=length, windows=len(fractions), coverage_ge50=int((fractions >= .5).sum()),
                            coverage_ge80=int((fractions >= .8).sum()), fully_observed=int((fractions == 1).sum())))
    return dict(eligible_associations=len(rows), unique_source_centers=len({r['broadcast_center'] for r in rows}),
                observed_public_frames=int(mask.sum()), public_frames=len(mask),
                additional_public_frames=int((mask & ~before).sum()), lost_public_frames=int((before & ~mask).sum()),
                net_additional_public_frames=int(mask.sum()-before.sum()),
                original_coverage_retention=float((mask & before).sum()/before.sum()),
                eligibility_counts=counts, fixed_label_free_windows=windows)


def opportunity_gate(summary, rule, parity):
    checks = dict(coverage=summary['observed_public_frames'] >= rule['min_observed_frames'],
                  added=summary['net_additional_public_frames'] >= rule['min_net_additional_frames'],
                  retained=summary['original_coverage_retention'] >= rule['min_original_coverage_retention'],
                  safety_and_prefix=bool(parity))
    return dict(checks=checks, passed=all(checks.values()))
