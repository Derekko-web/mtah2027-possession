"""Sparse broadcast-time interpolation and cut-bounded event features; no labels."""
from pathlib import Path
import numpy as np
import pandas as pd
from sports_action_features import PROBABILITY, add_summaries
from sports_action_sampling import clip_frames
from broadcast_action_sampling import mapping_components


def dense_actions(folder, records, rows, mapping, n, blocks):
    output = {name:np.full(n, np.nan) for name in PROBABILITY+['sa_event_max','sa_entropy','sa_displacement']}
    output['sa_seen'] = np.zeros(n)
    chosen_source = np.full(n, -1, int)
    components = mapping_components(mapping, n, blocks)
    rows = sorted(rows, key=lambda r:r['tactical_center'])
    if not rows:
        return pd.DataFrame(output).astype(np.float32), chosen_source, components
    centers = np.array([r['tactical_center'] for r in rows])
    query = np.arange(n)
    right = np.minimum(np.searchsorted(centers, query), len(centers)-1)
    left = np.maximum(right-1, 0)
    chosen = np.where(np.abs(query-centers[left]) <= np.abs(query-centers[right]), left, right)
    valid = (np.abs(query-centers[chosen]) <= 25) & (components >= 0)
    valid &= components == components[centers[chosen]]
    by_source = {r['center']:r for r in records}
    if len(by_source) != len(records):
        raise ValueError('Duplicate stored broadcast inference')
    for i, row in enumerate(rows):
        use = np.flatnonzero(valid & (chosen == i))
        if not len(use):
            continue
        source_center = row['broadcast_center']
        with np.load(Path(folder)/by_source[source_center]['file']) as data:
            frames = clip_frames(source_center)
            np.testing.assert_array_equal(data['frames'], frames)
            source_query = mapping['frames'][use]
            if np.any(source_query < frames[0]) or np.any(source_query > frames[-1]):
                raise ValueError('Mapped query outside broadcast clip')
            logits = data['im_feat'][:, :13].astype(float)
            probability = np.exp(logits-logits.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
            position = (source_query-frames[0])/2
            lo, hi = np.floor(position).astype(int), np.ceil(position).astype(int)
            alpha = position-lo
            p = probability[lo]*(1-alpha[:, None])+probability[hi]*alpha[:, None]
            for j, name in enumerate(PROBABILITY):
                output[name][use] = p[:, j]
            output['sa_event_max'][use] = p[:, 1:].max(axis=1)
            output['sa_entropy'][use] = -(p*np.log(np.maximum(p, 1e-30))).sum(axis=1)
            displacement = data['displ_feat']
            output['sa_displacement'][use] = displacement[lo]*(1-alpha)+displacement[hi]*alpha
            output['sa_seen'][use] = 1
            chosen_source[use] = source_center
    return pd.DataFrame(output).astype(np.float32), chosen_source, components


def feature_view(dense, components):
    seen = dense.sa_seen.to_numpy() > 0
    # Each summary run contains only observed queries in a single alignment component.
    edge = np.r_[0, np.flatnonzero((seen[1:] != seen[:-1]) | (components[1:] != components[:-1]))+1, len(dense)]
    runs = [(int(a), int(b)) for a, b in zip(edge[:-1], edge[1:]) if seen[a]]
    return add_summaries(dense, PROBABILITY[1:]+['sa_event_max'], runs)


def support_summary(dense, sources, rule):
    a, b = rule['public_start'], rule['public_stop']
    public = dense.iloc[a:b]
    value = np.nan_to_num(public.sa_event_max.to_numpy(float), nan=0.)
    candidates = np.flatnonzero((value >= rule['peak_threshold']) & (value > np.r_[-np.inf, value[:-1]]) & (value >= np.r_[value[1:], -np.inf]))
    peaks = []
    for i in sorted(candidates, key=lambda i:(-value[i], i)):
        if all(abs(i-old) >= rule['peak_separation'] for old in peaks):
            peaks.append(int(i))
    seen = public.sa_seen.to_numpy() > 0
    variation = float(value[seen].std()) if seen.any() else 0.
    unique = len(set(sources[a+np.array(peaks, dtype=int)].tolist())-{-1})
    checks = dict(coverage=int(seen.sum()) >= rule['min_observed_frames'],
                  peaks=len(peaks) >= rule['min_confident_local_peaks'],
                  unique_peak_clips=unique >= rule['min_unique_peak_source_clips'],
                  variation=variation >= rule['min_observed_event_max_std'])
    return dict(observed_public_frames=int(seen.sum()), public_frames=b-a,
                confident_local_peaks=len(peaks), unique_peak_source_clips=unique,
                observed_event_max_std=variation,
                mean_observed_entropy=float(public.sa_entropy[seen].mean()) if seen.any() else None,
                checks=checks, support_passed=all(checks.values()),
                eligible_heads=['events_only'] if all(checks.values()) else [])
