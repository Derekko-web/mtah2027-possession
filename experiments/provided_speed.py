"""Supplied ball-speed observations and selected8-compatible frozen decoders."""

import numpy as np
import pandas as pd

from control_refinement import contact_support
from motion_dead_trim import decode_motion_dead_trim
from pair_candidate import codes
from pair_motion import enclosure_mask
from weak_ownership import ownership_prediction
from xgboost_baseline import CLASSES


def native_speed_core(source, scalar, unit_factor, frames):
    frames = np.asarray(frames, dtype=int)
    n = len(source['ball'])
    given = np.asarray(scalar, dtype=float)[frames]*unit_factor
    given[(given < 0) | ~source['valid'][frames]] = np.nan
    seen = np.isfinite(given)
    lo, hi = np.maximum(frames-5, 0), np.minimum(frames+5, n-1)
    bad = np.r_[0, np.cumsum(~source['valid'])]
    valid = (frames >= 5) & (frames+5 < n) & (bad[hi+1]-bad[lo] == 0)
    geometric = np.linalg.norm(source['ball'][hi].astype(float)-source['ball'][lo], axis=1)/.4
    geometric[~valid | (geometric > 80)] = np.nan
    values = dict(given=given, seen=seen.astype(float),
        without_xy=(seen & ~np.isfinite(source['ball'][frames]).all(axis=1)).astype(float),
        coordinate_speed5=geometric, given_minus_coordinate=given-geometric)
    for team, prefix in ((0, 'h__'), (1, 'a__')):
        xy = source['positions'][frames][:, source['teams'] == team]
        speed = source['speeds'][frames][:, source['teams'] == team].astype(float)*unit_factor
        speed[speed < 0] = np.nan
        distance = np.linalg.norm(xy-source['ball'][frames, None], axis=2)
        safe = np.where(np.isfinite(distance), distance, np.inf)
        nearest = safe.argmin(axis=1)
        near_speed = speed[np.arange(len(frames)), nearest].copy()
        near_speed[~np.isfinite(safe.min(axis=1))] = np.nan
        values[prefix+'nearest_signed_gap'] = given-near_speed
        values[prefix+'nearest_abs_gap'] = np.abs(given-near_speed)
        gap = np.abs(given[:, None]-speed)
        for radius in (3, 5):
            minimum = np.min(np.where(np.isfinite(gap) & (distance <= radius), gap, np.inf), axis=1)
            minimum[~np.isfinite(minimum)] = np.nan
            values[prefix+f'gap{radius}'] = minimum
    return pd.DataFrame(values)


def bounded_statistics(values, radius, direction):
    """Each fixed window has a fixed summation order, independent of cache length."""
    values = np.asarray(values, dtype=float)
    padded = np.pad(values, (radius, 0) if direction == 'past' else (0, radius), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, radius+1)
    finite = np.isfinite(windows)
    count = finite.sum(axis=1)
    mean = np.full(len(values), np.nan)
    np.divide(np.where(finite, windows, 0).sum(axis=1), count, out=mean, where=count > 0)
    variance = np.full(len(values), np.nan)
    np.divide(np.where(finite, (windows-mean[:, None])**2, 0).sum(axis=1), count, out=variance, where=count > 0)
    available = np.minimum(np.arange(len(values))+1, radius+1)
    if direction == 'future':
        available = available[::-1]
    return mean, np.sqrt(variance), count/available


def finish_speed(core, prefix, blocks):
    n = len(core)
    result = pd.DataFrame(np.nan, index=np.arange(n), columns=core.columns)
    extra = {}
    for start, end in blocks:
        start, end = max(0, start), min(n, end)
        if start >= end:
            continue
        result.iloc[start:end] = core.iloc[start:end].to_numpy()
        for radius in (25, 75, 125):
            for direction in ('past', 'future'):
                stats = bounded_statistics(core.given.iloc[start:end], radius, direction)
                for name, values in zip(('mean', 'std', 'fraction'), stats):
                    column = f'given_{direction}{radius}_{name}'
                    extra.setdefault(column, np.full(n, np.nan))[start:end] = values
        for team in ('h__', 'a__'):
            for direction in ('past', 'future'):
                values = bounded_statistics(core[team+'gap5'].iloc[start:end], 25, direction)[0]
                extra.setdefault(team+f'gap5_{direction}25', np.full(n, np.nan))[start:end] = values
    result[['seen', 'without_xy']] = result[['seen', 'without_xy']].fillna(0.)
    result = result.join(pd.DataFrame(extra))
    rename = lambda name:name[:3]+prefix+name[3:] if name.startswith(('h__', 'a__')) else prefix+name
    return result.rename(columns=rename).astype(np.float32)


def speed_views(tactical, broadcast, tactical_scalar, broadcast_scalar, tactical_factor, broadcast_factor, mapping, blocks):
    n = len(tactical['ball'])
    tactical_core = native_speed_core(tactical, tactical_scalar, tactical_factor, np.arange(n))
    frames = mapping['frames']
    queries = np.unique(frames[frames >= 0])
    if not len(queries):
        raise ValueError('No aligned broadcast frames')
    native = native_speed_core(broadcast, broadcast_scalar, broadcast_factor, queries)
    allow = (frames >= 0) & enclosure_mask(mapping, 5, 10)
    aligned = pd.DataFrame(np.nan, index=np.arange(n), columns=native.columns)
    aligned.loc[allow] = native.iloc[np.searchsorted(queries, frames[allow])].to_numpy()
    aligned.loc[~allow, ['seen', 'without_xy']] = 0.
    t = finish_speed(tactical_core, 'ps_t_', blocks)
    b = finish_speed(aligned, 'ps_b_', blocks)
    cross = pd.DataFrame(dict(ps_given_difference=t.ps_t_given-b.ps_b_given,
        ps_given_absolute_difference=np.abs(t.ps_t_given-b.ps_b_given),
        ps_both_given=((t.ps_t_seen > 0) & (b.ps_b_seen > 0)).astype(float),
        ps_given_xor=((t.ps_t_seen > 0) ^ (b.ps_b_seen > 0)).astype(float)))
    fused = t.join(b).join(cross).astype(np.float32)
    info = pd.DataFrame(dict(tactical_seen=t.ps_t_seen, broadcast_seen=b.ps_b_seen,
        tactical_context=t[['ps_t_given_past125_fraction', 'ps_t_given_future125_fraction']].max(axis=1),
        broadcast_context=b[['ps_b_given_past125_fraction', 'ps_b_given_future125_fraction']].max(axis=1))).astype(np.float32)
    return dict(tactical_speed=t, broadcast_speed=b, fused_speed=fused), info


def decode_provided_speed(arrays, probability, bc, motion_info, raw, blocks, mode):
    reference = decode_motion_dead_trim(*arrays, bc, motion_info, raw, blocks)
    if mode == 'live_replace':
        conditional = probability[:, :2]/probability[:, :2].sum(axis=1, keepdims=True)
        support = contact_support(raw)
        result = reference.copy()
        for start, end in blocks:
            value = ownership_prediction(codes(reference[start:end]), conditional[start:end], support[start:end],
                np.zeros((end-start, 2)), 'replace_live')
            result[start:end] = CLASSES[value]
        if not np.array_equal(result == 'D', reference == 'D'):
            raise AssertionError('Provided-speed live decoder changed D')
        return result
    alternative = decode_motion_dead_trim(probability, *arrays[1:], bc, motion_info, raw, blocks)
    if mode == 'replace_current':
        return alternative
    if mode != 'dead_trim':
        raise ValueError('Unknown provided-speed decoder')
    result = reference.copy()
    use = (reference == 'D') & np.isin(alternative, ['H', 'A'])
    result[use] = alternative[use]
    if not np.array_equal(result[reference != 'D'], reference[reference != 'D']):
        raise AssertionError('Provided-speed dead trim changed an existing live/X frame')
    if ((result == 'D') & (reference != 'D')).any():
        raise AssertionError('Provided-speed dead trim added D')
    return result
