"""Local directional ball/player interaction pooling, without player identities."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from build_candidate import PARAMS, objective_weights
from control_refinement import contact_support
from motion_dead_trim import decode_motion_dead_trim
from pair_candidate import codes
from pair_motion import enclosure_mask
from rich_features import swap_teams
from weak_ownership import ownership_prediction
from xgboost_baseline import CLASSES


def matched_velocity(current, other, duration, valid):
    """Reject ambiguous correspondence rather than trusting slot continuity."""
    cost = np.linalg.norm(current[:, :, None]-other[:, None], axis=3)
    cost[~np.isfinite(cost)] = np.inf
    match = cost.argmin(axis=2)
    reverse = cost.argmin(axis=1)
    first = np.sort(cost, axis=2)[:, :, :2]
    margin = np.full_like(first[:, :, 0], np.nan)
    np.subtract(first[:, :, 1], first[:, :, 0], out=margin,
                where=np.isfinite(first[:, :, 0]))
    rows = np.arange(len(current))[:, None]
    good = (reverse[rows, match] == np.arange(current.shape[1])[None])
    good &= (margin >= .25) & (first[:, :, 0]/abs(duration) <= 15) & valid[:, None]
    velocity = (current-other[rows, match])/duration
    velocity[~good] = np.nan
    return velocity


def weighted_pool(value, distance, bandwidth):
    use = np.isfinite(value) & np.isfinite(distance) & (distance <= 10)
    weight = np.where(use, np.exp(-.5*(distance/bandwidth)**2), 0)
    denominator = weight.sum(axis=1)
    result = np.full(len(value), np.nan)
    np.divide((np.where(use, value, 0)*weight).sum(axis=1), denominator,
              out=result, where=denominator > 0)
    return result


def native_graph(source, frames):
    frames = np.asarray(frames, dtype=int)
    positions, ball, teams = source['positions'], source['ball'], source['teams']
    n = len(positions)
    bad = np.r_[0, np.cumsum(~source['valid'])]
    result = {}
    info = {}
    for start in range(0, len(frames), 256):
        query = frames[start:start+256]
        lo, hi = np.clip(query-5, 0, n-1), np.clip(query+5, 0, n-1)
        valid = (query >= 5) & (query+5 < n) & (bad[hi+1]-bad[lo] == 0)
        incoming, outgoing = (ball[query]-ball[lo])/.2, (ball[hi]-ball[query])/.2
        for velocity in (incoming, outgoing):
            velocity[(np.linalg.norm(velocity, axis=1) > 80) | ~valid] = np.nan
        speed_in, speed_out = np.linalg.norm(incoming, axis=1), np.linalg.norm(outgoing, axis=1)
        turn = np.full(len(query), np.nan)
        np.divide((incoming*outgoing).sum(axis=1), speed_in*speed_out,
                  out=turn, where=(speed_in > .1) & (speed_out > .1))
        values = dict(ball_turn=turn)
        observed = np.isfinite(incoming).all(axis=1) & np.isfinite(outgoing).all(axis=1)
        for team, prefix in ((0, 'h__'), (1, 'a__')):
            xy = positions[query][:, teams == team]
            opponent = positions[query][:, teams == 1-team]
            offset = xy-ball[query, None]
            distance = np.linalg.norm(offset, axis=2)
            safe = np.where(np.isfinite(distance), distance, np.inf)
            nearest = safe.min(axis=1)
            nearest[~np.isfinite(nearest) | ~valid] = np.nan
            info.setdefault(prefix+'distance1', np.full(len(frames), np.nan))[start:start+len(query)] = nearest
            unit = offset/np.maximum(distance[:, :, None], 1e-6)
            past = matched_velocity(xy, positions[lo][:, teams == team], .2, valid)
            future = matched_velocity(xy, positions[hi][:, teams == team], -.2, valid)
            observed &= (np.isfinite(past).all(axis=2) & np.isfinite(future).all(axis=2)).any(axis=1)
            fields = {}
            for name, bv, pv, lower, upper in (
                    ('in', incoming, past, -1, 0), ('out', outgoing, future, 0, 1)):
                relative = bv[:, None]-pv
                fields['radial_'+name] = (relative*unit).sum(axis=2)
                fields['tangent_'+name] = np.abs(relative[:, :, 0]*unit[:, :, 1]-relative[:, :, 1]*unit[:, :, 0])
                denominator = np.linalg.norm(bv, axis=1)[:, None]*np.linalg.norm(pv, axis=2)
                heading = np.full_like(distance, np.nan)
                np.divide((bv[:, None]*pv).sum(axis=2), denominator, out=heading, where=denominator > .01)
                fields['heading_'+name] = heading
                time = (offset*relative).sum(axis=2)/np.maximum((relative*relative).sum(axis=2), 1e-6)
                fields['closest_'+name] = np.linalg.norm(offset-relative*np.clip(time, lower, upper)[:, :, None], axis=2)
            fields['impulse_radial'] = ((outgoing-incoming)[:, None]*unit).sum(axis=2)
            pressure = np.linalg.norm(xy[:, :, None]-opponent[:, None], axis=3)
            pressure = np.min(np.where(np.isfinite(pressure), pressure, np.inf), axis=2)
            pressure[~np.isfinite(pressure)] = np.nan
            fields['pressure'] = pressure
            support = np.linalg.norm(xy[:, :, None]-xy[:, None], axis=3)
            support[:, np.arange(xy.shape[1]), np.arange(xy.shape[1])] = np.nan
            fields['support'] = np.sum(support <= 5, axis=2).astype(float)
            fields['support'][~np.isfinite(xy).all(axis=2)] = np.nan
            for bandwidth in (2, 5):
                for name, field in fields.items():
                    pooled = weighted_pool(field, distance, bandwidth)
                    pooled[~valid] = np.nan
                    values[prefix+name+f'_b{bandwidth}'] = pooled
        values['seen'] = observed.astype(float)
        for name, value in values.items():
            result.setdefault(name, np.full(len(frames), np.nan))[start:start+len(query)] = value
    info['seen'] = result['seen']
    return pd.DataFrame(result).astype(np.float32), pd.DataFrame(info).astype(np.float32)


def finish_graph(features, prefix):
    rename = lambda name: name[:3]+prefix+name[3:] if name.startswith(('h__', 'a__')) else prefix+name
    result = features.rename(columns=rename)
    extra = {}
    for team in ('h__', 'a__'):
        for name in ('closest_out_b2', 'impulse_radial_b2', 'pressure_b2'):
            column = result[team+prefix+name]
            for radius in (25, 75):
                extra[team+prefix+name+f'_past{radius}'] = column.rolling(radius+1, min_periods=1).mean()
                extra[team+prefix+name+f'_future{radius}'] = column.iloc[::-1].rolling(radius+1, min_periods=1).mean().iloc[::-1]
    return result.join(pd.DataFrame(extra)).astype(np.float32)


def graph_views(tactical, broadcast, mapping):
    n = len(tactical['positions'])
    tg, ti = native_graph(tactical, np.arange(n))
    frames = mapping['frames']
    queries = np.unique(frames[frames >= 0])
    if not len(queries):
        raise ValueError('No aligned broadcast observations')
    bg, bi = native_graph(broadcast, queries)
    allow = (frames >= 0) & enclosure_mask(mapping, 5, 10)
    aligned = []
    for native in (bg, bi):
        table = pd.DataFrame(np.nan, index=np.arange(n), columns=native.columns, dtype=np.float32)
        table.loc[allow] = native.iloc[np.searchsorted(queries, frames[allow])].to_numpy()
        table.loc[~allow, 'seen'] = 0
        aligned.append(table)
    rename_info = lambda table, prefix: table.rename(columns={name:(name[:3]+prefix+name[3:] if name.startswith(('h__', 'a__')) else prefix+name) for name in table})
    return finish_graph(tg, 'tg_').join(finish_graph(aligned[0], 'bg_')), rename_info(ti, 'tg_').join(rename_info(aligned[1], 'bg_'))


def fitting_weights(labels, train, transition):
    train = np.asarray(train, dtype=int)
    if not len(train) or np.any(np.diff(train) <= 0) or set(labels[train]) != set(CLASSES):
        raise ValueError('Training rows must be unique sorted eligible H/A/D')
    target = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
    y, weight = objective_weights(target, train, False)
    if transition:
        multiplier = np.ones(len(train))
        component = np.cumsum(np.r_[True, np.diff(train) != 1])
        changes = np.flatnonzero((np.diff(train) == 1) & (target[1:] != target[:-1]) & (target[1:] < 2) & (target[:-1] < 2))+1
        for index in changes:
            multiplier[(component == component[index]) & (np.abs(train-train[index]) <= 25)] = 4
        weight *= np.r_[multiplier, multiplier]
        for team in range(3):
            use = y == team
            weight[use] *= len(y)/(3*weight[use].sum())
    return y, weight


def fit_graph(inputs, labels, train, transition, seed):
    y, weight = fitting_weights(labels, train, transition)
    augmented = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
    model = XGBClassifier(**dict(PARAMS, max_depth=3, n_estimators=160, random_state=seed))
    model.fit(augmented, y, sample_weight=weight)
    return model, y, weight


def decode_graph(arrays, probability, bc, motion_info, info, raw, blocks, mode):
    observed = (info.tg_seen.to_numpy() == 1) | (info.bg_seen.to_numpy() == 1)
    if mode == 'replace_current':
        replacement = np.where(observed[:, None], probability, arrays[0])
        return decode_motion_dead_trim(replacement, *arrays[1:], bc, motion_info, raw, blocks)
    if mode != 'short_proposals':
        raise ValueError('Unknown graph decoder')
    reference = decode_motion_dead_trim(*arrays, bc, motion_info, raw, blocks)
    support = contact_support(raw).copy()
    for prefix in ('tg_', 'bg_'):
        h, a = (info[team+prefix+'distance1'].to_numpy() for team in ('h__', 'a__'))
        valid = info[prefix+'seen'].to_numpy() == 1
        for team, own, other in ((0, h, a), (1, a, h)):
            support[:, team] = np.maximum(support[:, team], valid & (own <= 3) & (other-own >= .5))
    conditional = probability[:, :2]/probability[:, :2].sum(axis=1, keepdims=True)
    result = reference.copy()
    for start, end in blocks:
        value = ownership_prediction(codes(reference[start:end]), conditional[start:end], support[start:end], np.zeros((end-start, 2)), 'short_proposals')
        use = observed[start:end]
        result[start:end][use] = CLASSES[value[use]]
    if not np.array_equal(result == 'D', reference == 'D'):
        raise AssertionError('Graph ownership proposals changed D')
    return result
