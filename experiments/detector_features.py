"""Fallible local detector observations as bounded possession predictors."""

import numpy as np
import pandas as pd

from modern_ball import unletterbox_objects
from native_detector import image_box
from provided_speed import bounded_statistics
from starter.load_homography import unproject

PITCH = np.array([105.3, 68.])
PRESENCE = ['vd_crop_observed', 'vd_center_interpolated', 'vd_homography_error',
    'vd_s_ball_count', 'vd_m_ball_count', 'vd_s_score', 'vd_m_score',
    'vd_s_person_count', 'vd_m_person_count', 'vd_pair_count', 'vd_pair_seen',
    'vd_pair_score', 'vd_pair_separation', 'vd_pair_margin', 'vd_pair_consistency', 'vd_near_original']
GEOMETRY = ['vd_geometry_seen', 'vd_offset_norm', 'vd_upward_offset', 'vd_size_ratio', 'vd_person_foot_gap']
GEOMETRY += [team+'vd_'+field for team in ('h__', 'a__')
             for field in ('distance1', 'distance2', 'count2', 'count5', 'kernel', 'control_share')]
PRESENCE_SUMMARIES = ['vd_pair_seen', 'vd_pair_score', 'vd_s_score', 'vd_m_score']
GEOMETRY_SUMMARIES = ['h__vd_distance1', 'a__vd_distance1', 'h__vd_control_share',
                       'a__vd_control_share', 'vd_offset_norm', 'vd_upward_offset']


def candidates(prediction, record):
    balls, people = [], []
    for obj in unletterbox_objects(prediction['objects']):
        box = image_box(obj['box'], 256, (record['x'], record['y']))
        center = box[:2]+box[2:]/2
        scores = {v['label']:v['confidence'] for v in obj['labels']}
        if scores.get('sports ball', 0) >= .25 and (np.abs(center-[record['x'], record['y']]) <= 128).all():
            balls.append((scores['sports ball'], box))
        if scores.get('person', 0) >= .25:
            people.append(box[:2]+box[2:]*[.5, 1.])
    return balls, people


def consensus_pairs(first, second):
    if not first or not second:
        return []
    a = np.array([box[:2]+box[2:]/2 for _, box in first])
    b = np.array([box[:2]+box[2:]/2 for _, box in second])
    distance = np.linalg.norm(a[:, None]-b[None], axis=2)
    ab, ba = distance.argmin(axis=1), distance.argmin(axis=0)
    pairs = []
    for i, j in enumerate(ab):
        if ba[j] != i or distance[i, j] > 8:
            continue
        point = (a[i]+b[j])/2
        size = (np.sqrt(np.prod(first[i][1][2:]))+np.sqrt(np.prod(second[j][1][2:])))/2
        pairs.append(dict(score=min(first[i][0], second[j][0]), separation=float(distance[i, j]),
                          point=point, size=float(size)))
    return sorted(pairs, key=lambda p:(-p['score'], p['separation'], *p['point']))


def sample_features(record, first, second, positions, teams, matrix, error):
    result = {name:np.nan for name in PRESENCE+GEOMETRY}
    result.update(vd_crop_observed=1., vd_center_interpolated=float(record['interpolated']),
                  vd_homography_error=float(error), vd_pair_seen=0., vd_geometry_seen=0.,
                  vd_pair_score=0., vd_pair_margin=0.)
    a, pa = candidates(first, record)
    b, pb = candidates(second, record)
    for name, balls, people in [('s', a, pa), ('m', b, pb)]:
        result['vd_'+name+'_ball_count'] = len(balls)
        result['vd_'+name+'_person_count'] = len(people)
        result['vd_'+name+'_score'] = max((s for s, _ in balls), default=0.)
    pairs = consensus_pairs(a, b)
    result['vd_pair_count'] = len(pairs)
    result['vd_pair_consistency'] = len(pairs)/max(1, len(a), len(b))
    if not pairs:
        return result
    best = pairs[0]
    point = best['point']
    offset = point-[record['x'], record['y']]
    diameter = record['diameter']
    result.update(vd_pair_seen=1., vd_pair_score=best['score'], vd_pair_separation=best['separation'],
        vd_pair_margin=best['score']-(pairs[1]['score'] if len(pairs) > 1 else 0.),
        vd_offset_norm=float(np.linalg.norm(offset)/diameter), vd_upward_offset=float(-offset[1]/diameter),
        vd_size_ratio=best['size']/diameter)
    if not record['interpolated']:
        result['vd_near_original'] = float(np.linalg.norm(offset) <= max(4., min(12., 2*diameter)))
    if pa or pb:
        result['vd_person_foot_gap'] = float(np.min(np.linalg.norm(np.array(pa+pb)-point, axis=1))/diameter)
    try:
        nominal = np.array(unproject(matrix, *point))*PITCH
    except np.linalg.LinAlgError:
        return result
    if not np.isfinite(nominal).all() or (nominal < -5).any() or (nominal > PITCH+5).any():
        return result
    result['vd_geometry_seen'] = 1.
    kernels = []
    for team, prefix in ((0, 'h__'), (1, 'a__')):
        distance = np.linalg.norm(positions[teams == team]-nominal, axis=1)
        distance = np.sort(distance[np.isfinite(distance)])
        for index in (1, 2):
            result[prefix+f'vd_distance{index}'] = float(distance[index-1]) if len(distance) >= index else np.nan
        for radius in (2, 5):
            result[prefix+f'vd_count{radius}'] = float((distance <= radius).sum())
        kernel = float(np.exp(-.5*(distance/2)**2).sum()) if len(distance) else np.nan
        result[prefix+'vd_kernel'] = kernel
        kernels.append(kernel)
    denominator = sum(kernels)
    for prefix, kernel in zip(('h__', 'a__'), kernels):
        result[prefix+'vd_control_share'] = kernel/denominator if np.isfinite(denominator) and denominator > 0 else np.nan
    return result


def interpolate_samples(samples, n):
    result = pd.DataFrame(np.nan, index=np.arange(n), columns=samples.columns)
    samples = samples.loc[samples.index < n].sort_index()
    result.loc[samples.index] = samples.to_numpy()
    for left, right in zip(samples.index[:-1], samples.index[1:]):
        if right-left != 5:
            continue
        a, b = samples.loc[left].to_numpy(), samples.loc[right].to_numpy()
        good = np.isfinite(a) & np.isfinite(b)
        for frame in range(left+1, right):
            values = np.full(len(a), np.nan)
            values[good] = a[good]+(b[good]-a[good])*(frame-left)/5
            result.loc[frame] = values
    for name in ('vd_crop_observed', 'vd_pair_seen', 'vd_geometry_seen'):
        result[name] = result[name].fillna(0.)
    return result


def temporal_view(dense, columns, summary_columns, blocks):
    output = dense[columns].copy()
    extra = {}
    for column in summary_columns:
        for radius in (25, 75, 125):
            for direction in ('past', 'future'):
                values = np.full(len(dense), np.nan)
                for start, end in blocks:
                    start, end = max(0, start), min(len(dense), end)
                    if start < end:
                        values[start:end] = bounded_statistics(dense[column].iloc[start:end], radius, direction)[0]
                extra[column+f'_{direction}{radius}'] = values
    return output.join(pd.DataFrame(extra)).astype(np.float32)


def feature_views(samples, n, blocks):
    dense = interpolate_samples(samples, n)
    return dict(visual_presence=temporal_view(dense, PRESENCE, PRESENCE_SUMMARIES, blocks),
        visual_geometry=temporal_view(dense, PRESENCE+GEOMETRY, PRESENCE_SUMMARIES+GEOMETRY_SUMMARIES, blocks)), dense
