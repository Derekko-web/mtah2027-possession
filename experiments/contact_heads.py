"""Bounded histories of cached fallible optical/color contact scores."""
import numpy as np
import pandas as pd
from motion_dead_trim import decode_motion_dead_trim
from provided_speed import bounded_statistics, decode_provided_speed


def contact_view(scores, name, n, blocks):
    scores = np.asarray(scores)[:n]
    dense = np.full((n, 2), np.nan)
    for left, right in blocks:
        left, right = max(left, 0), min(right, n)
        if left >= right:
            continue
        frames = np.flatnonzero(np.isfinite(scores[left:right]).all(axis=1))+left
        dense[frames] = scores[frames]
        for a, b in zip(frames[:-1], frames[1:]):
            if b-a == 5:
                alpha = np.arange(6)/5
                dense[a:b+1] = scores[a]+alpha[:, None]*(scores[b]-scores[a])
    result = {f'cc_{name}_seen':np.isfinite(dense).all(axis=1).astype(float)}
    for team, prefix in enumerate(('h__', 'a__')):
        key = prefix+'cc_'+name
        result[key+'_score'] = dense[:, team]
        for radius in (25, 75, 125):
            for direction in ('past', 'future'):
                mean, fraction = np.full(n, np.nan), np.zeros(n)
                for left, right in blocks:
                    left, right = max(0, left), min(n, right)
                    if left < right:
                        values = bounded_statistics(dense[left:right, team], radius, direction)
                        mean[left:right], fraction[left:right] = values[0], values[2]
                result[key+f'_{direction}{radius}_mean'] = mean
                result[key+f'_{direction}{radius}_fraction'] = fraction
    return pd.DataFrame(result).astype(np.float32)


def feature_views(contact, n, blocks):
    metadata = contact_view(contact['detected_metadata'], 'metadata', n, blocks)
    color = metadata.join(contact_view(contact['tracked_color'], 'tracked_color', n, blocks))
    color = color.join(contact_view(contact['detected_color'], 'detected_color', n, blocks))
    return dict(contact_metadata=metadata, contact_color=color)


def decode_contact_head(arrays, probability, bc, info, raw, blocks, mode):
    if mode == 'live_replace':
        return decode_provided_speed(arrays, probability, bc, info, raw, blocks, mode)
    changed = list(arrays)
    changed[4] = probability
    alternative = decode_motion_dead_trim(*changed, bc, info, raw, blocks)
    if mode == 'joint_slot':
        return alternative
    if mode != 'dead_trim':
        raise ValueError('Unknown contact-head decoder')
    reference = decode_motion_dead_trim(*arrays, bc, info, raw, blocks)
    result = reference.copy()
    use = (reference == 'D') & np.isin(alternative, ['H', 'A'])
    result[use] = alternative[use]
    if not np.array_equal(result[reference != 'D'], reference[reference != 'D']):
        raise AssertionError('Contact head changed existing H/A/X')
    if ((result == 'D') & (reference != 'D')).any():
        raise AssertionError('Contact head added D')
    return result
