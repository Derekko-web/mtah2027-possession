"""Pure target-blind source schedule and mixed-parity broadcast extraction."""
import numpy as np
from sports_action_sampling import clip_frames, contiguous_ranges


def schedule(n, mapping, blocks, reserved, held, broadcast_n=164488):
    rows = []
    for center in range(11000, 18951, 50):
        left, right = center-150, center+150
        if left < 0 or right >= n or not any(a <= left and right <= b for _, a, b in blocks):
            continue
        if any(left <= r['end']+250 and right >= r['start']-250 for r in reserved):
            continue
        if np.any((np.asarray(held)+250 >= left) & (np.asarray(held)-250 <= right)):
            continue
        mapped = mapping['frames'][left:right+1]
        orientation = mapping['orientation'][left:right+1]
        if np.any(mapped < 0) or len(np.unique(orientation)) != 1 or np.any(np.diff(mapped) < 0):
            continue
        source = clip_frames(mapping['frames'][center])
        if source[0] < mapped.min() or source[-1] > mapped.max() or np.any(np.min(np.abs(source[:, None]-mapped), axis=1) > 1):
            continue
        if source[0] < 0 or source[-1] >= broadcast_n or np.any(np.diff(mapped) > 2):
            raise ValueError('Feasibility schedule violates broadcast bounds/cut guard')
        rows.append(dict(tactical_center=center, broadcast_center=int(mapping['frames'][center])))
    return rows


def source_centers(rows):
    """One inference per source center, associations retained in schedule rows."""
    return sorted({r['broadcast_center'] for r in rows})


def source_ranges(needed):
    """Overlapping odd/even runs are a union, never a global phase selector."""
    needed = np.unique(np.asarray(needed, dtype=int))
    return sorted(contiguous_ranges(needed[needed % 2 == 0])+contiguous_ranges(needed[needed % 2 == 1]))


def selector(needed):
    first = int(np.min(needed))
    clauses = [f'between(n,{a-first},{b-first})*not(mod(n-{a-first},2))' for a, b in source_ranges(needed)]
    return 'gt('+ '+'.join(clauses)+',0)'


def mapping_components(mapping, n, blocks):
    mapped = mapping['frames'][:n]
    orientation = mapping['orientation'][:n]
    valid = (mapped >= 0) & (mapped < 164488) & (orientation >= 0)
    half = np.full(n, -1, int)
    for i, (_, left, right) in enumerate(blocks):
        half[max(0, left):min(n, right+1)] = i
    valid &= half >= 0
    edge = np.r_[True, (~valid[:-1]) | (~valid[1:]) | (orientation[1:] != orientation[:-1]) |
                 (np.diff(mapped) < 0) | (np.diff(mapped) > 2) | (half[1:] != half[:-1])]
    components = np.cumsum(edge)
    components[~valid] = -1
    return components


def validate_video_metadata(metadata):
    if (metadata['width'], metadata['height'], int(metadata['nb_frames']), metadata['r_frame_rate'], float(metadata['start_time'])) != (1920,1080,164487,'25/1',0.):
        raise ValueError('Broadcast source metadata differs')
    return int(metadata['nb_frames'])
