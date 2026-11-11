"""Label-free bounded crop centers; original tracking stays unchanged."""
import numpy as np


def crop_centers(ball, blocks, maximum_gap=25):
    ball = np.asarray(ball, dtype=float)
    result = np.full_like(ball, np.nan)
    interpolated = np.zeros(len(ball), bool)
    valid = np.isfinite(ball).all(axis=1) & (ball >= -5).all(axis=1) & (ball <= [110.3, 73.]).all(axis=1)
    for start, end in blocks:
        start, end = max(0, start), min(len(ball), end)
        observed = np.flatnonzero(valid[start:end])+start
        result[observed] = ball[observed]
        for left, right in zip(observed[:-1], observed[1:]):
            if right-left > 2*maximum_gap:
                continue
            query = np.arange(left+1, right)
            query = query[(query-left <= maximum_gap) & (right-query <= maximum_gap)]
            weight = (query-left)/(right-left)
            result[query] = ball[left]+weight[:, None]*(ball[right]-ball[left])
            interpolated[query] = True
    return result, interpolated
