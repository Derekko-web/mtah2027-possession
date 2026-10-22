"""Translation-tolerant ball-spot statistics from local normalized weak crops."""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def blob_map(gray, sigma):
    xx = gaussian_filter(gray, sigma, order=(0, 2), mode='nearest')
    yy = gaussian_filter(gray, sigma, order=(2, 0), mode='nearest')
    xy = gaussian_filter(gray, sigma, order=(1, 1), mode='nearest')
    trace = xx+yy
    difference = np.sqrt((xx-yy)**2+4*xy**2)
    low, high = (trace-difference)/2, (trace+difference)/2
    strength = np.maximum(-high, 0)*sigma**2
    ratio = np.maximum(-high, 0)/np.maximum(-low, 1e-9)
    return strength, ratio


def pooled_one(features):
    rgb = np.asarray(features[:768], float).reshape(16, 16, 3)
    gray = rgb.mean(axis=2)
    values = [np.quantile(rgb, [.05, .25, .5, .75, .95], axis=(0, 1)).ravel(),
              rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)), features[768:]]
    yy, xx = np.indices(gray.shape)
    for sigma in (.5, 1., 2.):
        dog = gaussian_filter(gray, sigma, mode='nearest')-gaussian_filter(gray, 2*sigma, mode='nearest')
        strength, ratio = blob_map(gray, sigma)
        for field in (dog, strength):
            values.append(np.r_[np.quantile(field, [.5, .9, .99]), field.max(), np.maximum(field, 0).mean(), field.std()])
        peaks = (strength == maximum_filter(strength, size=3, mode='nearest')) & (strength > 0)
        candidates = np.flatnonzero(peaks)
        candidates = candidates[np.argsort(strength.ravel()[candidates])[::-1]][:3]
        summary = np.zeros((3, 3))
        for index, point in enumerate(candidates):
            row, col = np.unravel_index(point, gray.shape)
            summary[index] = [strength[row, col], ratio[row, col], gray[row, col]]
        values.append(summary.ravel())
        mass = np.maximum(dog, 0)
        total = mass.sum()
        if total > 1e-9:
            cx, cy = (mass*xx).sum()/total, (mass*yy).sum()/total
            vx, vy = (mass*(xx-cx)**2).sum()/total, (mass*(yy-cy)**2).sum()/total
            cov = (mass*(xx-cx)*(yy-cy)).sum()/total
            eigen = np.linalg.eigvalsh([[vx, cov], [cov, vy]])
            values.append(np.r_[eigen, eigen[0]/max(eigen[1], 1e-9)])
        else:
            values.append(np.zeros(3))
    result = np.concatenate(values).astype(np.float32)
    if not np.isfinite(result).all():
        raise AssertionError('Invalid pooled appearance features')
    return result


def pooled_features(features):
    return np.stack([pooled_one(row) for row in features])
