"""Local projective background compensation for moving-ball visual support."""

import numpy as np

from ball_appearance import FEATURES, HEIGHT, WIDTH, image_geometry, negative_positions
from ball_pooled import pooled_one

MOTION_FEATURES = 206


def bilinear_sample(image, x, y):
    if not (np.isfinite(x).all() and np.isfinite(y).all() and (x >= 0).all() and
            (x < WIDTH-1).all() and (y >= 0).all() and (y < HEIGHT-1).all()):
        return None
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = (x-left)[..., None], (y-top)[..., None]
    return ((1-dx)*(1-dy)*image[top, left]+dx*(1-dy)*image[top, left+1]+
            (1-dx)*dy*image[top+1, left]+dx*dy*image[top+1, left+1])/255


def vector_features(rgb, residual=None):
    ring = np.ones((16, 16), bool)
    ring[2:-2, 2:-2] = False
    background = np.median(rgb[ring], axis=0)
    if residual is None:
        signal = np.clip((rgb-background)/.25, -2, 2)
        grass = (rgb[:, :, 1] > 1.15*rgb[:, :, 0]) & (rgb[:, :, 1] > 1.05*rgb[:, :, 2])
        fraction = (~grass).mean()
    else:
        signal = np.clip(residual/.25, -2, 2)
        rgb = residual
        background = np.median(rgb[ring], axis=0)
        fraction = (rgb.mean(axis=2) > .15).mean()
    center = rgb[6:10, 6:10].reshape(-1, 3)
    value = np.r_[signal.ravel(), background, center.mean(axis=0), center.std(axis=0), fraction]
    if value.shape != (FEATURES,):
        raise AssertionError('Invalid temporal patch vector')
    return value


def compensated_features(images, transfers, x, y, diameter, profile):
    radius = diameter*profile
    axis = (np.arange(16)+.5)*2*radius/16-radius
    xx, yy = np.meshgrid(x+axis, y+axis)
    points = np.stack((xx.ravel(), yy.ravel(), np.ones(256)))
    patches = []
    for image, matrix in zip(images, transfers):
        mapped = matrix@points
        sx, sy = mapped[0]/mapped[2], mapped[1]/mapped[2]
        patch = bilinear_sample(image, sx.reshape(16, 16), sy.reshape(16, 16))
        if patch is None:
            return None
        patches.append(patch)
    ring = np.ones((16, 16), bool)
    ring[2:-2, 2:-2] = False
    centered = [patch-np.median(patch[ring], axis=0) for patch in patches]
    difference = centered[1]-(centered[0]+centered[2])/2
    value = np.r_[pooled_one(vector_features(patches[1])), pooled_one(vector_features(patches[1], difference))]
    if value.shape != (MOTION_FEATURES,) or not np.isfinite(value).all():
        raise AssertionError('Invalid compensated visual feature schema')
    return value.astype(np.float32)


def motion_frame_features(images, matrices, ball, positions, profiles, training):
    result = np.full((len(profiles), 11, MOTION_FEATURES), np.nan, np.float32)
    target = np.full(11, -1, dtype=int)
    if not np.isfinite(matrices).all():
        return result, target
    try:
        inverse = np.linalg.inv(matrices[1])
    except np.linalg.LinAlgError:
        return result, target
    transfers = [matrix@inverse for matrix in matrices]
    geometry = image_geometry(matrices[1], ball)
    if geometry is None:
        return result, target
    x, y, diameter = geometry
    jitter = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] if training else [(0, 0)]
    crops = [(x+dx, y+dy, diameter, 1) for dx, dy in jitter]
    for point in negative_positions(ball, positions):
        geometry = image_geometry(matrices[1], point)
        if geometry is not None:
            crops.append((*geometry, 0))
    for index, (x, y, diameter, label) in enumerate(crops):
        target[index] = label
        for profile_index, profile in enumerate(profiles):
            value = compensated_features(images, transfers, x, y, diameter, profile)
            if value is not None:
                result[profile_index, index] = value
    return result, target
