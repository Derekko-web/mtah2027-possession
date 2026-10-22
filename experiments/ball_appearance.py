"""Local ball-patch features and excluded weak sampling; no visual output."""

import subprocess

import numpy as np
from PIL import Image

from games import require
from starter.load_homography import project

WIDTH, HEIGHT = 1920, 1080
PITCH = np.array([105.3, 68.0])
FEATURES = 778


def decode_local_frame(frame):
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-ss', f'{int(frame)/25:.8f}',
               '-i', require(['video'])[0], '-frames:v', '1', '-filter_threads', '1',
               '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    result = subprocess.run(command, capture_output=True)
    if result.returncode or len(result.stdout) != WIDTH*HEIGHT*3:
        raise ValueError('Local full-resolution tactical frame decode failed')
    return np.frombuffer(result.stdout, np.uint8).reshape(HEIGHT, WIDTH, 3)


def image_geometry(matrix, position):
    if not np.isfinite(position).all() or not np.isfinite(matrix).all():
        return None
    point = np.asarray(position)/PITCH
    x, y = project(matrix, *point)
    xx, xy = project(matrix, point[0]+1/PITCH[0], point[1])
    yx, yy = project(matrix, point[0], point[1]+1/PITCH[1])
    jacobian = np.array([[xx-x, yx-x], [xy-y, yy-y]])
    if not np.isfinite(jacobian).all():
        return None
    diameter = np.clip(.22*np.linalg.svd(jacobian, compute_uv=False)[0], 2, 12)
    return float(x), float(y), float(diameter)


def patch_features(image, x, y, diameter, profile):
    radius = diameter*profile
    if x-radius < 0 or y-radius < 0 or x+radius > WIDTH or y+radius > HEIGHT:
        return None
    patch = np.asarray(image.transform((16, 16), Image.Transform.AFFINE,
        (2*radius/16, 0, x-radius, 0, 2*radius/16, y-radius), Image.Resampling.BILINEAR), dtype=float)/255
    ring = np.ones((16, 16), bool)
    ring[2:-2, 2:-2] = False
    background = np.median(patch[ring], axis=0)
    residual = np.clip((patch-background)/.25, -2, 2)
    center = patch[6:10, 6:10].reshape(-1, 3)
    grass = (patch[:, :, 1] > 1.15*patch[:, :, 0]) & (patch[:, :, 1] > 1.05*patch[:, :, 2])
    result = np.r_[residual.ravel(), background, center.mean(axis=0), center.std(axis=0), (~grass).mean()]
    if len(result) != FEATURES or not np.isfinite(result).all():
        raise AssertionError('Invalid ball-patch feature schema')
    return result.astype(np.float32)


def weak_eligibility(source, error, valid):
    ball, positions = source['ball'], source['positions']
    n = len(ball)
    incoming, outgoing = np.full(n, np.nan), np.full(n, np.nan)
    incoming[5:] = np.linalg.norm(ball[5:]-ball[:-5], axis=1)/.2
    outgoing[:-5] = np.linalg.norm(ball[5:]-ball[:-5], axis=1)/.2
    distance = np.linalg.norm(positions-ball[:, None], axis=2)
    nearest = np.min(np.where(np.isfinite(distance), distance, np.inf), axis=1)
    return (np.isfinite(ball).all(axis=1) & (incoming >= 1) & (incoming <= 35) &
            (outgoing >= 1) & (outgoing <= 35) & (nearest >= 3) &
            (ball[:, 0] >= 5) & (ball[:, 0] <= PITCH[0]-5) &
            (ball[:, 1] >= 3) & (ball[:, 1] <= PITCH[1]-3) & valid & (error <= 1))


def weak_split(eligible, blocks, reserved, seed=42):
    eligible = np.asarray(eligible, bool).copy()
    inside = np.zeros(len(eligible), bool)
    for _, start, end in blocks:
        inside[max(0, start):min(len(inside), end+1)] = True
    eligible &= inside & (np.arange(len(inside)) % 25 == 0)
    eligible[max(0, 11185-250):min(len(eligible), 18685+250)] = False
    for window in reserved:
        eligible[max(0, window['start']-250):min(len(eligible), window['end']+251)] = False
    candidates = np.flatnonzero(eligible)
    if len(candidates) < 220:
        raise ValueError('Insufficient excluded weak ball-frame candidates')
    rng = np.random.default_rng(seed)
    validation = np.sort(rng.choice(candidates, 60, replace=False))
    available = candidates[np.min(np.abs(candidates[:, None]-validation[None]), axis=1) > 250]
    if len(available) < 160:
        raise ValueError('Insufficient ball training frames after validation exclusion')
    train = np.sort(rng.choice(available, 160, replace=False))
    return train, validation, len(candidates), len(available)


def negative_positions(ball, positions):
    points = [ball+offset for offset in np.array([[4., 0], [-4, 0], [0, 4], [0, -4]])]
    valid = np.isfinite(positions).all(axis=1)
    xy = np.unique(positions[valid], axis=0)
    distance = np.linalg.norm(xy-ball, axis=1)
    order = np.lexsort((xy[:, 1], xy[:, 0], distance))
    points.extend(xy[order][distance[order] >= 4][:2])
    return [point for point in points if (point >= 0).all() and (point <= PITCH).all()]


def frame_features(image, matrix, ball, positions, profiles, training):
    image = Image.fromarray(image)
    result = np.full((len(profiles), 11, FEATURES), np.nan, np.float32)
    target = np.full(11, -1, dtype=int)
    geometry = image_geometry(matrix, ball)
    if geometry is None:
        return result, target
    x, y, diameter = geometry
    jitter = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] if training else [(0, 0)]
    crops = [(x+dx, y+dy, diameter, 1) for dx, dy in jitter]
    for point in negative_positions(ball, positions):
        geometry = image_geometry(matrix, point)
        if geometry is not None:
            crops.append((*geometry, 0))
    for index, (x, y, diameter, label) in enumerate(crops):
        target[index] = label
        for profile_index, profile in enumerate(profiles):
            value = patch_features(image, x, y, diameter, profile)
            if value is not None:
                result[profile_index, index] = value
    return result, target
