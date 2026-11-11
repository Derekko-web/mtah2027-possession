"""Offline team possession from ball and player tracking in metres."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rules:
    contact_radius: float = 1.8
    control_duration: float = .12
    relative_speed_weight: float = .12
    stationary_speed: float = .65
    stationary_duration: float = 1.0


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def box_sum(values, before, after):
    offsets = np.arange(len(values))
    total = np.cumsum(np.concatenate((np.zeros_like(values[:1]), values), axis=0), axis=0)
    return total[np.minimum(len(values), offsets + after + 1)] - total[np.maximum(0, offsets - before)]


def smooth_ball(ball, radius):
    seen = np.isfinite(ball).all(axis=1)
    sums = box_sum(np.where(seen[:, None], ball, 0), radius, radius)
    count = box_sum(seen, radius, radius)
    return np.divide(sums, count[:, None], out=np.full_like(ball, np.nan),
                     where=seen[:, None])


def velocity(xy, radius, fps):
    value = np.full_like(xy, np.nan)
    value[radius:len(xy) - radius] = (xy[2 * radius:] - xy[:-2 * radius]) * (fps / (2 * radius))
    return value


@dataclass(frozen=True)
class TrackingEvidence:
    distance: np.ndarray
    relative_speed: np.ndarray
    ball_speed: np.ndarray
    border: np.ndarray
    fps: float


def tracking_evidence(positions, teams, ball, pitch, *, fps=25):
    """Use centred pitch coordinates and a stable player track within each half."""
    positions, teams = np.asarray(positions), np.asarray(teams)
    ball, pitch = np.asarray(ball, dtype=float), np.asarray(pitch, dtype=float)
    if (ball.ndim != 2 or ball.shape[1] != 2 or len(ball) == 0 or teams.ndim != 1
            or positions.shape != (len(ball), len(teams), 2) or set(teams) != {0, 1}
            or pitch.shape != (2,) or not (np.isfinite(pitch) & (pitch > 0)).all()
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Expected aligned half-match tracking, two teams and positive pitch dimensions and fps')

    centre = smooth_ball(ball, max(1, round(.08 * fps)))
    stationary_centre = smooth_ball(ball, max(1, round(.20 * fps)))
    short_radius = max(1, round(.12 * fps))
    ball_velocity = velocity(centre, short_radius, fps)
    ball_speed = np.linalg.norm(velocity(stationary_centre, max(1, round(.32 * fps)), fps), axis=1)
    distance = np.empty((len(ball), 2))
    relative_speed = np.empty_like(distance)
    rows = np.arange(len(ball))
    for k in (0, 1):
        players = positions[:, teams == k].astype(float)
        separation = np.linalg.norm(players - centre[:, None, :], axis=2)
        separation[~np.isfinite(separation)] = np.inf
        nearest = np.argmin(separation, axis=1)
        distance[:, k] = separation[rows, nearest]
        player_velocity = velocity(players, short_radius, fps)[rows, nearest]
        relative_speed[:, k] = np.linalg.norm(ball_velocity - player_velocity, axis=1)
    border = np.max(np.abs(centre) - pitch / 2, axis=1)
    return TrackingEvidence(distance, relative_speed, ball_speed, border, fps)


def stoppages(evidence, rules):
    # Require sustained evidence so a tracking spike cannot split a possession.
    # For stationarity, missing observations can connect corroborated samples.
    slow = evidence.ball_speed < rules.stationary_speed
    has_speed = np.isfinite(evidence.ball_speed)
    radius = max(1, round(.40 * evidence.fps))
    count = box_sum(has_speed, radius, radius)
    moving = evidence.ball_speed > 2 * rules.stationary_speed
    stationary = ((box_sum(slow, radius, radius) >= .8 * count)
                  & (count >= max(1, round(.25 * evidence.fps))) & ~moving)
    dead = np.zeros(len(slow), dtype=bool)
    for signal, seconds in ((evidence.border > .25, .24), (stationary, rules.stationary_duration)):
        for start, end in runs(signal):
            if end - start >= max(1, round(seconds * evidence.fps)):
                dead[start:end] = True

    # Join two confirmed dead intervals through a short gap with no indication
    # of live motion inside the pitch. Ball disappearance alone adds no stop.
    for start, end in runs(~dead):
        if 0 < start and end < len(dead) and end - start <= round(5 * evidence.fps):
            moving_inside = ((evidence.ball_speed[start:end] > rules.stationary_speed)
                             & (evidence.border[start:end] <= .25))
            if not moving_inside.any():
                dead[start:end] = True
    return dead


def decode_evidence(evidence, rules=Rules()):
    """Resolve short control bouts and carry the last owner through uncertain play."""
    distance = evidence.distance
    relative = np.where(np.isfinite(evidence.relative_speed), evidence.relative_speed, 3.)
    cost = distance + rules.relative_speed_weight * np.minimum(relative, 12.)
    contact = ((distance <= rules.contact_radius) & (cost + .25 < cost[:, ::-1])
               & ((relative <= 5.) | (distance <= .9)))
    dead = stoppages(evidence, rules)
    labels = np.where(dead, 'D', 'X')
    length = max(1, round(.40 * evidence.fps))
    required = max(1, round(rules.control_duration * evidence.fps))
    for start, end in runs(~dead):
        local = contact[start:end]
        votes = box_sum(local, 0, length - 1)
        accepted = local & (votes >= required) & (votes > votes[:, ::-1])
        indices, teams = np.nonzero(accepted)
        if not len(indices):
            d = distance[start:end]
            observed = np.flatnonzero(d[:, 0] != d[:, 1])
            if not len(observed):
                continue
            indices = observed[:1]
            teams = d[indices].argmin(axis=1)
        owner = teams[np.maximum(0, np.searchsorted(indices, np.arange(end - start), side='right') - 1)]
        labels[start:end] = np.where(owner == 0, 'H', 'A')
    return labels


def decode(positions, teams, ball, pitch, *, fps=25, rules=Rules()):
    return decode_evidence(tracking_evidence(positions, teams, ball, pitch, fps=fps), rules)
