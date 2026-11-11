"""Infer stoppages in missing-ball intervals from sustained player motion."""

import numpy as np

from rule_based_decoder import TrackingEvidence, box_sum, decode_evidence, runs, velocity


SLOW_THRESHOLDS = {'slow_1_0': 1., 'slow_1_5': 1.5, 'slow_2_0': 2.}
VARIANTS = ('baseline', 'restart_reseed', *SLOW_THRESHOLDS)


def player_speed(positions, fps):
    speed = np.linalg.norm(velocity(positions, max(1, round(.24 * fps)), fps), axis=2)
    return np.where(speed <= 12., speed, np.nan)


def motion_stoppages(speed, teams, ball, fps, threshold):
    """Mark slow periods only inside ball gaps and with observations of both teams."""
    teams = np.asarray(teams)
    radius = round(.24 * fps)
    slow = np.ones(len(ball), dtype=bool)
    for team in (0, 1):
        tracks = speed[:, teams == team]
        present = np.isfinite(tracks).sum(axis=1)
        supported = present >= 3
        slow_frame = supported & ((tracks <= threshold).sum(axis=1) >= (2 / 3) * present)
        slow &= box_sum(slow_frame, radius, radius) >= .8 * (2 * radius + 1)
    missing = ~np.isfinite(ball).all(axis=1)
    dead = np.zeros(len(ball), dtype=bool)
    for start, end in runs(slow & missing):
        if end - start >= round(1.2 * fps):
            dead[start:end] = True
    return dead


def reseed(reference, evidence):
    """Decode each live interval using only ownership evidence from that interval."""
    labels = reference.copy()
    for start, end in runs(reference != 'D'):
        segment = slice(start, end)
        ignored = np.full(end - start, np.nan)
        local = TrackingEvidence(evidence.distance[segment], evidence.relative_speed[segment],
                                 ignored, ignored, evidence.fps)
        labels[segment] = decode_evidence(local)
    return labels


def adjust(reference, evidence, ball, teams, speed, variant):
    if variant == 'baseline':
        return reference.copy()
    if variant == 'restart_reseed':
        return reseed(reference, evidence)
    dead = motion_stoppages(speed, teams, ball, evidence.fps, SLOW_THRESHOLDS[variant])
    return np.where(dead, 'D', reference)
