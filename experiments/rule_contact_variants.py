"""Alternative contact rules for the frozen possession decoder."""

from dataclasses import replace

import numpy as np

from rule_based_decoder import Rules, decode_evidence, tracking_evidence


REFERENCE = Rules()
VARIANTS = {
    'baseline': REFERENCE,
    'quicker_confirmation': replace(REFERENCE, control_duration=.08),
    'longer_confirmation': replace(REFERENCE, control_duration=.20),
    'tight_contact': replace(REFERENCE, contact_radius=1.4),
    'wide_contact': replace(REFERENCE, contact_radius=2.4),
    'wide_sustained_contact': replace(REFERENCE, contact_radius=2.4, control_duration=.20),
    'reject_ball_spikes': REFERENCE,
}
MAX_BALL_SPEED = 40.


def reject_ball_spikes(ball, fps):
    """Drop an isolated point when both steps are implausible and its neighbors agree."""
    clean = ball.copy()
    limit = MAX_BALL_SPEED / fps
    steps = np.linalg.norm(np.diff(ball, axis=0), axis=1)
    neighbor_step = np.linalg.norm(ball[2:] - ball[:-2], axis=1)
    spikes = (steps[:-1] > limit) & (steps[1:] > limit) & (neighbor_step <= 2 * limit)
    clean[1:-1][spikes] = np.nan
    return clean


def decode_contacts(positions, teams, ball, pitch, evidence, variant):
    if variant == 'reject_ball_spikes':
        cleaned = reject_ball_spikes(ball, evidence.fps)
        updated = tracking_evidence(positions, teams, cleaned, pitch, fps=evidence.fps)
        evidence = replace(evidence, distance=updated.distance, relative_speed=updated.relative_speed)
    return decode_evidence(evidence, VARIANTS[variant])
