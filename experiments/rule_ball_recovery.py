"""Recover missing ball observations without changing stoppage decisions."""

from dataclasses import dataclass, replace

import numpy as np

from rule_based_decoder import box_sum, decode_evidence, runs, tracking_evidence


@dataclass(frozen=True)
class RecoveryRules:
    gap_seconds: float = .24
    gap_max_speed: float = 28.
    agreement_metres: float = 2.
    agreement_seconds: float = 2.
    agreement_samples: int = 3
    donor_max_step_speed: float = 45.


VARIANTS = {
    'baseline': (False, False),
    'interpolate': (True, False),
    'broadcast': (False, True),
    'combined': (True, True),
}


def inside(ball, pitch):
    return (np.abs(ball) <= np.asarray(pitch) / 2).all(axis=1)


def recover(ball, donor, pitch, live, *, fps, variant, rules=RecoveryRules()):
    interpolate, use_donor = VARIANTS[variant]
    observed = np.isfinite(ball).all(axis=1)
    fused = ball.copy()
    counts = dict(broadcast=0, interpolate=0)
    if use_donor:
        # Agreement is measured on overlapping observations from the same live run.
        # An unsupported camera cut or conflicting overlap cannot establish a track.
        for start, end in runs(live):
            tactical, other = ball[start:end], donor[start:end]
            valid = inside(other, pitch)
            jointly_seen = observed[start:end] & valid
            agrees = jointly_seen & (np.linalg.norm(tactical - other, axis=1) <= rules.agreement_metres)
            jump = np.linalg.norm(np.diff(other, axis=0), axis=1) > rules.donor_max_step_speed / fps
            trusted = valid & ~np.r_[False, jump] & ~np.r_[jump, False]
            radius = round(rules.agreement_seconds * fps)
            for first, last in runs(trusted):
                overlap = jointly_seen[first:last]
                support = box_sum(agrees[first:last], radius, radius)
                usable = ((support >= rules.agreement_samples)
                          & (support >= .75 * box_sum(overlap, radius, radius))
                          & ~observed[start + first:start + last])
                fused[start + first:start + last][usable] = other[first:last][usable]
                counts['broadcast'] += int(usable.sum())

    if interpolate:
        # Use only original tactical endpoints. Donor points cannot extend a fill.
        eligible_endpoints = observed & live & inside(ball, pitch)
        for start, end in runs(~observed & live):
            if (end - start > round(rules.gap_seconds * fps) or start == 0 or end == len(ball)
                    or not eligible_endpoints[start - 1] or not eligible_endpoints[end]):
                continue
            delta = ball[end] - ball[start - 1]
            if np.linalg.norm(delta) * fps > rules.gap_max_speed * (end - start + 1):
                continue
            missing = ~np.isfinite(fused[start:end]).all(axis=1)
            fractions = np.arange(1, end - start + 1) / (end - start + 1)
            fused[start:end][missing] = ball[start - 1] + fractions[missing, None] * delta
            counts['interpolate'] += int(missing.sum())
    return fused, counts


def recover_possession(positions, teams, ball, donor, pitch, evidence, reference, *, variant,
                       rules=RecoveryRules()):
    live = reference != 'D'
    fused, counts = recover(ball, donor, pitch, live, fps=evidence.fps, variant=variant, rules=rules)
    if not any(counts.values()):
        return reference.copy(), counts
    updated = tracking_evidence(positions, teams, fused, pitch, fps=evidence.fps)
    # Keep the original stoppage inputs and the reference's Smart Tagging overrides.
    ownership = decode_evidence(replace(evidence, distance=updated.distance,
                                        relative_speed=updated.relative_speed))
    return np.where(live & np.isin(ownership, ('H', 'A')), ownership, reference), counts
