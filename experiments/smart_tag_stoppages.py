"""Extend dead-ball intervals with pitch exits and automatic restart windows."""

from dataclasses import dataclass

import numpy as np

from rule_based_decoder import runs, smooth_ball


# Compatible exit axis, maximum pairing interval and maximum tag-only lookback,
# all in seconds where applicable. The tag window bounds the restart search.
RESTARTS = {
    'throw-in': (1, 30, 8),
    'corner kick': (0, 50, 15),
    'goal kick': (0, 50, 15),
    'free kick': (None, 0, 15),
    'kick off (after goal)': (0, 90, 30),
    'kick off (start)': (None, 0, 10),
}


@dataclass(frozen=True)
class StoppageRules:
    line_margin_m: float = .25
    crossing_confirmation_s: float = .12
    launch_lookahead_s: float = .40
    launch_travel_m: float = 1.0
    untagged_exit_grace_s: float = 6.0


def restart_windows(events, start, end):
    windows = []
    for event in events:
        kind = (event['tags'].get('Type') or '').strip().lower()
        left, right = max(start, event['start_frame']), min(end, event['end_frame'])
        if (event['code'] or '').strip().lower() == 'set pieces' and kind in RESTARTS and left < right:
            windows.append((left - start, right - start, kind))
    return sorted(windows)


def restart_location(ball, boundary, kind):
    if kind == 'throw-in':
        return np.abs(boundary[:, 1]) <= 2
    if kind == 'corner kick':
        return (np.abs(boundary) <= 2).all(axis=1)
    if kind == 'goal kick':
        return ((boundary[:, 0] >= -6.5) & (boundary[:, 0] <= .25)
                & (np.abs(ball[:, 1]) <= 10))
    if kind.startswith('kick off'):
        return np.linalg.norm(ball, axis=1) <= 2
    return (boundary <= .25).all(axis=1)


def refine(reference, ball, pitch, windows, *, fps=25, rules=StoppageRules()):
    """Add D labels from confirmed exits and restart tags, scoped to one half."""
    reference = np.asarray(reference)
    ball = np.asarray(ball, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    n = len(reference)
    if (reference.shape != (n,) or ball.shape != (n, 2) or pitch.shape != (2,)
            or not np.isin(reference, ['H', 'A', 'D', 'X']).all() or fps <= 0):
        raise ValueError('Expected aligned half-match labels, ball and pitch dimensions')
    if not n:
        return reference.copy(), []

    centre = smooth_ball(ball, max(1, round(.08 * fps)))
    boundary = np.abs(centre) - pitch / 2
    border = boundary.max(axis=1)
    lag = min(n, max(1, round(rules.launch_lookahead_s * fps)))
    displacement = np.full(n, np.nan)
    displacement[:-lag] = np.linalg.norm(centre[lag:] - centre[:-lag], axis=1)
    previous_displacement = np.full(n, np.nan)
    previous_displacement[lag:] = displacement[:-lag]
    future_border = np.full(n, np.nan)
    future_border[:-lag] = border[lag:]
    moving_inside = (displacement >= rules.launch_travel_m) & (future_border <= rules.line_margin_m)
    idle_before = ~np.isfinite(previous_displacement) | (previous_displacement <= .5)
    inward = np.zeros(n, dtype=bool)
    inward[:-lag] = boundary[:-lag, 1] - boundary[lag:, 1] >= .5

    # Require continuity on one side of one line to reject isolated tracking spikes.
    confirmation = max(1, round(rules.crossing_confirmation_s * fps))
    exits = sorted((left, right, axis) for axis in (0, 1) for sign in (-1, 1)
                   for left, right in runs(sign * ball[:, axis] > pitch[axis] / 2 + rules.line_margin_m)
                   if right - left >= confirmation)
    settled_play = np.zeros(n, dtype=bool)
    for left, right in runs((border < -2) & moving_inside):
        if right - left >= max(1, round(.60 * fps)):
            settled_play[left:right] = True

    dead = reference == 'D'
    events = []
    restarts = []
    last_restart = 0
    for start, end, kind in windows:
        if not 0 <= start < end <= n or kind not in RESTARTS:
            raise ValueError('Invalid half-local restart window')
        middle = (start + end) // 2
        local = slice(start, end)
        support = idle_before[local] | (inward[local] if kind == 'throw-in' else False)
        launch = moving_inside[local] & support & restart_location(centre[local], boundary[local], kind)
        choices = start + np.flatnonzero(launch)
        restart = int(choices[np.argmin(np.abs(choices - middle))]) if len(choices) else middle
        if restart <= last_restart:
            continue

        axis, max_pair, max_tag_only = RESTARTS[kind]
        left = max(last_restart, restart - round(max(max_pair, max_tag_only) * fps))
        live = np.flatnonzero(settled_play[left:max(left, start)])
        if len(live):
            left += int(live[-1]) + 1
        compatible = [max(a, left) for a, b, direction in exits
                      if direction == axis and a < restart and b > left and a >= restart - round(max_pair * fps)]
        if compatible:
            begin, reason = min(compatible), 'exit_and_tag'
        else:
            begin, reason = max(left, restart - round(max_tag_only * fps)), 'restart_tag'
            # A tag without a plausible launch or exit cannot override observed play.
            if not len(choices) and settled_play[max(start, restart - lag):restart].any():
                continue
        if begin < restart:
            dead[begin:restart] = True
            events.append(dict(start=begin, end_exclusive=restart, kind=reason,
                               restart_type=kind, motion_confirmed=bool(len(choices))))
        last_restart = restart
        restarts.append(restart)

    # A track can disappear right after the ball crosses a line. Hold D until
    # an observed return, a tagged restart, or the bounded missing-ball grace.
    returns = np.array([left for left, right in runs((border < -rules.line_margin_m) & moving_inside)
                        if right - left >= confirmation], dtype=int)
    limits = np.sort(np.concatenate((returns, np.asarray(restarts, dtype=int), [n])))
    for start, end, _ in exits:
        next_play = limits[np.searchsorted(limits, start, side='right')]
        stop = min(n, end + round(rules.untagged_exit_grace_s * fps), int(next_play))
        if dead[start:stop].all():
            continue
        dead[start:stop] = True
        events.append(dict(start=start, end_exclusive=stop, kind='pitch_exit'))

    return np.where(dead, 'D', reference), events
