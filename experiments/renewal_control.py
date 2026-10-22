"""Keep the anchored reference while testing newly learned short proposals."""

import numpy as np

from boundary_refinement import runs
from consensus_control import consensus_team
from control_refinement import compose, smooth
from transition_penalty import decode


def anchored_emissions(current, previous, fast, support, evidence):
    """Reconstruct the frozen reference emissions; runners assert exact parity."""
    slow, old, short = smooth(current, 25), smooth(previous, 25), smooth(fast, 5)
    owner = consensus_team(slow, old, short, support)
    attenuation = 1-.75*evidence.max(axis=1)
    dead = slow[:, 2]*attenuation
    baseline = decode(compose(dead, owner))
    for start, end, team in runs(decode(old)):
        if team == 2 or not 8 <= end-start <= 125:
            continue
        if short[start:end, team].mean() < .65 or support[start:end, team].sum() < 3:
            continue
        anchor = baseline[max(0, start-13):min(len(baseline), end+13)]
        if not np.any(anchor == team) or not np.any(baseline[start:end] == 2):
            continue
        owner[start:end] = short[start:end]
        dead[start:end] = np.minimum(slow[start:end, 2], old[start:end, 2])*attenuation[start:end]
    return compose(dead, owner)


def short_proposals(emissions, probability, support, bracket):
    """Model-generated brief live islands, bounded by opposite predicted control."""
    states = decode(emissions)
    proposal_probability = smooth(probability, 5)
    result = emissions.copy()
    for start, end, team in runs(decode(proposal_probability)):
        if team == 2 or not 8 <= end-start <= 125 or start == 0 or end == len(states):
            continue
        if proposal_probability[start:end, team].mean() < .65 or np.any(states[start:end] == 2):
            continue
        if support[start:end, team].sum() < 3 and bracket[start:end, team].sum() < 3:
            continue
        before, after = states[max(0, start-13):start], states[end:min(len(states), end+13)]
        if np.mean(before == 1-team) <= .5 or np.mean(after == 1-team) <= .5:
            continue
        owner = proposal_probability[start:end, :2]
        owner = owner/owner.sum(axis=1, keepdims=True)
        result[start:end] = compose(emissions[start:end, 2], owner)
    return result
