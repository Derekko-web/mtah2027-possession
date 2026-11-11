"""Fixed equal-probability pool of three independently seeded latent heads."""

import json
from pathlib import Path

import numpy as np

from player_representation import (
    ROOT, PROTOCOLS, check_inputs as check_representation_inputs, digest,
)

ROUND = ROOT / "data/control_latent_seed_ensemble_v1"
PLAN = ROOT / "experiments/LATENT_SEED_ENSEMBLE_PLAN.json"
HISTORICAL_BEST_BUFFERED = .7542857142857143
CLASS_ORDER = ("H", "A", "D")


def member_seeds(main_seed):
    if main_seed not in (42, 43, 44):
        raise ValueError("Expected a fixed reference seed42,43,44")
    return (main_seed, main_seed + 100, main_seed + 200)


def pool_probabilities(members, class_order=CLASS_ORDER):
    """Exactly three H/A/D members, accumulated in their prescribed seed order."""
    if tuple(class_order) != CLASS_ORDER:
        raise ValueError("Expected H/A/D class order")
    if len(members) != 3:
        raise ValueError("Exactly three fixed members are required")
    result = None
    for member in members:
        probability = np.asarray(member, dtype=np.float64)
        if (probability.ndim != 2 or probability.shape[1] != 3
                or not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(1), 1., rtol=1e-6, atol=1e-7)):
            raise ValueError("Expected finite normalized H/A/D probabilities")
        if result is None:
            result = probability.copy()
        elif probability.shape != result.shape:
            raise ValueError("Member probability shapes differ")
        else:
            result += probability
    return result / 3.


def check_inputs():
    """Verify the new prospective freeze and unchanged R37 provenance."""
    check_representation_inputs()
    plan = json.loads(PLAN.read_text())
    freeze = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (freeze / "plan.json").read_bytes():
        raise ValueError("Latent seed ensemble plan changed")
    for group in (plan["input_sha256"],
                  json.loads((freeze / "implementation.json").read_text()),
                  json.loads((freeze / "protected_artifacts.json").read_text())):
        for name, expected in group.items():
            if Path(name).is_absolute() or digest(ROOT / name) != expected:
                raise ValueError("Frozen dependency changed: " + name)
    if not (ROOT / "experiments/EXPERIMENT_LEDGER.csv").read_bytes().startswith(
            (freeze / "EXPERIMENT_LEDGER.csv").read_bytes()):
        raise ValueError("Earlier ledger bytes changed")
    for name, target in json.loads((freeze / "symlinks.json").read_text()).items():
        path = ROOT / name
        if not path.is_symlink() or str(path.readlink()) != target:
            raise ValueError("Source symlink changed: " + name)
    return plan
