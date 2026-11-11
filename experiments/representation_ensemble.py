"""Fixed, equal-weight probability pooling for the external representation head.

The original and representation heads are pooled before the unchanged selected
decoder. Only H/A decisions may differ from the selected reference.
"""

import json
from pathlib import Path

import numpy as np

from motion_dead_trim import decode_motion_dead_trim
from player_representation import (
    ROOT, PROTOCOLS, check_inputs as check_representation_inputs, digest,
    live_overlay,
)

ROUND = ROOT / "data/control_representation_ensemble_v1"
PLAN = ROOT / "experiments/REPRESENTATION_ENSEMBLE_PLAN.json"
HISTORICAL_BEST_BUFFERED = .7542857142857143


def equal_probability_pool(original, representation):
    """Arithmetic mean of two aligned H/A/D distributions; no fitted weight."""
    original = np.asarray(original, dtype=np.float64)
    representation = np.asarray(representation, dtype=np.float64)
    if original.shape != representation.shape or original.ndim != 2 or original.shape[1] != 3:
        raise ValueError("Expected matching H/A/D probability arrays")
    for probability in (original, representation):
        if (not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(axis=1), 1., rtol=1e-6, atol=1e-7)):
            raise ValueError("Invalid probability distribution")
    return (original + representation) / 2.


def decode_ensemble(arrays, representation, broadcast, motion_info, raw, blocks):
    """Use fixed pooled slot0 inside each decode block, retaining reference D/X."""
    reference = decode_motion_dead_trim(*arrays, broadcast, motion_info, raw, blocks)
    pooled = arrays[0].copy()
    for start, end in blocks:
        pooled[start:end] = equal_probability_pool(
            arrays[0][start:end], representation[start:end])
    alternate = [pooled, *arrays[1:]]
    prediction = live_overlay(reference, decode_motion_dead_trim(
        *alternate, broadcast, motion_info, raw, blocks))
    np.testing.assert_array_equal(prediction == "D", reference == "D")
    np.testing.assert_array_equal(prediction == "X", reference == "X")
    return prediction


def check_inputs():
    """Verify the prospective protocol and all retained upstream provenance."""
    check_representation_inputs()
    plan = json.loads(PLAN.read_text())
    freeze = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (freeze / "plan.json").read_bytes():
        raise ValueError("Representation ensemble plan changed")
    if tuple(plan.get("protocols", PROTOCOLS)) != PROTOCOLS:
        raise ValueError("Unexpected validation protocols")
    for identities in (
        plan["input_sha256"],
        json.loads((freeze / "implementation.json").read_text()),
        json.loads((freeze / "protected_artifacts.json").read_text()),
    ):
        for name, expected in identities.items():
            if Path(name).is_absolute() or digest(ROOT / name) != expected:
                raise ValueError("Frozen dependency changed: " + name)
    ledger = ROOT / "experiments/EXPERIMENT_LEDGER.csv"
    if not ledger.read_bytes().startswith((freeze / ledger.name).read_bytes()):
        raise ValueError("Earlier experiment ledger bytes changed")
    symlinks = freeze / "symlinks.json"
    if symlinks.exists():
        for name, target in json.loads(symlinks.read_text()).items():
            path = ROOT / name
            if not path.is_symlink() or str(path.readlink()) != target:
                raise ValueError("Source symlink changed: " + name)
    return plan
