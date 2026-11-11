"""Nested, training-only ownership stacking of two frozen expert interfaces."""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from player_representation import ROOT, PROTOCOLS, check_inputs as check_upstream, digest
from factorized_latent import decode_owner

ROUND = ROOT / "data/control_nested_owner_stack_v1"
PLAN = ROOT / "experiments/NESTED_OWNER_STACK_PLAN.json"
PUBLIC = np.arange(11185, 18685)
CHUNKS = ((11185, 13685), (13685, 16185), (16185, 18685))
HISTORICAL_BEST_BUFFERED = .7542857142857143


def inner_splits(outer_train, outer_validation):
    """Three fixed chunks, each with a 250-frame inner training exclusion."""
    outer_train, outer_validation = np.asarray(outer_train), np.asarray(outer_validation)
    for values in (outer_train, outer_validation):
        if (values.ndim != 1 or values.dtype.kind not in "iu"
                or not np.isin(values, PUBLIC).all()
                or len(np.unique(values)) != len(values)):
            raise ValueError("Expected unique public frame indices")
    if np.intersect1d(outer_train, outer_validation).size:
        raise ValueError("Outer training overlaps validation")
    result = []
    for number, (start, end) in enumerate(CHUNKS, 1):
        validation = outer_train[(outer_train >= start) & (outer_train < end)]
        train = outer_train[(outer_train < start - 250) | (outer_train >= end + 250)]
        result.append((number, train, validation))
    covered = np.concatenate([validation for _, _, validation in result])
    np.testing.assert_array_equal(np.sort(covered), np.sort(outer_train))
    return result


def stack_inputs(original, representation):
    """Two H/A log odds; no scaling, frame identifiers, or target features."""
    columns = []
    shape = None
    for probability in (original, representation):
        probability = np.asarray(probability, dtype=np.float64)
        if (probability.ndim != 2 or probability.shape[1] != 3
                or not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(1), 1., rtol=1e-6, atol=1e-7)):
            raise ValueError("Expected finite normalized H/A/D probabilities")
        if shape is not None and probability.shape != shape:
            raise ValueError("Expert probabilities do not align")
        shape = probability.shape
        log = np.log(np.clip(probability[:, :2], 1e-6, 1.))
        columns.append(log[:, 0] - log[:, 1])
    return np.column_stack(columns)


def meta_record(labels, frames, original, representation):
    """Mirror live inner-OOF logits and targets; uniform augmented weights."""
    labels, frames = np.asarray(labels), np.asarray(frames)
    if (frames.ndim != 1 or frames.dtype.kind not in "iu"
            or not np.isin(frames, PUBLIC).all() or len(np.unique(frames)) != len(frames)
            or not np.isin(labels[frames], ["H", "A", "D"]).all()):
        raise ValueError("Invalid meta training frames")
    matrix = stack_inputs(original, representation)
    if len(matrix) != len(frames):
        raise ValueError("Meta inputs and frames do not align")
    keep = np.isin(labels[frames], ["H", "A"])
    if not keep.any():
        raise ValueError("No live meta training rows")
    values = matrix[keep]
    target = (labels[frames[keep]] == "H").astype(np.int64)
    return dict(train=frames[keep], inputs=np.vstack((values, -values)),
                target=np.r_[target, 1-target], weight=np.ones(2*len(values)))


def fit_meta(record):
    model = LogisticRegression(C=1., fit_intercept=False, solver="lbfgs", max_iter=2000)
    model.fit(record["inputs"], record["target"], sample_weight=record["weight"])
    return model


def predict_meta(model, original, representation):
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Unexpected meta classes")
    home = model.predict_proba(stack_inputs(original, representation))[:, 1]
    return np.column_stack((home, 1-home))


def check_inputs():
    check_upstream()
    freeze = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (freeze / "plan.json").read_bytes():
        raise ValueError("Nested owner plan changed")
    plan = json.loads(PLAN.read_text())
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
            raise ValueError("Source symlink changed")
    return plan
