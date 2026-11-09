"""Live-only CatBoost ownership, with the selected dead-ball probability retained."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from motion_dead_trim import decode_motion_dead_trim
from player_representation import (
    ROOT, PROTOCOLS, check_inputs as check_representation_inputs, digest, live_overlay,
)
from rich_features import swap_teams
from temporal_candidate import CAT_PARAMS

ROUND = ROOT / "data/control_factorized_latent_v1"
PLAN = ROOT / "experiments/FACTORIZED_LATENT_PLAN.json"
VIEWS = ("control_owner", "pretrained_owner")
HISTORICAL_BEST_BUFFERED = .7542857142857143


def training_record(labels, train, validation):
    """Use only permitted H/A rows; augmentation gives equal home/away mass."""
    labels, train, validation = np.asarray(labels), np.asarray(train), np.asarray(validation)
    if len(np.intersect1d(train, validation)) or not np.isin(labels[train], ["H", "A", "D"]).all():
        raise ValueError("Invalid public training split")
    live = train[np.isin(labels[train], ["H", "A"])]
    if not len(live):
        raise ValueError("No live ownership training rows")
    target = (labels[live] == "A").astype(np.int64)
    target = np.r_[target, 1 - target]
    return dict(split_train=train, train=live, validation=validation,
                target=target, weight=np.ones(len(target), dtype=np.float64))


def fit_owner(inputs, labels, train, validation, seed):
    record = training_record(labels, train, validation)
    values = inputs.iloc[record["train"]]
    augmented = pd.concat([values, swap_teams(values)], ignore_index=True)
    model = CatBoostClassifier(**dict(CAT_PARAMS, loss_function="Logloss", random_seed=seed))
    model.fit(augmented, record["target"], sample_weight=record["weight"])
    return model


def predict_owner(model, inputs):
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Unexpected binary owner class order")
    return (model.predict_proba(inputs) +
            model.predict_proba(swap_teams(inputs))[:, [1, 0]]) / 2.


def compose_owner(original, owner):
    """Factorized P(owner,live)=P(owner|live)*P(live), with exact original D."""
    original, owner = np.asarray(original, float), np.asarray(owner, float)
    if original.ndim != 2 or original.shape[1] != 3 or owner.shape != (len(original), 2):
        raise ValueError("Expected aligned three-state and binary probabilities")
    for probability in (original, owner):
        if (not np.isfinite(probability).all() or (probability < 0).any()
                or not np.allclose(probability.sum(axis=1), 1., rtol=1e-6, atol=1e-7)):
            raise ValueError("Invalid state probabilities")
    result = original.copy()
    result[:, :2] = owner * original[:, :2].sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(result[:, 2], original[:, 2])
    return result


def decode_owner(arrays, owner, supported, broadcast, motion, raw, blocks):
    reference = decode_motion_dead_trim(*arrays, broadcast, motion, raw, blocks)
    probability = arrays[0].copy()
    for start, end in blocks:
        value = compose_owner(arrays[0][start:end], owner[start:end])
        keep = supported[start:end]
        probability[start:end][keep] = value[keep]
    alternate = decode_motion_dead_trim(probability, *arrays[1:], broadcast, motion, raw, blocks)
    result = live_overlay(reference, alternate)
    np.testing.assert_array_equal(result == "D", reference == "D")
    np.testing.assert_array_equal(result == "X", reference == "X")
    return result


def check_inputs():
    check_representation_inputs()
    plan = json.loads(PLAN.read_text())
    freeze = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (freeze / "plan.json").read_bytes():
        raise ValueError("Factorized ownership plan changed")
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
        raise ValueError("Earlier ledger bytes changed")
    if (freeze / "symlinks.json").exists():
        for name, target in json.loads((freeze / "symlinks.json").read_text()).items():
            path = ROOT / name
            if not path.is_symlink() or str(path.readlink()) != target:
                raise ValueError("Source symlink changed: " + name)
    return plan
