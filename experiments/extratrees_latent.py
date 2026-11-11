"""Matched ExtraTrees heads with train-only imputation and live-owner overlay."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

from motion_dead_trim import decode_motion_dead_trim
from player_representation import (
    ROOT, PROTOCOLS, check_inputs as check_representation_inputs, digest, live_overlay,
)
from rich_features import swap_teams

ROUND = ROOT / "data/control_extratrees_latent_v1"
PLAN = ROOT / "experiments/EXTRATREES_LATENT_PLAN.json"
VIEWS = ("control_extra", "pretrained_extra")
HISTORICAL_BEST_BUFFERED = .7542857142857143
PARAMS = dict(n_estimators=200, max_depth=8, min_samples_leaf=20,
              max_features=.8, class_weight="balanced", random_state=42, n_jobs=2)


def training_record(labels, train, validation):
    labels, train, validation = np.asarray(labels), np.asarray(train), np.asarray(validation)
    if len(np.intersect1d(train, validation)) or not np.isin(labels[train], ["H", "A", "D"]).all():
        raise ValueError("Invalid public training split")
    target = np.asarray([{"H": 0, "A": 1, "D": 2}[label] for label in labels[train]], dtype=np.int64)
    target = np.r_[target, np.where(target == 0, 1, np.where(target == 1, 0, 2))]
    if not np.array_equal(np.unique(target), [0, 1, 2]):
        raise ValueError("Training needs H/A/D classes after augmentation")
    # Audit the exact effective weights sklearn's balanced class_weight uses;
    # they are applied internally, never supplied a second time to fit().
    weight = len(target) / (3 * np.bincount(target)[target])
    return dict(train=train, validation=validation, target=target, weight=weight)


def fit_extra(inputs, labels, train, validation, seed):
    record = training_record(labels, train, validation)
    augmented = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    values = imputer.fit_transform(augmented)
    model = ExtraTreesClassifier(**dict(PARAMS, random_state=seed))
    model.fit(values, record["target"])
    # Forest probability reduction can sum trees in a scheduling-dependent
    # order. Serial prediction fixes that order without changing parallel fit.
    model.set_params(n_jobs=1)
    return imputer, model


def predict_extra(bundle, inputs):
    imputer, model = bundle
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("Unexpected H/A/D class order")
    model.set_params(n_jobs=1)
    return (model.predict_proba(imputer.transform(inputs)) +
            model.predict_proba(imputer.transform(swap_teams(inputs)))[:, [1, 0, 2]]) / 2.


def decode_extra(arrays, probability, supported, broadcast, motion, raw, blocks):
    reference = decode_motion_dead_trim(*arrays, broadcast, motion, raw, blocks)
    replacement = arrays[0].copy()
    for start, end in blocks:
        keep = supported[start:end]
        replacement[start:end][keep] = probability[start:end][keep]
    alternate = decode_motion_dead_trim(replacement, *arrays[1:], broadcast, motion, raw, blocks)
    result = live_overlay(reference, alternate)
    np.testing.assert_array_equal(result == "D", reference == "D")
    np.testing.assert_array_equal(result == "X", reference == "X")
    return result


def check_inputs():
    check_representation_inputs()
    plan = json.loads(PLAN.read_text())
    freeze = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (freeze / "plan.json").read_bytes():
        raise ValueError("ExtraTrees ownership plan changed")
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
