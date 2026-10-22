"""Shared feature, fitting and decoding path for the expanded round finalists.

All feature construction is label-free. Fit targets and class weights read only
explicit training indices. Geometry in metres, two-sided offline inputs <=5 s,
and the original gap prior's <=10 s context fit the 250-frame exclusion buffers.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from ball_kinematics import build_kinematics
from build_candidate import PARAMS as XGB_PARAMS, objective_weights
from evaluate_physics_context import apply_context, gap_context
from relative_motion import build_relative_motion
from rich_features import build_features, swap_teams
from temporal_control import contact_memory, temporal_taps
from train_factorized_baseline import compact_columns
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing

CAT_PARAMS = dict(loss_function="MultiClass", depth=4, iterations=300,
                  learning_rate=.04, l2_leaf_reg=10, random_seed=42,
                  thread_count=2, allow_writing_files=False, verbose=False,
                  nan_mode="Min", boosting_type="Ordered")
METHODS = ("previous_ensemble", "ordered_catboost_frame", "ordered_xgboost_frame")


def ordered_inputs(base):
    """102 compact observations, omitting absolute distance to pitch boundaries."""
    return base.join(contact_memory(base)).join(temporal_taps(base)).drop(
        columns=["ball_goal_line_distance", "ball_touchline_distance"])


def build_inputs(n_frames):
    raw = build_features(n_frames)
    base = raw[compact_columns(raw)].join(build_kinematics(n_frames)).join(
        build_relative_motion(n_frames))
    return raw, base, ordered_inputs(base)


def fit_models(method, inputs, labels, train, seed=42):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    train = np.asarray(train)
    if not len(train) or not set(labels[train]) == set(CLASSES):
        raise ValueError("Training indices must contain only public H/A/D and all three classes")
    target = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
    augmented = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
    models = []
    for segment_balance in ((False, True) if method == "previous_ensemble" else (False,)):
        y, weight = objective_weights(target, train, segment_balance)
        if method == "ordered_catboost_frame":
            model = CatBoostClassifier(**dict(CAT_PARAMS, random_seed=seed))
        else:
            params = dict(XGB_PARAMS, random_state=seed)
            if method == "ordered_xgboost_frame":
                params.update(max_depth=3, n_estimators=160)
            model = XGBClassifier(**params)
        model.fit(augmented, y, sample_weight=weight)
        models.append(model)
    return models


def predict_models(models, inputs):
    result = np.zeros((len(inputs), 3), dtype=float)
    for model in models:
        # Every model is trained on integer class codes H=0, A=1, D=2.
        if not np.array_equal(model.classes_, np.arange(3)):
            raise ValueError("Unexpected model class order")
        result += (model.predict_proba(inputs) +
                   model.predict_proba(swap_teams(inputs))[:, [1, 0, 2]]) / (2*len(models))
    return result


def decode_blocks(probability, side, blocks):
    """Decode disjoint (start, exclusive end) blocks; abstain everywhere else."""
    probability = np.asarray(probability)
    if probability.ndim != 2 or probability.shape[1] != 3 or len(side) != len(probability):
        raise ValueError("Expected aligned H/A/D probabilities and gap context")
    prediction = np.full(len(probability), "X", dtype="<U1")
    for start, end in blocks:
        if not 0 <= start < end <= len(probability) or np.any(prediction[start:end] != "X"):
            raise ValueError("Invalid or overlapping decode blocks")
        block = probability[start:end]
        if not np.isfinite(block).all() or (block < 0).any() or not np.allclose(block.sum(axis=1), 1):
            raise ValueError("Incomplete or invalid block probabilities")
        adjusted = apply_context(block, side[start:end], np.zeros(end-start), False)
        prediction[start:end] = CLASSES[decode(probability_smoothing(adjusted))]
    return prediction


def gap_sides(raw):
    return gap_context(raw)
