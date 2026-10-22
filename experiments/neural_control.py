"""Small neural ownership head with strictly training-only preprocessing."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from rich_features import swap_teams
from xgboost_baseline import CLASSES


def fit_neural(inputs, labels, train, live_only, seed=42):
    train = np.asarray(train)
    if live_only:
        train = train[np.isin(labels[train], ["H", "A"])]
    expected = set(CLASSES[:2] if live_only else CLASSES)
    if set(labels[train]) != expected:
        raise ValueError("Unexpected eligible neural training targets")
    target = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
    swapped = np.where(target == 0, 1, np.where(target == 1, 0, 2))
    y = np.r_[target, swapped]
    x = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
    weight = len(y)/(len(expected)*np.bincount(y)[y])
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("network", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1.,
                                  max_iter=150, batch_size=128, learning_rate_init=.001, early_stopping=False,
                                  tol=0, n_iter_no_change=151, random_state=seed)),
    ])
    with threadpool_limits(limits=2):
        model.fit(x, y, network__sample_weight=weight)
    return model


def predict_neural(model, inputs):
    classes = model.named_steps["network"].classes_
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Unexpected neural class ordering")
    order = [1, 0] if len(classes) == 2 else [1, 0, 2]
    with threadpool_limits(limits=2):
        return (model.predict_proba(inputs)+model.predict_proba(swap_teams(inputs))[:, order])/2
