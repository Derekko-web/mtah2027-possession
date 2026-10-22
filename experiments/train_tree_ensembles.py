"""Fixed random-forest/extra-trees alternatives and a soft baseline anchor.

Medians and missingness indicators are fitted on training rows only. Training
uses team-swap augmentation; features and validation buffers match prior runs.
No parameter sweep or validation early stopping is performed.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from rich_features import swap_teams
from train_factorized_baseline import compact_columns
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing


def main():
    output = ROOT / "data/tree_ensembles"
    output.mkdir(exist_ok=True)
    x = pd.read_pickle(ROOT / "data/rich_baseline/input_features.pkl")
    gt = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    labels = gt.label.to_numpy()
    public = np.flatnonzero(labels != "X")
    reference = pm.read_label_csv(ROOT / "evaluation/results/baseline_smooth50.csv").label.to_numpy()
    soft_reference = np.array([[.8 if value == state else .1 for state in CLASSES] for value in reference])
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    specifications = {
        "extra_compact": (ExtraTreesClassifier, compact_columns(x)),
        "forest_compact": (RandomForestClassifier, compact_columns(x)),
        "extra_geometry": (ExtraTreesClassifier, [name for name in x if "tag_" not in name]),
        "forest_geometry": (RandomForestClassifier, [name for name in x if "tag_" not in name]),
    }
    probabilities = {name: np.full((len(public), 3), np.nan) for name in specifications}
    for minute, train, validation in folds:
        y = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
        y_aug = np.r_[y, np.where(y == 0, 1, np.where(y == 1, 0, 2))]
        for name, (family, columns) in specifications.items():
            print("Training", name, "fold", minute, flush=True)
            train_x = pd.concat([x.iloc[train][columns], swap_teams(x.iloc[train][columns])])
            imputer = SimpleImputer(add_indicator=True, keep_empty_features=True)
            train_array = imputer.fit_transform(train_x)
            model = family(n_estimators=200, max_depth=8, min_samples_leaf=20,
                           max_features=.8, class_weight="balanced", random_state=42, n_jobs=2)
            model.fit(train_array, y_aug)
            p = (model.predict_proba(imputer.transform(x.iloc[validation][columns])) +
                 model.predict_proba(imputer.transform(swap_teams(x.iloc[validation][columns])))[:, [1, 0, 2]])/2
            probabilities[name][validation-public[0]] = p
    table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
    summaries, details = [], []
    for name, p in probabilities.items():
        np.save(output / f"{name}_oof_probabilities.npy", p)
        for anchor in [False, True]:
            key = name + ("_anchored" if anchor else "")
            prediction = np.empty(len(public), dtype="<U1")
            for minute, _, validation in folds:
                ix = validation-public[0]
                block = probability_smoothing(p[ix])
                if anchor:
                    block = .5*block + .5*soft_reference[validation]
                prediction[ix] = CLASSES[decode(block)]
                score = pm.evaluate(pm.label_codes(labels[validation]), pm.label_codes(prediction[ix]))
                details.append(dict(model=key, minute=minute, spellF1=score["spell_f1_mean"]))
            table[key] = prediction
            score = pm.evaluate(pm.label_codes(labels[public]), pm.label_codes(prediction), bootstrap_n=1000)
            summaries.append(dict(model=key, spellF1=score["spell_f1_mean"], macroF1=score["macro_f1"],
                                  possessions=score["n_pred_spells"]))
    table.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "oof_scores.csv", index=False)
    pd.DataFrame(details).to_csv(output / "fold_scores.csv", index=False)
    print(pd.DataFrame(details).pivot(index="model", columns="minute", values="spellF1").round(4))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
