"""Learn transition probabilities only from each fold's training labels.

This replaces the hand-set switch penalty with a three-state Markov prior.
Counts exclude discontinuities between non-contiguous training blocks. Swapped
team counts enforce equal treatment of home and away. No held-out labels enter
the transition matrix or the input probabilities.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from xgboost_baseline import CLASSES, probability_smoothing


def learn_transition(labels, frames):
    codes = pm.label_codes(labels[frames])
    valid = np.diff(frames) == 1
    counts = np.zeros((3, 3))
    np.add.at(counts, (codes[:-1][valid], codes[1:][valid]), 1)
    counts = counts + counts[[1, 0, 2]][:, [1, 0, 2]]
    counts += 1  # One fixed pseudocount per transition; no zero-probability prohibitions.
    return counts / counts.sum(axis=1, keepdims=True)


def decode_markov(probabilities, transition):
    probability = np.maximum(np.asarray(probabilities), 1e-12)
    emission = -np.log(probability / probability.sum(axis=1, keepdims=True))
    transition_cost = -np.log(transition)
    back = np.zeros_like(probability, dtype=int)
    cost = emission[0].copy()
    for frame in range(1, len(probability)):
        candidates = cost[:, None] + transition_cost
        back[frame] = candidates.argmin(axis=0)
        cost = emission[frame] + candidates[back[frame], np.arange(3)]
    path = np.empty(len(probability), dtype=int)
    path[-1] = cost.argmin()
    for frame in range(len(probability)-1, 0, -1):
        path[frame-1] = back[frame, path[frame]]
    return path


def main():
    output = ROOT / "data/markov_decoder"
    output.mkdir(exist_ok=True)
    gt = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    labels = gt.label.to_numpy()
    public = np.flatnonzero(labels != "X")
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    compact = pd.read_csv(ROOT / "data/factorized_baseline/oof_predictions.csv")
    rich = pd.read_csv(ROOT / "data/rich_baseline/oof_predictions.csv")
    probability_sets = {
        "compact": compact[[f"compact_multiclass_probability_{s}" for s in CLASSES]].to_numpy(),
        "factorized": compact[[f"factorized_probability_{s}" for s in CLASSES]].to_numpy(),
        "rich_geometry": rich[[f"rich_geometry_probability_{s}" for s in CLASSES]].to_numpy(),
    }
    predictions = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
    summaries, details = [], []
    for name, probabilities in probability_sets.items():
        for smooth in [False, True]:
            method = name + ("_smoothed" if smooth else "_raw")
            prediction = np.empty(len(public), dtype="<U1")
            for minute, train, validation in folds:
                ix = validation-public[0]
                transition = learn_transition(labels, train)
                p = probability_smoothing(probabilities[ix]) if smooth else probabilities[ix]
                prediction[ix] = CLASSES[decode_markov(p, transition)]
                score = pm.evaluate(pm.label_codes(labels[validation]), pm.label_codes(prediction[ix]))
                details.append(dict(model=method, minute=minute, spellF1=score["spell_f1_mean"]))
            predictions[method] = prediction
            score = pm.evaluate(pm.label_codes(labels[public]), pm.label_codes(prediction), bootstrap_n=1000)
            summaries.append(dict(model=method, spellF1=score["spell_f1_mean"], macroF1=score["macro_f1"],
                                  possessions=score["n_pred_spells"]))
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "oof_scores.csv", index=False)
    pd.DataFrame(details).to_csv(output / "fold_scores.csv", index=False)
    print(pd.DataFrame(details).pivot(index="model", columns="minute", values="spellF1").round(4))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
