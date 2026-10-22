"""General input-only hypotheses for occlusions and slow unclaimed balls.

These are adaptive development hypotheses, not independent confirmation. No
frame-specific rules, labels, event boundaries or match-clock inputs are used.
Each occlusion considers at most 9.48 seconds of gap and 0.52 seconds of preceding
observations, so training-label buffers of ten seconds remain sufficient.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from xgboost_baseline import CLASSES, probability_smoothing
from transition_penalty import decode


def gap_context(features):
    """Preserve the previous nearest-team evidence during an active short occlusion.

    The initial before/after agreement rule fired on only five public frames.
    This revised hypothesis follows the possession rule that an unobserved ball
    does not itself establish a change of control. At most 9.48 seconds of gap
    and 0.52 seconds of preceding inputs are inspected, within the 10-s buffers.
    """
    seen = features.ball_seen.to_numpy() == 1
    control = features.h__control_share.to_numpy()
    movement = features[["h__speed_median", "a__speed_median"]].mean(axis=1).to_numpy()
    side = np.full(len(features), -1, dtype=int)
    gaps = np.flatnonzero(np.diff(np.r_[False, ~seen, False]))
    for start, end in zip(gaps[::2], gaps[1::2]):
        if end-start > 237 or start < 13:
            continue
        before = control[start-13:start]
        before = before[np.isfinite(before)]
        if len(before) < 3:
            continue
        b = np.median(before)
        if not np.isfinite(movement[start:end]).any() or np.nanmedian(movement[start:end]) <= 1.5:
            continue
        if b > .5:
            side[start:end] = 0
        elif b < .5:
            side[start:end] = 1
    return side


def physical_probabilities(features):
    nearest = features[["h__distance_1", "a__distance_1"]].min(axis=1).to_numpy()
    movement = features[["h__speed_median", "a__speed_median"]].mean(axis=1).to_numpy()
    speed = features["ball_displacement_1s"].to_numpy()
    # A soft hypothesis: ball far beyond reach, moving slowly, players relatively slow.
    dead = expit((nearest-3)/.75)*expit((4.5-speed)/1)*expit((2-movement)/.5)
    dead[~np.isfinite(dead)] = 0
    return dead


def apply_context(probability, side, prior, add_dead):
    result = probability.copy()
    for team in (0, 1):
        mask = side == team
        # Soft evidence, not a hard label override. Preserve competing model evidence.
        context = np.tile([.075, .075, .05], (int(mask.sum()), 1))
        context[:, team] = .875
        result[mask] = .25*result[mask] + .75*context
    if add_dead:
        pd_ = np.maximum(result[:, 2], prior)
        alive = result[:, :2] / np.maximum(result[:, :2].sum(axis=1, keepdims=True), 1e-12)
        result = np.column_stack((alive*(1-pd_)[:, None], pd_))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persistent-dead',action='store_true',
                        help='require at least two seconds of slow-unclaimed-ball evidence')
    args=parser.parse_args()
    output = ROOT / ('data/physics_context_persistent' if args.persistent_dead else 'data/physics_context_momentum')
    output.mkdir(exist_ok=True)
    features = pd.read_pickle(ROOT / "data/rich_baseline/input_features.pkl")
    gt = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    labels = gt.label.to_numpy()
    public = np.flatnonzero(labels != "X")
    side = gap_context(features)[public]
    prior = physical_probabilities(features)[public]
    if args.persistent_dead:
        # Accept the whole qualifying run retrospectively. Two seconds of future
        # input evidence at most is needed to establish persistence, with no labels.
        valid=prior>.5
        edges=np.flatnonzero(np.diff(np.r_[False,valid,False]))
        keep=np.zeros(len(prior),dtype=bool)
        for start,end in zip(edges[::2],edges[1::2]):
            if end-start>=50:keep[start:end]=True
        prior[~keep]=0
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    compact = pd.read_csv(ROOT / "data/factorized_baseline/oof_predictions.csv")
    probabilities = {
        "xgb_compact": compact[[f"compact_multiclass_probability_{s}" for s in CLASSES]].to_numpy(),
        "extra_compact": np.load(ROOT / "data/tree_ensembles/extra_compact_oof_probabilities.npy"),
        "forest_compact": np.load(ROOT / "data/tree_ensembles/forest_compact_oof_probabilities.npy"),
    }
    if args.persistent_dead:
        probabilities['kinematics']=np.load(ROOT/'data/kinematics_baseline/oof_probabilities.npy')
        probabilities['relative_motion']=np.load(ROOT/'data/relative_motion_baseline/oof_probabilities.npy')
    table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
    summaries, fold_scores = [], []
    print("Input-only gap continuity evidence on", int((side >= 0).sum()), "public frames")
    for name, p in probabilities.items():
        for with_dead in [False, True]:
            key = name + ("_gap_and_dead" if with_dead else "_gap_only")
            adjusted = apply_context(p, side, prior, with_dead)
            prediction = np.empty(len(public), dtype="<U1")
            for minute, _, validation in folds:
                ix = validation-public[0]
                prediction[ix] = CLASSES[decode(probability_smoothing(adjusted[ix]))]
                score = pm.evaluate(pm.label_codes(labels[validation]), pm.label_codes(prediction[ix]))
                fold_scores.append(dict(model=key, minute=minute, spellF1=score["spell_f1_mean"]))
            table[key] = prediction
            score = pm.evaluate(pm.label_codes(labels[public]), pm.label_codes(prediction), bootstrap_n=1000)
            summaries.append(dict(model=key, spellF1=score["spell_f1_mean"], macroF1=score["macro_f1"],
                                  possessions=score["n_pred_spells"]))
    table.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "oof_scores.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(output / "fold_scores.csv", index=False)
    print(pd.DataFrame(fold_scores).pivot(index="model", columns="minute", values="spellF1").round(4))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
