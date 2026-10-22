"""Run three frozen alternatives and a whole-spell exclusion stress check.

First run run_selected.py. All generated arrays, models, scores and tracking-derived
artifacts stay under ignored data/. This script never changes the selected model
or the existing candidate. It refuses to use a different public-label window.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_candidate import PARAMS, objective_weights
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from rich_features import swap_teams
from robust_motion import build_motion_inputs
from sequence_diagnostics import sequence_details
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing

PLAN_PATH = ROOT / "experiments/CONTINUATION_PLAN.json"
METHODS = ("selected", "robust_ball_motion", "slot_invariant_motion", "penalty25")


def purged_spell_blocks(labels, folds):
    """Use labels only to define excluded groups; return no labels to the learner."""
    labels = np.asarray(labels)
    segments = pm.segments(pm.label_codes(labels), pm.valid_runs(pm.label_codes(labels)))
    result = []
    for minute, train, validation in folds:
        eligible = np.zeros(len(labels), dtype=bool)
        eligible[train] = True
        for start, stop, _ in segments:
            if start <= validation[-1] and stop > validation[0]:
                eligible[start:stop] = False
        result.append((minute, np.flatnonzero(eligible), validation.copy()))
    return result


def fit_ensemble(inputs, labels, train, validation, output, minute):
    target = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
    if set(target) != {0, 1, 2}:
        raise ValueError("Fold training lacks a target class")
    augmented_x = pd.concat([inputs.iloc[train], swap_teams(inputs.iloc[train])], ignore_index=True)
    probabilities = []
    for balance, name in ((False, "frame"), (True, "segment")):
        y, weight = objective_weights(target, train, balance)
        model = XGBClassifier(**PARAMS)
        model.fit(augmented_x, y, sample_weight=weight)
        probabilities.append((model.predict_proba(inputs.iloc[validation]) +
                              model.predict_proba(swap_teams(inputs.iloc[validation]))[:, [1, 0, 2]]) / 2)
        model.save_model(output / f"fold_{minute}_{name}.ubj")
    # Match selected OOF's float64 accumulation before the continuity mixture.
    return (probabilities[0].astype(float) + probabilities[1].astype(float)) / 2


def decode_folds(probability, side, public, folds, penalty):
    adjusted = apply_context(probability, side, np.zeros(len(public)), False)
    prediction = np.empty(len(public), dtype="<U1")
    for _, _, validation in folds:
        ix = validation-public[0]
        prediction[ix] = CLASSES[decode(probability_smoothing(adjusted[ix]), penalty=penalty)]
    return prediction


def screen_promotion(scores, fold_scores, plan):
    """Apply the predeclared gates; never select by a favorable single metric."""
    gate = plan["promotion_requirements"]
    result = {}
    index = scores.set_index(["protocol", "model"])
    detail = fold_scores.set_index(["protocol", "model", "minute"]).sort_index()
    for method in METHODS[1:]:
        baseline = index.loc[("buffered_minutes", "selected")]
        candidate = index.loc[("buffered_minutes", method)]
        purged_base = index.loc[("purged_spells", "selected")]
        purged = index.loc[("purged_spells", method)]
        delta = (detail.loc[("buffered_minutes", method), "spellF1"] -
                 detail.loc[("buffered_minutes", "selected"), "spellF1"])
        checks = dict(
            stitched_gain=bool(candidate.spellF1-baseline.spellF1 >= gate["buffered_stitched_spellF1_gain_at_least"]),
            mean_minute_gain=bool(candidate.mean_minute_spellF1-baseline.mean_minute_spellF1 >= gate["buffered_mean_minute_spellF1_gain_at_least"]),
            minute_regressions=bool(delta.min() >= -gate["maximum_minute_spellF1_regression"]),
            fewer_missed_stoppages=bool(candidate.missed_stoppages < baseline.missed_stoppages),
            false_dead_frames=bool(candidate.false_dead_frames-baseline.false_dead_frames <= gate["additional_false_dead_frames_at_most"]),
            purged_stitched=bool(purged.spellF1-purged_base.spellF1 >= -gate["purged_stitched_spellF1_regression_at_most"]),
            purged_mean_minute=bool(purged.mean_minute_spellF1-purged_base.mean_minute_spellF1 >= -gate["purged_mean_minute_spellF1_regression_at_most"]),
            purged_stoppages=bool(purged.missed_stoppages <= purged_base.missed_stoppages),
        )
        result[method] = dict(eligible=all(checks.values()), checks=checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/continuation_20261021")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN_PATH.read_text())
    gt_path = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(gt_path).label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    raw = pd.read_pickle(ROOT / "data/rich_baseline/input_features.pkl")
    base = pd.read_pickle(ROOT / "data/relative_motion_baseline/input_features.pkl")
    robust, position_sets = build_motion_inputs(len(raw))
    replace = [p+n for p in ("h__", "a__") for n in
               ("nearest_relative_velocity", "relative_velocity_within_3m")]
    features = {"selected": base, "robust_ball_motion": base.join(robust),
                "slot_invariant_motion": base.drop(columns=replace).join(position_sets)}
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    side = gap_context(raw)[public]
    rows, details, segments, split_rows = [], [], [], []
    for protocol, blocks in protocols.items():
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        probabilities = {}
        for method, inputs in features.items():
            output = args.out / protocol / method
            output.mkdir(parents=True, exist_ok=True)
            if protocol == "buffered_minutes" and method == "selected":
                a = np.load(ROOT / "data/relative_motion_baseline/oof_probabilities.npy")
                b = np.load(ROOT / "data/relative_motion_baseline_segment_weights/oof_probabilities.npy")
                p = (a+b)/2
            else:
                p = np.full((len(public), 3), np.nan)
                for minute, train, validation in blocks:
                    print(protocol, method, "fold", minute, flush=True)
                    if np.intersect1d(train, validation).size or (labels[train] == "X").any():
                        raise ValueError("Invalid training pool")
                    p[validation-public[0]] = fit_ensemble(inputs, labels, train, validation, output, minute)
            if p.shape != (7500, 3) or not np.isfinite(p).all():
                raise ValueError("Incomplete OOF probabilities")
            probabilities[method] = p
            np.save(output / "oof_probabilities.npy", p)
        probabilities["penalty25"] = probabilities["selected"]
        for method in METHODS:
            prediction = decode_folds(probabilities[method], side, public, blocks, 25 if method == "penalty25" else 3)
            if protocol == "buffered_minutes" and method == "selected":
                reference = pd.read_csv(ROOT / "data/selected_evaluation/oof_predictions.csv")
                np.testing.assert_array_equal(reference.selected_ensemble.to_numpy(), prediction)
            table[method] = prediction
            summary, spans = sequence_details(public, labels[public], prediction)
            for span in spans:
                segments.append(dict(protocol=protocol, model=method, **span))
            for minute, train, validation in blocks:
                ix = validation-public[0]
                score, _ = sequence_details(validation, labels[validation], prediction[ix])
                details.append(dict(protocol=protocol, model=method, minute=minute, **score))
                if method == "selected":
                    # Explicit frame indices stay in ignored artifacts for audit.
                    split_rows.append(dict(protocol=protocol, minute=minute,
                                           training_frames=train.tolist(), validation_frames=validation.tolist()))
            summary["mean_minute_spellF1"] = float(np.mean([r["spellF1"] for r in details
                                                           if r["protocol"] == protocol and r["model"] == method]))
            rows.append(dict(protocol=protocol, model=method, **summary))
            oof = np.full(len(labels), "X", dtype="<U1")
            oof[public] = prediction
            target = args.out / protocol / "scorer_inputs"
            target.mkdir(exist_ok=True)
            pd.DataFrame({"frame": np.arange(len(labels)), "label": oof}).to_csv(target / f"{method}.csv", index=False)
        table.to_csv(args.out / protocol / "oof_predictions.csv", index=False)
    scores, fold_scores = pd.DataFrame(rows), pd.DataFrame(details)
    scores.to_csv(args.out / "scores.csv", index=False)
    fold_scores.to_csv(args.out / "fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(args.out / "segment_diagnostics.csv", index=False)
    (args.out / "splits.json").write_text(json.dumps(split_rows, indent=2) + "\n")
    decision = screen_promotion(scores, fold_scores, plan)
    source_files = [Path(__file__), PLAN_PATH, ROOT / "experiments/robust_motion.py",
                    ROOT / "experiments/sequence_diagnostics.py"]
    manifest = dict(plan=plan, decision=decision,
                    public_gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                    features={name: list(x.columns) for name, x in features.items()},
                    new_alternatives=3, method_protocol_evaluations=len(scores),
                    warning="Adaptive public development; purged spells are a stress check, not independent labels.")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(scores.to_string(index=False))
    print(fold_scores.pivot(index=["protocol", "model"], columns="minute", values="spellF1").round(4))
    print("Promotion screening:", json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
