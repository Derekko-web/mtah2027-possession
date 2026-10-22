"""Nested cross-fitting of dead/live fusion; outer labels never train the stacker.

For each outer fold, inner minute models train on the intersection of eligible
outer and inner training rows. Their predictions on eligible inner validation
rows form stacker training inputs. Full outer validation labels are never seen
by a base model, specialist or meta model used for that outer prediction.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from research_monotone import monotone_inputs, PARAMS as MONOTONE_PARAMS
from research_sequence import adaptive_decode, control_support
from research_stoppages import binary_weights, record_predictions
from run_continuation import fit_ensemble, purged_spell_blocks
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing


def stack_inputs(p, specialist, teacher):
    matrix = np.column_stack((p[:, 2], specialist, teacher))
    logit = np.log(np.clip(matrix, .001, .999)/np.clip(1-matrix, .001, .999))
    return pd.DataFrame(np.column_stack((matrix, logit)), columns=
                        ["base_dead", "specialist_dead", "external_dead", "base_logit", "specialist_logit", "external_logit"])


def main():
    output = ROOT/"data/stacked_dead_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    mono_x, signs = monotone_inputs(base)
    teacher = np.load(ROOT/"data/external_teacher_research/teacher_probability.npy")
    side = gap_context(raw)
    support = control_support(raw)[public]
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    audit_splits = []
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        parent = apply_context(np.load(ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy"),
                               side[public], np.zeros(len(public)), False)
        specialist = np.load(ROOT/"data/monotone_research"/protocol/"monotone_segments_head.npy")
        validation_inputs = stack_inputs(parent, specialist, teacher[public])
        probabilities = {family: np.full(len(public), np.nan) for family in ("linear", "tree")}
        for outer_minute, outer_train, outer_validation in blocks:
            print(protocol, "outer", outer_minute, flush=True)
            train_inputs, train_frames = [], []
            for inner_minute, inner_train, inner_validation in folds:
                if inner_minute == outer_minute:
                    continue
                train = np.intersect1d(outer_train, inner_train)
                validation = np.intersect1d(outer_train, inner_validation)
                if not len(validation):
                    continue
                if np.intersect1d(train, outer_validation).size or np.intersect1d(validation, outer_validation).size:
                    raise ValueError("Nested split touches outer validation")
                target = folder/f"outer_{outer_minute}_inner_{inner_minute}"
                target.mkdir(exist_ok=True)
                p = fit_ensemble(base, labels, train, validation, target, inner_minute)
                p = apply_context(p, side[validation], np.zeros(len(validation)), False)
                eligible = train[base.ball_seen.iloc[train].to_numpy() == 1]
                y = (labels[eligible] == "D").astype(int)
                model = XGBClassifier(**MONOTONE_PARAMS, monotone_constraints=signs)
                model.fit(mono_x.iloc[eligible], y, sample_weight=binary_weights(y, eligible, True))
                head = model.predict_proba(mono_x.iloc[validation])[:, 1]
                missing = base.ball_seen.iloc[validation].to_numpy() != 1
                head[missing] = p[missing, 2]
                train_inputs.append(stack_inputs(p, head, teacher[validation]))
                train_frames.append(validation)
                audit_splits.append(dict(protocol=protocol, outer=outer_minute, inner=inner_minute,
                                         train=train.tolist(), validation=validation.tolist()))
            x = pd.concat(train_inputs, ignore_index=True)
            frames = np.concatenate(train_frames)
            order = np.argsort(frames)
            x, frames = x.iloc[order], frames[order]
            y = (labels[frames] == "D").astype(int)
            weight = binary_weights(y, frames, True)
            ix = outer_validation-public[0]
            for family in probabilities:
                if family == "linear":
                    meta = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000, random_state=42))
                    meta.fit(x, y, logisticregression__sample_weight=weight)
                else:
                    meta = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                         max_depth=2, n_estimators=80, learning_rate=.04, reg_lambda=10,
                                         min_child_weight=30, random_state=42, n_jobs=2)
                    meta.fit(x, y, sample_weight=weight)
                    meta.save_model(folder/f"meta_{family}_outer_{outer_minute}.ubj")
                probabilities[family][ix] = meta.predict_proba(validation_inputs.iloc[ix])[:, 1]
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        ph = parent[:, 0]/parent[:, :2].sum(axis=1)
        for family, dead in probabilities.items():
            np.save(folder/f"{family}_dead.npy", dead)
            p = np.column_stack(((1-dead)*ph, (1-dead)*(1-ph), dead))
            for adaptive in (False, True):
                for dead_penalty in (3, 25):
                    method = family+("_adaptive" if adaptive else "_fixed")+f"_dead{dead_penalty}"
                    prediction = np.empty(len(public), dtype="<U1")
                    for _, _, validation in blocks:
                        ix = validation-public[0]
                        smooth = probability_smoothing(p[ix])
                        contact = support[ix] if adaptive else np.ones((len(ix), 2), dtype=bool)
                        prediction[ix] = CLASSES[adaptive_decode(smooth, contact, dead_penalty=dead_penalty)]
                    table[method] = prediction
                    record_predictions(output, protocol, method, public, labels, blocks, prediction, rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    (output/"nested_splits.json").write_text(__import__("json").dumps(audit_splits)+"\n")
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
