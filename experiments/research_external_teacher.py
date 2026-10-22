"""Transfer a player-motion-only stoppage teacher from two public sample games.

Teacher fitting never sees competition labels. Its probability can be consumed
as a pretrained feature by the original two-objective target ensemble, fitted
strictly on each fold's eligible public labels. No corrected fixture data is used.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from external_stoppages import DATA, competition_features
from research_sequence import adaptive_decode, control_support
from research_stoppages import record_predictions
from run_continuation import fit_ensemble, purged_spell_blocks
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing

TEACHER_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                      max_depth=3, n_estimators=240, learning_rate=.04, reg_lambda=10,
                      min_child_weight=50, subsample=.8, colsample_bytree=1., random_state=42, n_jobs=2)


def player_columns(features):
    return [c for c in features if c.startswith(("speed_median_", "moving_fraction_"))]


def train_teachers(output):
    games = {}
    for game in (1, 2):
        inputs = pd.read_pickle(DATA/f"game_{game}_features.pkl")
        labels = np.load(DATA/f"game_{game}_weak_labels.npy")
        eligible = np.flatnonzero(labels != "X")[::5]
        games[game] = (inputs[player_columns(inputs)].iloc[eligible], (labels[eligible] == "D").astype(int))
    models, evidence = [], []
    for game in (1, 2):
        x, y = games[game]
        model = XGBClassifier(**TEACHER_PARAMS, monotone_constraints=tuple(-1 for _ in x.columns))
        weight = len(y)/(2*np.bincount(y)[y])
        model.fit(x, y, sample_weight=weight)
        model.save_model(output/f"external_game_{game}_teacher.ubj")
        other_x, other_y = games[3-game]
        p = model.predict_proba(other_x)[:, 1]
        evidence.append(dict(train_game=game, held_out_game=3-game,
                             weak_label_balanced_accuracy=balanced_accuracy_score(other_y, p > .5),
                             weak_label_ROC_AUC=roc_auc_score(other_y, p), training_rows=len(y)))
        models.append(model)
    return models, evidence


def main():
    output = ROOT/"data/external_teacher_research"
    output.mkdir(exist_ok=True)
    models, evidence = train_teachers(output)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    physical = competition_features(len(base))
    physical.to_pickle(output/"competition_inputs.pkl")
    features = physical[player_columns(physical)]
    teacher = np.mean([m.predict_proba(features)[:, 1] for m in models], axis=0)
    np.save(output/"teacher_probability.npy", teacher)
    x = base.assign(external_teacher_dead=teacher)
    variants = {"teacher_ensemble": x,
                "teacher_without_location": x.drop(columns=["ball_goal_line_distance", "ball_touchline_distance"])}
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    side = gap_context(raw)[public]
    support = control_support(raw)[public]
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        parent = apply_context(np.load(ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy"),
                               side, np.zeros(len(public)), False)
        ph = parent[:, 0]/parent[:, :2].sum(axis=1)
        probabilities = {}
        for name, inputs in variants.items():
            print(protocol, name, flush=True)
            model_folder = folder/name
            model_folder.mkdir(exist_ok=True)
            p = np.full((len(public), 3), np.nan)
            for minute, train, validation in blocks:
                p[validation-public[0]] = fit_ensemble(inputs, labels, train, validation, model_folder, minute)
            probabilities[name] = apply_context(p, side, np.zeros(len(public)), False)
            np.save(model_folder/"oof_probabilities.npy", p)
        for fusion in ("direct", "equal", "visible_equal"):
            dead = teacher[public].copy() if fusion == "direct" else (teacher[public]+parent[:, 2])/2
            if fusion == "visible_equal":
                mask = base.ball_seen.iloc[public].to_numpy() != 1
                dead[mask] = parent[mask, 2]
            probabilities["teacher_"+fusion] = np.column_stack(((1-dead)*ph, (1-dead)*(1-ph), dead))
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, probability in probabilities.items():
            for adaptive in (False, True):
                method = name+("_adaptive" if adaptive else "_fixed")
                prediction = np.empty(len(public), dtype="<U1")
                for _, _, validation in blocks:
                    ix = validation-public[0]
                    smooth = probability_smoothing(probability[ix])
                    prediction[ix] = CLASSES[adaptive_decode(smooth, support[ix]) if adaptive else decode(smooth)]
                table[method] = prediction
                record_predictions(output, protocol, method, public, labels, blocks, prediction, rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    (output/"manifest.json").write_text(json.dumps(dict(
        teacher_parameters=TEACHER_PARAMS, teacher_features=player_columns(physical),
        external_holdout_checks=evidence,
        source="https://github.com/metrica-sports/sample-data/tree/e706dd506b360d69d9d123d5b8026e7294b13996",
        warning="External labels are event-derived and weak; competition scores are adaptive public OOF only."), indent=2)+"\n")
    print(json.dumps(evidence, indent=2))
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
