"""Explore new dead/live heads with preserved team probabilities and two protocols.

No early stopping or held-out fitting. This is openly adaptive development on the
same public window. Six fixed head designs and joint/hierarchical composition
are compared; every result, including failures, is saved. Nothing is promoted or
submitted by this runner.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from run_continuation import purged_spell_blocks
from sequence_diagnostics import sequence_details
from stoppage_features import build_stoppage_features
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing

HEADS = {
    "physical_class": dict(family="tree", features="physical", weights="class", visible=False),
    "physical_segments": dict(family="tree", features="physical", weights="segments", visible=False),
    "restart_segments": dict(family="tree", features="combined", weights="segments", visible=False),
    "visible_specialist": dict(family="tree", features="physical", weights="segments", visible=True),
    "linear_physical": dict(family="linear", features="physical", weights="segments", visible=False),
    "linear_restart": dict(family="linear", features="combined", weights="segments", visible=False),
}
AUDIO_HEADS = {
    "audio_only": dict(family="tree", features="audio", weights="class", visible=False),
    "audio_physical": dict(family="tree", features="audio_physical", weights="class", visible=False),
    "audio_segments": dict(family="tree", features="audio_physical", weights="segments", visible=False),
    "audio_restart": dict(family="tree", features="audio_combined", weights="segments", visible=False),
    "linear_audio": dict(family="linear", features="audio_physical", weights="segments", visible=False),
}
TEMPO_HEADS = {
    "tempo_tree": dict(family="tree", features="tempo", weights="segments", visible=False),
    "tempo_linear": dict(family="linear", features="tempo", weights="segments", visible=False),
    "tempo_physical": dict(family="tree", features="tempo_physical", weights="segments", visible=False),
    "tempo_restart": dict(family="tree", features="tempo_combined", weights="segments", visible=False),
}
TREE_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                   n_estimators=160, max_depth=2, learning_rate=.04, min_child_weight=15,
                   reg_lambda=10, subsample=.8, colsample_bytree=.9, random_state=42, n_jobs=2)


def binary_weights(target, frames, segment_balance):
    target = np.asarray(target)
    if set(target) != {0, 1}:
        raise ValueError("Dead/live training needs both classes")
    if not segment_balance:
        return len(target)/(2*np.bincount(target)[target])
    boundaries = np.r_[True, (np.diff(frames) != 1) | (target[1:] != target[:-1])]
    segment = np.cumsum(boundaries)-1
    weights = 1/np.bincount(segment)[segment]
    # Every dead segment shares its class mass equally; likewise live segments.
    for label in (0, 1):
        weights[target == label] /= weights[target == label].sum()
    return weights/weights.mean()


def fit_head(inputs, labels, train, validation, config):
    y = (labels[train] == "D").astype(int)
    weight = binary_weights(y, train, config["weights"] == "segments")
    if config["family"] == "tree":
        model = XGBClassifier(**TREE_PARAMS)
        model.fit(inputs.iloc[train], y, sample_weight=weight)
    else:
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              StandardScaler(), LogisticRegression(C=.1, max_iter=2000, random_state=42))
        model.fit(inputs.iloc[train], y, logisticregression__sample_weight=weight)
    return model, model.predict_proba(inputs.iloc[validation])[:, 1]


def compose_sequence(team_probability, dead, folds, public, hierarchical):
    home = team_probability[:, 0]/np.maximum(team_probability[:, :2].sum(axis=1), 1e-12)
    prediction = np.empty(len(public), dtype="<U1")
    for _, _, validation in folds:
        ix = validation-public[0]
        if hierarchical:
            d = decode(probability_smoothing(np.column_stack((1-dead[ix], dead[ix]))))
            t = decode(probability_smoothing(np.column_stack((home[ix], 1-home[ix]))))
            prediction[ix] = np.where(d == 1, "D", CLASSES[t])
        else:
            p = np.column_stack(((1-dead[ix])*home[ix], (1-dead[ix])*(1-home[ix]), dead[ix]))
            prediction[ix] = CLASSES[decode(probability_smoothing(p))]
    return prediction


def record_predictions(output, protocol, method, public, labels, folds, prediction, rows, details, segments):
    summary, spans = sequence_details(public, labels[public], prediction)
    for span in spans:
        segments.append(dict(protocol=protocol, model=method, **span))
    for minute, _, validation in folds:
        ix = validation-public[0]
        score, _ = sequence_details(validation, labels[validation], prediction[ix])
        details.append(dict(protocol=protocol, model=method, minute=minute, **score))
    summary["mean_minute_spellF1"] = float(np.mean([r["spellF1"] for r in details
                                                   if r["protocol"] == protocol and r["model"] == method]))
    rows.append(dict(protocol=protocol, model=method, **summary))
    target = output/protocol/"scorer_inputs"
    target.mkdir(exist_ok=True)
    full = np.full(len(labels), "X", dtype="<U1")
    full[public] = prediction
    pd.DataFrame({"frame": np.arange(len(full)), "label": full}).to_csv(target/f"{method}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--audio", action="store_true", help="test local spectral evidence")
    parser.add_argument("--tempo", action="store_true", help="test sustained collective inactivity")
    args = parser.parse_args()
    if args.audio and args.tempo:
        parser.error("Audio and tempo are separate research rounds")
    args.out = args.out or ROOT/("data/tempo_research" if args.tempo else
                                 "data/audio_research" if args.audio else "data/stoppage_research")
    args.out.mkdir(parents=True, exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    physical, restarts = build_stoppage_features(base)
    inputs = {"physical": physical, "combined": physical.join(restarts)}
    heads = HEADS
    if args.audio:
        from audio_features import build_audio_features
        cache = ROOT/"data/stoppage_research/audio_inputs.pkl"
        audio = pd.read_pickle(cache) if cache.exists() else build_audio_features(len(base))
        if len(audio) != len(base):
            raise ValueError("Audio cache length differs from tracking inputs")
        inputs.update(audio=audio, audio_physical=physical.join(audio),
                      audio_combined=physical.join(restarts).join(audio))
        heads = AUDIO_HEADS
    if args.tempo:
        from tempo_features import tempo_features
        tempo = tempo_features(raw)
        inputs.update(tempo=tempo, tempo_physical=physical.join(tempo),
                      tempo_combined=physical.join(restarts).join(tempo))
        heads = TEMPO_HEADS
    for name, features in inputs.items():
        features.to_pickle(args.out/f"{name}_inputs.pkl")
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    side = gap_context(raw)[public]
    for protocol, blocks in protocols.items():
        folder = args.out/protocol
        folder.mkdir(exist_ok=True)
        parent = ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy"
        team = apply_context(np.load(parent), side, np.zeros(len(public)), False)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, config in heads.items():
            print(protocol, name, flush=True)
            dead = np.full(len(public), np.nan)
            for minute, train, validation in blocks:
                if config["visible"]:
                    eligible = train[base.ball_seen.iloc[train].to_numpy() == 1]
                else:
                    eligible = train
                model, prediction = fit_head(inputs[config["features"]], labels, eligible, validation, config)
                dead[validation-public[0]] = prediction
                if config["family"] == "tree":
                    model.save_model(folder/f"{name}_fold_{minute}.ubj")
            if config["visible"]:
                # Specialize only where a ball observation exists; preserve the
                # original detector on occluded frames instead of extrapolating.
                seen = base.ball_seen.iloc[public].to_numpy() == 1
                dead[~seen] = team[~seen, 2]
            if not np.isfinite(dead).all():
                raise ValueError("Incomplete dead/live OOF probabilities")
            np.save(folder/f"{name}_dead.npy", dead)
            for hierarchical in (False, True):
                method = name+("_hierarchical" if hierarchical else "_joint")
                prediction = compose_sequence(team, dead, blocks, public, hierarchical)
                table[method] = prediction
                record_predictions(args.out, protocol, method, public, labels, blocks, prediction,
                                   rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.out/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(args.out/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(args.out/"segment_diagnostics.csv", index=False)
    manifest = dict(heads=heads, tree_parameters=TREE_PARAMS,
                    features={name: list(x.columns) for name, x in inputs.items()},
                    source_sha256={name: hashlib.sha256((ROOT/"experiments"/name).read_bytes()).hexdigest()
                                   for name in ("stoppage_features.py", "research_stoppages.py")},
                    warning="Adaptive search on the same 17 public possessions; no private or fresh audit labels.")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))
    print(pd.DataFrame(details).pivot(index=["protocol", "model"], columns="minute", values="spellF1").round(4))


if __name__ == "__main__":
    main()
