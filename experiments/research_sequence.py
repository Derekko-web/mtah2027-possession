"""Input-dependent change costs and physically constrained stoppage evidence.

This is adaptive public development, with every evaluated alternative saved.
General rules use distances, collective activity and ball displacement; no frame
IDs or true interval endpoints enter inference. Both buffered and purged folds
use the same fixed rule definitions.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from research_stoppages import record_predictions
from run_continuation import purged_spell_blocks
from tempo_features import tempo_features
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing


def control_support(raw):
    result = []
    for own, other in (("h__", "a__"), ("a__", "h__")):
        close = (raw[own+"distance_1"] <= 1.5) & (raw[other+"distance_1"]-raw[own+"distance_1"] >= .5)
        result.append(close.astype(float).rolling(13, center=True, min_periods=1).sum().to_numpy() >= 3)
    return np.column_stack(result)


def adaptive_decode(probabilities, support, dead_penalty=3):
    probability = np.asarray(probabilities, dtype=float)
    if probability.shape != (len(support), 3) or not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Expected finite nonnegative three-state probabilities and aligned support")
    if (probability.sum(axis=1) <= 0).any():
        raise ValueError("Probability rows need positive mass")
    if not len(probability):
        return np.array([], dtype=int)
    loss = -np.log(np.maximum(probability/probability.sum(axis=1, keepdims=True), 1e-12))
    back = np.zeros((len(probability), 3), dtype=int)
    cost = loss[0]
    for frame in range(1, len(probability)):
        penalty = np.full((3, 3), float(dead_penalty))
        penalty[0, 1] = 3 if support[frame, 1] else 25
        penalty[1, 0] = 3 if support[frame, 0] else 25
        np.fill_diagonal(penalty, 0)
        candidates = cost[:, None]+penalty
        back[frame] = candidates.argmin(axis=0)
        cost = loss[frame]+candidates[back[frame], np.arange(3)]
    path = np.empty(len(probability), dtype=int)
    path[-1] = cost.argmin()
    for frame in range(len(probability)-1, 0, -1):
        path[frame-1] = back[frame, path[frame]]
    return path


def constrained_prior(raw, tempo, combine, activity="maximum"):
    nearest = raw[["h__distance_1", "a__distance_1"]].min(axis=1).to_numpy()
    contact = expit((nearest-2)/.75)
    slow_ball = expit((4.5-raw.ball_displacement_1s.to_numpy())/1)
    inactive = expit((1.5-tempo[f"collective_speed_{activity}_3s"].to_numpy())/.25)
    evidence = np.column_stack((contact, slow_ball, inactive))
    value = evidence.min(axis=1) if combine == "minimum" else evidence.prod(axis=1)
    value[~np.isfinite(value)] = 0
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"data/sequence_research")
    parser.add_argument("--stoppage-penalty", action="store_true", help="cost 25 for dead/live changes only")
    args = parser.parse_args()
    if args.stoppage_penalty and args.out == ROOT/"data/sequence_research":
        args.out = ROOT/"data/duration_research"
    args.out.mkdir(parents=True, exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    tempo = tempo_features(raw)
    support = control_support(raw)[public]
    priors = {name: constrained_prior(raw, tempo, name)[public] for name in ("minimum", "product")}
    priors.update({name+"_mean": constrained_prior(raw, tempo, name, "mean")[public]
                   for name in ("minimum", "product")})
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    for protocol, blocks in protocols.items():
        folder = args.out/protocol
        folder.mkdir(exist_ok=True)
        team = apply_context(np.load(ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy"),
                             gap_context(raw)[public], np.zeros(len(public)), False)
        ph = team[:, 0]/team[:, :2].sum(axis=1)
        probabilities = {"selected": team}
        for name, prior in priors.items():
            dead = np.maximum(team[:, 2], prior)
            probabilities["physical_"+name] = np.column_stack(((1-dead)*ph, (1-dead)*(1-ph), dead))
        # Blend a specialist with the conservative original rather than replacing it.
        for name, source in (("visible_specialist", "stoppage_research"), ("linear_audio", "audio_research")):
            head = np.load(ROOT/"data"/source/protocol/f"{name}_dead.npy")
            for fusion in ("equal_blend", "conservative_union", "agreement_only"):
                if fusion == "equal_blend":
                    dead = (head+team[:, 2])/2
                elif fusion == "conservative_union":
                    dead = np.maximum(team[:, 2], np.where(head > .8, head, 0))
                else:
                    dead = np.maximum(team[:, 2], np.where((head > .7) & (team[:, 2] > .15), head, 0))
                probabilities[name+"_"+fusion] = np.column_stack(((1-dead)*ph, (1-dead)*(1-ph), dead))
        if args.stoppage_penalty:
            probabilities = {name: p for name, p in probabilities.items() if name in
                             ("selected", "physical_minimum_mean", "physical_product_mean",
                              "visible_specialist_equal_blend", "linear_audio_equal_blend")}
            for family in ("monotone_class", "monotone_segments"):
                head = np.load(ROOT/"data/monotone_research"/protocol/f"{family}_head.npy")
                for fusion in ("equal", "union"):
                    dead = (head+team[:, 2])/2 if fusion == "equal" else np.maximum(team[:, 2], head)
                    probabilities[family+"_"+fusion] = np.column_stack(((1-dead)*ph, (1-dead)*(1-ph), dead))
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, probability in probabilities.items():
            for adaptive in (False, True):
                method = name+("_adaptive" if adaptive else "_fixed")
                prediction = np.empty(len(public), dtype="<U1")
                for _, _, validation in blocks:
                    ix = validation-public[0]
                    smooth = probability_smoothing(probability[ix])
                    if args.stoppage_penalty:
                        contact = support[ix] if adaptive else np.ones((len(ix), 2), dtype=bool)
                        path = adaptive_decode(smooth, contact, dead_penalty=25)
                    else:
                        path = adaptive_decode(smooth, support[ix]) if adaptive else decode(smooth)
                    prediction[ix] = CLASSES[path]
                table[method] = prediction
                record_predictions(args.out, protocol, method, public, labels, blocks, prediction,
                                   rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.out/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(args.out/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(args.out/"segment_diagnostics.csv", index=False)
    (args.out/"method_definitions.json").write_text(json.dumps(dict(
        contact_m=2, ball_speed_m_s=4.5, activity_m_s=1.5, activity_max_window_frames=76,
        unsupported_team_change_penalty=25, supported_change_penalty=3,
        warning="Adaptive public development, no private/fresh audit labels; all alternatives retained."), indent=2)+"\n")
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
