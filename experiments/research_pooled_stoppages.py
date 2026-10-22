"""Pool asynchronous physical evidence over sustained low-activity episodes.

A retrieved ball can alternate between near/fast and far/slow observations.
Requiring all cues on one frame discards such evidence. Here candidate episodes
come only from player activity; their ball-distance and ball-speed majorities
are checked separately. No labels or hand-picked frame spans enter the rule.
"""

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from research_sequence import adaptive_decode, control_support
from research_stoppages import record_predictions
from run_continuation import purged_spell_blocks
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing
from starter.load_smart_tagging import load as load_tags


def pooled_prior(raw, whole_episode=False):
    activity = raw[["h__speed_median", "a__speed_median"]].mean(axis=1).rolling(
        26, center=True, min_periods=26).mean().to_numpy()
    inactive = np.isfinite(activity) & (activity < 1.5)
    # Bridge activity blips up to one second only if enclosed by inactivity.
    gaps = np.flatnonzero(np.diff(np.r_[False, ~inactive, False]))
    for start, stop in zip(gaps[::2], gaps[1::2]):
        if start > 0 and stop < len(raw) and stop-start <= 25:
            inactive[start:stop] = True
    bounds = np.flatnonzero(np.diff(np.r_[False, inactive, False]))
    nearest = raw[["h__distance_1", "a__distance_1"]].min(axis=1).to_numpy()
    speed = raw.ball_displacement_1s.to_numpy()
    prior = np.zeros(len(raw))
    episodes = []
    for start, stop in zip(bounds[::2], bounds[1::2]):
        if stop-start < 75:
            continue
        # Bound evidence radius; partition long episodes into <=8-second blocks.
        for left in ([start] if whole_episode else range(start, stop, 200)):
            right = stop if whole_episode else min(left+200, stop)
            if right-left < 75:
                left = max(start, right-75)
            visible = np.isfinite(nearest[left:right]) & np.isfinite(speed[left:right])
            if visible.sum() < max(25, (right-left)//2):
                continue
            far = float(np.mean(nearest[left:right][visible] > 2))
            slow = float(np.mean(speed[left:right][visible] < 4.5))
            qualifies = far > .5 and slow > .5
            if qualifies:
                prior[left:right] = .8
            episodes.append(dict(start=int(left), end=int(right-1), far_fraction=far,
                                 slow_fraction=slow, qualifies=qualifies))
    return prior, episodes


def goal_kick_prior(raw, events):
    prior = np.zeros(len(raw))
    ball_slow = raw.ball_displacement_1s.rolling(26, center=True, min_periods=13).median() < 4.5
    goal_area = (raw.ball_goal_line_distance <= 6) & (raw.ball_touchline_distance >= 24)
    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces" or \
                (event["tags"].get("Type") or "").strip().lower() != "goal kick":
            continue
        team = (event["tags"].get("Team") or "").strip().lower()
        opponent = "a__" if team == "barcelona" else "h__" if team == "atletic club" else None
        if opponent is None:
            continue
        start, stop = max(0, event["start_frame"]), min(len(raw), event["end_frame"])
        if stop <= start:
            continue
        mask = (ball_slow & goal_area & (raw[opponent+"distance_1"] >= 8)).iloc[start:stop].to_numpy()
        prior[start:stop] = np.maximum(prior[start:stop], np.where(mask, .9, 0))
    return prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole-episodes", action="store_true")
    args = parser.parse_args()
    output = ROOT/("data/episode_stoppage_research" if args.whole_episodes else "data/pooled_stoppage_research")
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    prior, episodes = pooled_prior(raw, args.whole_episodes)
    goal_kick = goal_kick_prior(raw, load_tags())
    pd.DataFrame(episodes).to_csv(output/"input_only_episodes.csv", index=False)
    support = control_support(raw)[public]
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        parent = apply_context(np.load(ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy"),
                               gap_context(raw)[public], np.zeros(len(public)), False)
        ph = parent[:, 0]/parent[:, :2].sum(axis=1)
        dead = np.maximum(parent[:, 2], prior[public])
        probabilities = {"pooled": np.column_stack(((1-dead)*ph, (1-dead)*(1-ph), dead))}
        if args.whole_episodes:
            dead = np.maximum(dead, goal_kick[public])
            probabilities["pooled_and_goal_kick"] = np.column_stack(((1-dead)*ph, (1-dead)*(1-ph), dead))
        else:
            mono = np.load(ROOT/"data/monotone_research"/protocol/"monotone_class_head.npy")
            dead = np.maximum(dead, mono)
            probabilities["pooled_and_monotone"] = np.column_stack(((1-dead)*ph, (1-dead)*(1-ph), dead))
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, p in probabilities.items():
            for adaptive in (False, True):
                for dead_penalty in (3, 25):
                    method = name+("_adaptive" if adaptive else "_fixed")+f"_dead{dead_penalty}"
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
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
