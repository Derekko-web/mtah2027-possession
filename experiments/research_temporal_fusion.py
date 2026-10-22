"""Compare the instantaneous and ordered, geography-free possession views."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from research_pooled_stoppages import pooled_prior, goal_kick_prior
from research_sequence import adaptive_decode, control_support
from research_stoppages import record_predictions
from run_continuation import purged_spell_blocks
from starter.load_smart_tagging import load as load_tags
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing


def compose(home, dead):
    return np.column_stack(((1-dead)*home, (1-dead)*(1-home), dead))


def main():
    output = ROOT/"data/temporal_fusion_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    support = control_support(raw)[public]
    prior, _ = pooled_prior(raw, whole_episode=True)
    restart = goal_kick_prior(raw, load_tags())
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        def adjusted(path):
            return apply_context(np.load(path), gap_context(raw)[public], np.zeros(len(public)), False)
        old = adjusted(ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy")
        new = adjusted(ROOT/"data/temporal_control_research"/protocol/"ordered_without_location/oof_probabilities.npy")
        old_team, new_team = old[:, 0]/old[:, :2].sum(axis=1), new[:, 0]/new[:, :2].sum(axis=1)
        probabilities = {"equal_views": (old+new)/2,
                         "old_team_new_dead": compose(old_team, new[:, 2]),
                         "equal_team_new_dead": compose((old_team+new_team)/2, new[:, 2]),
                         "new_team_old_dead": compose(new_team, old[:, 2])}
        for name in ("equal_views", "equal_team_new_dead"):
            p = probabilities[name]
            home = p[:, 0]/p[:, :2].sum(axis=1)
            dead = np.maximum(p[:, 2], prior[public])
            probabilities[name+"_pooled"] = compose(home, dead)
            dead = np.maximum(dead, restart[public])
            probabilities[name+"_pooled_restart"] = compose(home, dead)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, p in probabilities.items():
            for adaptive in (False, True):
                method = name+("_adaptive" if adaptive else "_fixed")
                prediction = np.empty(len(public), dtype="<U1")
                for _, _, validation in blocks:
                    ix = validation-public[0]
                    smooth = probability_smoothing(p[ix])
                    prediction[ix] = CLASSES[adaptive_decode(smooth, support[ix]) if adaptive else decode(smooth)]
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
