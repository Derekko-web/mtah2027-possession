"""Test bounded complementary ownership evidence with the current D posterior."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from control_refinement import compose, contact_support, smooth
from evaluate_physics_context import apply_context
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES


def short_island_team(slow, fast, support, maximum=125):
    """Use fast ownership only for bounded bouts with repeated close control."""
    result = slow.copy()
    winner = fast.argmax(axis=1)
    edges = np.r_[0, np.flatnonzero(winner[1:] != winner[:-1])+1, len(winner)]
    for start, end in zip(edges[:-1], edges[1:]):
        team = winner[start]
        if not 8 <= end-start <= maximum:
            continue
        if fast[start:end, team].mean() < .65 or support[start:end, team].sum() < 3:
            continue
        # A fast short bout has to be enclosed by the same opposite slow owner.
        if start == 0 or end == len(slow) or slow[max(0,start-13):start].mean(axis=0).argmax() != 1-team:
            continue
        if slow[end:min(len(slow),end+13)].mean(axis=0).argmax() != 1-team:
            continue
        result[start:end] = fast[start:end]
    return result


def main():
    source = ROOT/"data/advanced_validation"
    output = ROOT/"data/control_blend_research"
    output.mkdir(exist_ok=True)
    raw = pd.read_pickle(source/"raw.pkl")
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    side = gap_sides(raw)[public]
    support = contact_support(raw)[public]
    splits = json.loads((source/"splits.json").read_text())
    rows, details, segments = [], [], []
    for protocol in ("buffered_minutes", "purged_spells", "buffered_100_seconds"):
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        original = np.load(source/protocol/"ordered_xgboost_frame_seed42_probabilities.npy")
        p = apply_context(original, side, np.zeros(7500), False)
        previous = np.load(source/protocol/"previous_ensemble_seed42_probabilities.npy")
        previous = apply_context(previous, side, np.zeros(7500), False)
        fast = np.load(ROOT/"data/control_refinement"/protocol/"fast_probabilities.npy")
        fast = apply_context(np.column_stack((fast, np.zeros(7500))), side, np.zeros(7500), False)[:, :2]
        fast /= fast.sum(axis=1, keepdims=True)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        predictions = {}
        for _, _, validation in folds:
            ix = validation-public[0]
            slow = smooth(p[ix], 25)
            old_team = slow[:, :2]/slow[:, :2].sum(axis=1, keepdims=True)
            methods = {"current": slow}
            for half_window in (5, 25):
                prev = smooth(previous[ix], half_window)
                prev = prev[:, :2]/prev[:, :2].sum(axis=1, keepdims=True)
                for mixture in (.25, .5, .75):
                    methods[f"previous_half{half_window}_mix{mixture}"] = compose(
                        slow[:, 2], mixture*prev+(1-mixture)*old_team)
                for threshold in (.65, .8):
                    gate = ((prev > threshold) & support[ix]).any(axis=1)
                    methods[f"previous_half{half_window}_gate{threshold}"] = compose(
                        slow[:, 2], np.where(gate[:, None], prev, old_team))
                methods[f"previous_half{half_window}_islands"] = compose(slow[:, 2],
                    short_island_team(old_team, prev, support[ix]))
            methods["fast_islands"] = compose(slow[:, 2], short_island_team(old_team, smooth(fast[ix], 5), support[ix]))
            for name, value in methods.items():
                predictions.setdefault(name, np.empty(7500, dtype="<U1"))
                predictions[name][ix] = CLASSES[decode(value)]
        for name, prediction in predictions.items():
            table[name] = prediction
            record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
        add_short_metrics(rows, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
        pd.DataFrame(rows).to_csv(output/"scores.csv", index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
