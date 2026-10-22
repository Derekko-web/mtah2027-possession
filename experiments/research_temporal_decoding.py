"""Check whether the history-aware model is oversmoothed by the old 51-frame mean."""

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
from xgboost_baseline import CLASSES


def smooth(probability, half_window):
    return pd.DataFrame(probability).rolling(2*half_window+1, center=True, min_periods=1).mean().to_numpy()


def main():
    output = ROOT/"data/temporal_decoding_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    support = control_support(raw)[public]
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        sources = {"original": ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy",
                   "ordered_without_location": ROOT/"data/temporal_control_research"/protocol/"ordered_without_location/oof_probabilities.npy"}
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, source in sources.items():
            p = apply_context(np.load(source), gap_context(raw)[public], np.zeros(len(public)), False)
            for half_window in (0, 5, 25):
                for adaptive in (False, True):
                    method = name+f"_half{half_window}"+("_adaptive" if adaptive else "_fixed")
                    prediction = np.empty(len(public), dtype="<U1")
                    for _, _, validation in blocks:
                        ix = validation-public[0]
                        probability = smooth(p[ix], half_window)
                        prediction[ix] = CLASSES[adaptive_decode(probability, support[ix]) if adaptive else decode(probability)]
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
