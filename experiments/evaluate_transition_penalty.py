"""Evaluate one fixed switch penalty against existing chronological predictions.

Run: .venv/bin/python experiments/evaluate_transition_penalty.py
No models are trained and no ground-truth labels enter decoding.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from transition_penalty import SWITCH_PENALTY, decode
from xgboost_baseline import probability_smoothing

CLASSES = np.array(["H", "A", "D"])
GROUPS = ("distances", "plus_ball_speed", "plus_player_motion")


def switches(labels):
    return int(np.count_nonzero(labels[1:] != labels[:-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--buffered-blocks", action="store_true",
                        help="evaluate the five buffered offline validation blocks")
    args = parser.parse_args()
    source = "buffered_block_comparison" if args.buffered_blocks else "time_block_comparison"
    output_name = "buffered_transition_penalty" if args.buffered_blocks else "transition_penalty"
    output = ROOT / "data" / output_name
    if args.out is None:
        args.out = output
    args.out.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=True)
    scores = []
    minutes = (1, 2, 3, 4, 5) if args.buffered_blocks else (3, 4, 5)
    for minute in minutes:
        data = pd.read_csv(ROOT / f"data/{source}/minute_{minute}/validation_predictions.csv")
        if not np.all(np.diff(data.frame) == 1):
            raise ValueError("Decoder must receive one contiguous block")
        truth = pm.label_codes(data.ground_truth.to_numpy())
        for group in GROUPS:
            probabilities = data[[f"{group}_probability_{label}" for label in CLASSES]].to_numpy()
            decoded = CLASSES[decode(probabilities)]
            np.testing.assert_array_equal(CLASSES[probabilities.argmax(axis=1)], data[f"{group}_raw"])
            if switches(decoded) > switches(data[f"{group}_raw"].to_numpy()):
                raise ValueError("Nonnegative switch penalty unexpectedly increased raw switches")
            data[f"{group}_decoded"] = decoded
            data[f"{group}_smoothed_decoded"] = CLASSES[decode(probability_smoothing(probabilities))]
            if switches(data[f"{group}_smoothed_decoded"].to_numpy()) > switches(
                    data[f"{group}_smoothed"].to_numpy()):
                raise ValueError("Switch penalty increased switches relative to smoothing")
            for method in ("raw", "smoothed", "decoded", "smoothed_decoded"):
                prediction = data[f"{group}_{method}"].to_numpy()
                result = pm.evaluate(truth, pm.label_codes(prediction), bootstrap_n=1000)
                scores.append(dict(
                    validation_minute=minute, features=group, method=method,
                    spellF1=result["spell_f1_mean"], macroF1=result["macro_f1"],
                    switches=switches(prediction), true_switches=switches(data.ground_truth.to_numpy()),
                    predicted_possessions=result["n_pred_spells"], true_possessions=result["n_gt_spells"],
                    dead_ball_F1=result["per_class"]["D"]["f1"],
                    spellF1_low=result["spell_bootstrap_low"], spellF1_high=result["spell_bootstrap_high"],
                ))
        # Preserve source predictions; only write to a separate ignored experiment folder.
        data.to_csv(output / f"minute_{minute}_predictions.csv", index=False)
        if minute == 5:
            example = data
    scores = pd.DataFrame(scores)
    scores.to_csv(output / "validation_scores.csv", index=False)
    scores.to_csv(args.out / "transition-penalty-scores.csv", index=False)
    summary = scores.pivot(index=["features", "method"], columns="validation_minute", values="spellF1")
    mean_column = "mean_of_five_minutes" if args.buffered_blocks else "mean_of_three_minutes"
    summary[mean_column] = summary.mean(axis=1)
    summary.to_csv(output / "spellF1_comparison.csv")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nMinute 5 counts:")
    print(scores[scores.validation_minute == 5][[
        "features", "method", "switches", "true_switches", "predicted_possessions", "true_possessions",
    ]].to_string(index=False))
    manifest = dict(
        switch_penalty=SWITCH_PENALTY,
        objective="sum(-log(probability[state])) + penalty * number_of_state_changes",
        source=f"data/{source}", model_retrained=False, parameter_search=False,
        experiment_sequence=[
            "First evaluated penalty on raw probabilities against raw and smoothing references.",
            "After seeing raw decoding trail smoothing, evaluated the same fixed penalty "
            "on smoothed probabilities. Penalty strength was not changed.",
        ],
        caveats=[
            "Offline decoding uses future input probabilities within each validation block.",
            "All H/A/D transitions have equal cost; none is prohibited.",
            "No minimum possession duration or fixed boundary delay is imposed.",
            "Penalizing switches can erase real short possessions or stoppages.",
            "These same-match development blocks have already informed feature exploration.",
            "Frame probabilities are correlated and imperfectly calibrated; this is a heuristic score.",
            "Each minute has four scored possessions and truncates boundary possessions.",
        ],
    )
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Illustrate the previously identified over-fragmented five-feature model.
    colors = {"H": "#3178b5", "A": "#e5a13d", "D": "#c5cbd1"}
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 4.8))
    start = int(example.frame.iloc[0])
    for ax, column, label in zip(axes,
        ["ground_truth", "plus_player_motion_raw", "plus_player_motion_smoothed", "plus_player_motion_smoothed_decoded"],
        ["Answer key", "Raw XGBoost", "Smoothing", "Smoothing + penalty"]):
        blocks = example[column].ne(example[column].shift()).cumsum()
        for _, block in example.groupby(blocks):
            left = (int(block.frame.iloc[0]) - start) / 25
            right = (int(block.frame.iloc[-1]) + 1 - start) / 25
            state = block[column].iloc[0]
            ax.axvspan(left, right, color=colors[state], linewidth=0)
            if right - left > 1.8:
                ax.text((left + right) / 2, .5, state, ha="center", va="center",
                        transform=ax.get_xaxis_transform())
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=12)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlim(0, 60)
    axes[-1].set_xlabel(f"Seconds into public minute 5 (video frame {start:,} onward)")
    fig.suptitle("Can a change penalty remove unsupported possession flips?", fontsize=16, x=.03, ha="left")
    fig.legend(handles=[Patch(color=colors[k], label=v) for k, v in
                        [("H", "Barcelona"), ("A", "Athletic"), ("D", "Stopped play")]],
               loc="upper right", bbox_to_anchor=(.98, .92), ncol=3, frameon=False)
    fig.text(.03, .02, "Same five-feature XGBoost probabilities in all three prediction rows. "
             "One fixed penalty; no retraining or parameter search.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .86))
    fig.savefig(args.out / "transition-penalty-timeline.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
