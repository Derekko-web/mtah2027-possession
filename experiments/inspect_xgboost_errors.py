"""Describe held-out failures without changing or retraining the model."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/xgboost_baseline"
COLORS = {"H": "#3178b5", "A": "#e5a13d", "D": "#c5cbd1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(DATA / "validation_predictions.csv")
    if not np.all(np.diff(table.frame) == 1):
        raise ValueError("Expected a contiguous held-out block")
    fps = 25
    start = int(table.frame.iloc[0])
    missing = table[["nearest_home_m", "nearest_away_m"]].isna().all(axis=1)
    gt_groups = table.ground_truth.ne(table.ground_truth.shift()).cumsum()
    rows = []
    for _, block in table.groupby(gt_groups):
        rows.append(dict(
            first_frame=int(block.frame.iloc[0]), last_frame=int(block.frame.iloc[-1]),
            seconds=len(block) / fps, truth=block.ground_truth.iloc[0],
            correct_smoothed=int(block.xgboost_smooth25.eq(block.ground_truth).sum()),
            missing_both_distances=int(missing.loc[block.index].sum()),
            predicted_H=int(block.xgboost_smooth25.eq("H").sum()),
            predicted_A=int(block.xgboost_smooth25.eq("A").sum()),
            predicted_D=int(block.xgboost_smooth25.eq("D").sum()),
        ))
    segments = pd.DataFrame(rows)
    segments.to_csv(DATA / "error_segments.csv", index=False)
    for variant in ["xgboost_raw", "xgboost_smooth25"]:
        confusion = pd.crosstab(table.ground_truth, table[variant]).reindex(
            index=["H", "A", "D"], columns=["H", "A", "D"], fill_value=0,
        )
        confusion.to_csv(DATA / f"{variant}_confusion.csv")

    fig, axes = plt.subplots(5, 1, figsize=(13, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, .65, 1.8]})
    for ax, column, label in zip(axes[:3],
                                ["ground_truth", "xgboost_raw", "xgboost_smooth25"],
                                ["Answer key", "XGBoost raw", "XGBoost smoothed"]):
        groups = table[column].ne(table[column].shift()).cumsum()
        for _, block in table.groupby(groups):
            left = (int(block.frame.iloc[0]) - start) / fps
            right = (int(block.frame.iloc[-1]) + 1 - start) / fps
            state = block[column].iloc[0]
            ax.axvspan(left, right, color=COLORS[state], linewidth=0)
            if right - left > 2:
                ax.text((left + right) / 2, .5, state, ha="center", va="center",
                        transform=ax.get_xaxis_transform(), fontsize=10)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=12)
        ax.set_yticks([])
    axes[3].imshow(missing.to_numpy()[None, :], aspect="auto", cmap="Greys",
                   vmin=0, vmax=1, extent=(0, len(table) / fps, 0, 1), interpolation="nearest")
    axes[3].set_ylabel("Inputs missing", rotation=0, ha="right", va="center", labelpad=12)
    axes[3].set_yticks([])
    seconds = (table.frame.to_numpy() - start) / fps
    axes[4].plot(seconds, table.probability_D, color="#545b62", linewidth=1.3,
                 label="Raw predicted probability of D")
    axes[4].set_ylim(-.03, 1.03)
    axes[4].set_ylabel("P(stopped play)")
    axes[4].set_xlabel(f"Seconds into held-out stretch (starts at video frame {start:,})")
    axes[4].grid(axis="y", alpha=.2)
    axes[4].legend(loc="upper right", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlim(0, len(table) / fps)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Where our two-distance model gets possession wrong", fontsize=17, x=.04, ha="left")
    fig.legend(handles=[Patch(color=COLORS[k], label=v) for k, v in
                        [("H", "H: Barcelona"), ("A", "A: Athletic"), ("D", "D: stopped play")]],
               loc="upper right", bbox_to_anchor=(.98, 1), ncol=3, frameon=False)
    fig.text(.04, .015, "Black input bars = both distances missing. "
             "Smoothed predictions average probabilities over +/- 1 second. No retraining.", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .92))
    fig.savefig(args.out / "xgboost-error-timeline.png", dpi=160)
    plt.close(fig)
    print(segments.to_string(index=False))
    for frame in [16150, 16650, 18100]:
        print("\nExample:", table.loc[table.frame == frame].to_dict(orient="records")[0])
    print("\nChart:", args.out / "xgboost-error-timeline.png")


if __name__ == "__main__":
    main()
