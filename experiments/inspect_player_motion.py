"""Inspect player-motion clues using existing public labels; do not train a model."""

import argparse
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
from games import GAME, GAMES, require
from starter.baseline import team_of_each_track

MOVEMENT_THRESHOLD_M_S = 1.0  # Descriptive definition fixed before inspecting results.
SPEED = "median_player_speed_past_second_m_s"
MOVING = "moving_fraction_past_second"


def read_motion(last_frame):
    metadata, tracking = require(["atd_meta", "atd_raw"])
    slots = len(team_of_each_track(metadata))
    median_speed = np.full(last_frame + 1, np.nan)
    moving_fraction = np.full(last_frame + 1, np.nan)
    observed = np.zeros(last_frame + 1, dtype=int)
    fps = GAMES[GAME]["fps"]
    count = 0
    with open(tracking) as handle:
        for count, line in enumerate(handle, start=1):
            frame, raw_players, _ = line.strip().split(":")
            if int(frame) != count:
                raise ValueError("Tracking must be contiguous and one-based")
            players = np.fromstring(raw_players.replace(";", ","), sep=",")
            if players.size != slots * 3:
                raise ValueError(f"Unexpected player layout at tracking frame {frame}")
            players = players.reshape(slots, 3)
            # Supplied player speeds are m/s; do not difference identity-changing slots.
            valid = np.isfinite(players).all(axis=1) & (players[:, 2] >= 0)
            speed = players[valid, 2]
            observed[count - 1] = len(speed)
            if len(speed):
                median_speed[count - 1] = np.median(speed)
                moving_fraction[count - 1] = (speed > MOVEMENT_THRESHOLD_M_S).mean()
            if count == last_frame + 1:
                break
    if count != last_frame + 1:
        raise ValueError("Tracking ended before the public window")
    # Both windows are trailing and use inputs only, with at least 13 valid frames.
    return pd.DataFrame({
        "frame": np.arange(last_frame + 1),
        SPEED: pd.Series(median_speed).rolling(fps, min_periods=fps // 2 + 1).median(),
        MOVING: pd.Series(moving_fraction).rolling(fps, min_periods=fps // 2 + 1).mean(),
        "observed_speed_tracks": observed,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/player_motion")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    private_output = ROOT / "data/player_motion"
    private_output.mkdir(exist_ok=True)
    gt = pd.read_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    public = gt.loc[gt.label != "X"]
    motion = read_motion(int(public.frame.max()))
    table = public.merge(motion, on="frame", validate="one_to_one")
    table.to_csv(private_output / "public_motion.csv", index=False)
    validation_frames = pd.read_csv(
        ROOT / "data/xgboost_with_ball_speed/validation_predictions.csv", usecols=["frame"],
    )
    validation = validation_frames.merge(table, on="frame", validate="one_to_one")
    groups = validation.label.ne(validation.label.shift()).cumsum()
    summaries = []
    for _, block in validation.groupby(groups):
        summaries.append(dict(
            first_frame=int(block.frame.iloc[0]), last_frame=int(block.frame.iloc[-1]),
            label=block.label.iloc[0], seconds=len(block) / 25,
            median_speed_m_s=block[SPEED].median(),
            median_moving_fraction=block[MOVING].median(),
            median_observed_tracks=block.observed_speed_tracks.median(),
        ))
    summary = pd.DataFrame(summaries)
    summary.to_csv(private_output / "validation_segments.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for frame in [16150, 16650, 18100, 18420, 18500]:
        print("\nExample:", table.loc[table.frame == frame].to_dict(orient="records")[0])
    for name, block in [("training", table.loc[table.frame <= 15684]),
                        ("validation", validation)]:
        print(f"\n{name}: median measurements by ground-truth class")
        print(block.groupby("label")[[SPEED, MOVING]].median().to_string())
    print("\nMeasurement availability:", table[[SPEED, MOVING]].notna().mean().to_dict())

    start = int(validation.frame.iloc[0])
    fps = GAMES[GAME]["fps"]
    seconds = (validation.frame - start) / fps
    colors = {"H": "#3178b5", "A": "#e5a13d", "D": "#c5cbd1"}
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 6),
                             gridspec_kw={"height_ratios": [.5, 1, 1]})
    for row in summaries:
        left = (row["first_frame"] - start) / fps
        right = (row["last_frame"] + 1 - start) / fps
        label = row["label"]
        axes[0].axvspan(left, right, color=colors[label], linewidth=0)
        axes[0].text((left + right) / 2, .5, label, ha="center", va="center",
                     transform=axes[0].get_xaxis_transform())
        if label == "D":
            for ax in axes[1:]:
                ax.axvspan(left, right, color=colors["D"], alpha=.3, linewidth=0)
    axes[0].set_ylabel("Answer key")
    axes[0].set_yticks([])
    axes[1].plot(seconds, validation[SPEED], color="#245b83", linewidth=1.5)
    axes[1].set_ylabel("Median player speed\n(metres/second)")
    axes[1].set_ylim(bottom=0)
    axes[2].plot(seconds, validation[MOVING] * 100, color="#5d4e86", linewidth=1.5)
    axes[2].set_ylabel("Tracked players moving\nabove 1 m/s (%)")
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel(f"Seconds into held-out stretch (starts at video frame {start:,})")
    for ax in axes:
        ax.set_xlim(0, len(validation) / fps)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Does player movement reveal when play stops?", fontsize=17, x=.04, ha="left")
    fig.legend(handles=[Patch(color=colors[k], label=v) for k, v in
                        [("H", "Barcelona"), ("A", "Athletic"), ("D", "Stopped play")]],
               loc="upper right", bbox_to_anchor=(.98, 1), ncol=3, frameon=False)
    fig.text(.04, .015, "Motion summaries use the preceding second only. "
             "Shaded regions are stopped play in the answer key. No model retraining.", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .92))
    fig.savefig(args.out / "player-motion-inspection.png", dpi=160)
    plt.close(fig)
    print("\nChart:", args.out / "player-motion-inspection.png")


if __name__ == "__main__":
    main()
