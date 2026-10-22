"""Fast red-capable interval-precision diagnostic; never fits or edits predictions."""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT/"data/control_renewal_v1/recovery_confirmation/scores.csv")
    parser.add_argument("--protocol", default="buffered_minutes")
    parser.add_argument("--model", default="stoppage_consensus_seed42")
    parser.add_argument("--minimum", type=int, default=4)
    args = parser.parse_args()
    table = pd.read_csv(args.scores)
    rows = table[(table.protocol == args.protocol) & (table.model == args.model)]
    if len(rows) != 1:
        raise ValueError("Expected exactly one protocol/model score")
    matched = int(rows.iloc[0].short_live_matched_iou50)
    print(f"Short IoU>=.5 matches: {matched}; required: {args.minimum}")
    if matched < args.minimum:
        raise SystemExit("FAIL: short possession precision remains insufficient")


if __name__ == "__main__":
    main()
