"""Inspect public OOF sequences, without fitting or accessing restricted labels."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm


def sequence_details(frames, truth, prediction):
    """Return public label-run diagnostics; interval ends are inclusive in CSVs."""
    frames = np.asarray(frames)
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if not (len(frames) == len(truth) == len(prediction)):
        raise ValueError("Frame, truth and prediction lengths differ")
    if len(frames) == 0 or not np.all(np.diff(frames) == 1):
        raise ValueError("Diagnostics require a nonempty contiguous window")
    tc, pc = pm.label_codes(truth), pm.label_codes(prediction)
    if (truth == "X").any() or (prediction == "X").any():
        raise ValueError("Diagnostics require explicit H/A/D labels")
    runs = pm.valid_runs(tc)
    rows = []
    for start, stop, code in pm.segments(tc, runs):
        label = pm.SCORED[code]
        piece = prediction[start:stop]
        rows.append(dict(
            start=int(frames[start]), end=int(frames[stop - 1]), label=label,
            frames=stop-start, recall=float(np.mean(piece == label)),
            predicted_runs=1 + int(np.count_nonzero(piece[1:] != piece[:-1])),
            predicted_dead_frames=int(np.count_nonzero(piece == "D")),
        ))
    result = pm.evaluate(tc, pc, bootstrap_n=0)
    dead = [row for row in rows if row["label"] == "D"]
    summary = dict(
        spellF1=result["spell_f1_mean"], macroF1=result["macro_f1"],
        predicted_possessions=result["n_pred_spells"],
        true_possessions=result["n_gt_spells"],
        dead_precision=result["per_class"]["D"]["precision"],
        dead_recall=result["per_class"]["D"]["recall"],
        dead_F1=result["per_class"]["D"]["f1"],
        false_dead_frames=int(np.count_nonzero((truth != "D") & (prediction == "D"))),
        missed_stoppages=sum(row["recall"] < .5 for row in dead),
        live_run_fragments=sum(row["predicted_runs"] - 1 for row in rows if row["label"] != "D"),
    )
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "data/selected_evaluation/oof_predictions.csv")
    parser.add_argument("--column", default="selected_ensemble")
    parser.add_argument("--out", type=Path, default=ROOT / "data/continuation_20261021/diagnosis")
    parser.add_argument("--require-stoppages", action="store_true",
                        help="fail if any true stoppage has less than 50%% D recall")
    args = parser.parse_args()
    table = pd.read_csv(args.predictions)
    summary, rows = sequence_details(table.frame, table.ground_truth, table[args.column])
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "true_segments.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.require_stoppages and summary["missed_stoppages"]:
        raise SystemExit("FAIL: at least one true stoppage has less than 50% D recall")


if __name__ == "__main__":
    main()
