"""Official-score the frozen comparisons and record the declared promotion checks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from research_stoppages import record_predictions
from validate_temporal_candidate import PLAN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"data/advanced_validation")
    args = parser.parse_args()
    output = args.out.resolve()
    if (output/"plan.json").read_bytes() != PLAN.read_bytes():
        raise ValueError("Validation plan changed after the run")
    scores = pd.read_csv(output/"scores.csv")
    folds = pd.read_csv(output/"fold_scores.csv")
    if len(scores) != 27 or scores.duplicated(["protocol", "model"]).any():
        raise ValueError("Expected exactly three methods, three seeds and three protocols")
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    split_rows = json.loads((output/"splits.json").read_text())
    reference_rows, reference_folds, reference_segments = [], [], []
    official_rows = []
    for protocol in scores.protocol.unique():
        blocks = [(s["block"], np.array(s["train"]), np.array(s["validation"]))
                  for s in split_rows if s["protocol"] == protocol]
        for reference in ("smooth50", "smooth125"):
            prediction = pm.read_label_csv(ROOT/f"evaluation/results/baseline_{reference}.csv").label.to_numpy()[public]
            record_predictions(output, protocol, reference, public, labels, blocks, prediction,
                               reference_rows, reference_folds, reference_segments)
        with (output/protocol/"official.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT/"evaluation/run_evaluation.py"),
                            "--gt", "public", "--results", str(output/protocol/"scorer_inputs"),
                            "--out", str(output/protocol/"official"), "--bootstrap", "1000", "--no-detail"],
                           check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        table = pd.concat([scores[scores.protocol == protocol],
                           pd.DataFrame(reference_rows).query("protocol == @protocol")])
        for score in table.to_dict("records"):
            report = json.loads((output/protocol/"official/reports"/(score["model"]+".json")).read_text())
            for source, key in (("spellF1", "spell_f1_mean"), ("macroF1", "macro_f1")):
                if abs(score[source]-report[key]) > 1e-12:
                    raise ValueError("Official scorer disagrees: "+score["model"])
            official_rows.append(dict(protocol=protocol, model=score["model"],
                                      bootstrap90_low=report["spell_bootstrap_low"],
                                      bootstrap90_high=report["spell_bootstrap_high"],
                                      matched_iou50=report["spell_f1"]["0.5"]["n_matched"],
                                      spurious_iou50=report["spell_f1"]["0.5"]["n_spurious"],
                                      missed_iou50=report["spell_f1"]["0.5"]["n_missed"]))
    pd.DataFrame(reference_rows).to_csv(output/"reference_scores.csv", index=False)
    pd.DataFrame(reference_folds).to_csv(output/"reference_fold_scores.csv", index=False)
    official = pd.DataFrame(official_rows)
    official.to_csv(output/"official_summary.csv", index=False)
    scores = scores.merge(official, on=["protocol", "model"], validate="one_to_one")
    comparison = scores[scores.model.str.endswith("seed42")]
    comparison = pd.concat([comparison, pd.DataFrame(reference_rows).merge(official, on=["protocol", "model"])])
    comparison.to_csv(output/"comparison.csv", index=False)
    seed_summary = scores.assign(method=scores.model.str.replace(r"_seed\d+$", "", regex=True)).groupby(
        ["protocol", "method"])[["spellF1", "macroF1", "mean_minute_spellF1"]].agg(["min", "mean", "max", "std"])
    seed_summary.to_csv(output/"seed_summary.csv")
    decisions = {}
    live_frames = int(np.count_nonzero(labels[public] != "D"))
    for method in ("ordered_catboost_frame", "ordered_xgboost_frame"):
        checks = {}
        for protocol in scores.protocol.unique():
            table = scores[scores.protocol == protocol].set_index("model")
            old, new = table.loc["previous_ensemble_seed42"], table.loc[method+"_seed42"]
            checks[protocol] = dict(
                stitched_gain=bool(new.spellF1 > old.spellF1),
                mean_block_gain=bool(new.mean_minute_spellF1 > old.mean_minute_spellF1),
                macroF1=bool(new.macroF1 >= old.macroF1),
                false_dead_rate=bool(new.false_dead_frames/live_frames <= .05))
            if protocol != "buffered_100_seconds":
                checks[protocol]["fewer_missed_stoppages"] = bool(new.missed_stoppages < old.missed_stoppages)
            a = folds[(folds.protocol == protocol) & (folds.model == method+"_seed42")].set_index("minute")
            b = folds[(folds.protocol == protocol) & (folds.model == "previous_ensemble_seed42")].set_index("minute")
            delta = a.spellF1-b.spellF1
            checks[protocol]["block_deltas"] = {int(i): float(v) for i,v in delta.items()}
        decisions[method] = dict(passed=all(all(v for k,v in check.items() if k != "block_deltas")
                                                        for check in checks.values()), checks=checks)
    (output/"promotion_checks.json").write_text(json.dumps(decisions, indent=2)+"\n")
    # Inspect all paired seeds instead of picking a lucky seed. This descriptive
    # robustness assessment is separate from the predeclared seed-42 gates.
    robustness = {}
    for method in ("ordered_catboost_frame", "ordered_xgboost_frame"):
        paired = []
        for protocol in scores.protocol.unique():
            table = scores[scores.protocol == protocol].set_index("model")
            for seed in (42, 43, 44):
                old = table.loc[f"previous_ensemble_seed{seed}"]
                new = table.loc[f"{method}_seed{seed}"]
                paired.append(dict(protocol=protocol, seed=seed,
                                   spellF1_delta=float(new.spellF1-old.spellF1),
                                   mean_block_delta=float(new.mean_minute_spellF1-old.mean_minute_spellF1),
                                   macroF1_delta=float(new.macroF1-old.macroF1)))
        robustness[method] = dict(all_paired_gains=all(min(r["spellF1_delta"], r["mean_block_delta"],
                                                         r["macroF1_delta"]) > 0 for r in paired), paired=paired)
    selected = "ordered_xgboost_frame"
    if not decisions[selected]["passed"] or not robustness[selected]["all_paired_gains"]:
        raise ValueError("The recommended finalist no longer supports the recorded selection")
    (output/"selection_decision.json").write_text(json.dumps(dict(
        selected=selected, seed=42, initial_primary="ordered_catboost_frame",
        rationale="Prefer consistent gains across all fixed seeds/protocols over CatBoost's slightly higher headline score. No model settings or seeds were tuned to the longer-block results.",
        limitation="This is an adaptive development decision after examining stress results, not a predeclared independent test.",
        robustness=robustness), indent=2)+"\n")
    print(comparison.to_string(index=False))
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
