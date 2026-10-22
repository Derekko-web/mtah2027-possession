"""Enforce the frozen paired gates and verify official scores/shared inference."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from boundary_refinement import decode_refinement


def promotion_checks(scores, segments, plan):
    index = scores.set_index(["protocol", "model"])
    expected = len(plan["protocols"])*len(plan["seeds"])*len(plan["methods"])
    if len(scores) != expected or index.index.has_duplicates:
        raise ValueError("Incomplete or duplicate confirmation scores")
    checks = {}
    for method in plan["methods"]:
        if method == "current":
            continue
        paired = []
        for protocol in plan["protocols"]:
            for seed in plan["seeds"]:
                old_name, new_name = f"current_seed{seed}", f"{method}_seed{seed}"
                old, new = index.loc[(protocol, old_name)], index.loc[(protocol, new_name)]
                short = segments[(segments.protocol == protocol) & (segments.label != "D") & (segments.frames <= 125)]
                before = short[short.model == old_name].set_index(["start", "end", "label"])
                after = short[short.model == new_name].set_index(["start", "end", "label"])
                pd.testing.assert_index_equal(before.index, after.index)
                flags = dict(spellF1=bool(new.spellF1 >= old.spellF1-1e-12),
                             dead_recall=bool(new.dead_recall >= old.dead_recall-.02),
                             missed_stoppages=bool(new.missed_stoppages <= old.missed_stoppages),
                             macroF1=bool(new.macroF1 >= old.macroF1-.005),
                             prior_short_recovery=bool((after.recall[before.recall >= .5] >= .5).all()))
                if protocol == "buffered_minutes":
                    flags.update(primary_gain=bool(new.spellF1 >= old.spellF1+.01-1e-12),
                                 false_dead=bool(new.false_dead_frames < old.false_dead_frames),
                                 short_iou=bool(new.short_live_mean_best_iou > old.short_live_mean_best_iou))
                paired.append(dict(protocol=protocol, seed=seed, checks=flags,
                                   spellF1_delta=float(new.spellF1-old.spellF1),
                                   false_dead_delta=int(new.false_dead_frames-old.false_dead_frames),
                                   dead_recall_delta=float(new.dead_recall-old.dead_recall)))
        checks[method] = dict(passed=all(all(p["checks"].values()) for p in paired), paired=paired)
    eligible = [name for name, result in checks.items() if result["passed"]]
    # Frozen rule: false D first, short IoU second, using the buffered seeds.
    def ranking(name):
        subset = scores[(scores.protocol == "buffered_minutes") & scores.model.isin(
            [f"{name}_seed{seed}" for seed in plan["seeds"]])]
        return (subset.false_dead_frames.mean(), -subset.short_live_mean_best_iou.mean(), name)
    return dict(selected=min(eligible, key=ranking) if eligible else None, methods=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/control_boundary_v1/confirmation")
    parser.add_argument("--screen", type=Path, default=ROOT/"data/control_boundary_v1/screen")
    args = parser.parse_args()
    output = args.output.resolve()
    plan_path = ROOT/"experiments/BOUNDARY_CONFIRMATION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    if (output/"plan.json").read_bytes() != plan_path.read_bytes():
        raise ValueError("Plan changed after confirmation")
    for path, digest in plan["source_sha256"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source changed: "+path)
    source, confirmed = ROOT/"data/advanced_validation", ROOT/"data/control_boundary_v1/baseline_confirmation"
    raw = pd.read_pickle(confirmed/"raw.pkl")
    public = np.arange(11185, 18685)
    splits = json.loads((source/"splits.json").read_text())
    scores = pd.read_csv(output/"scores.csv")
    segments = pd.read_csv(output/"segment_diagnostics.csv")
    result = promotion_checks(scores, segments, plan)
    official_rows = []
    for protocol in plan["protocols"]:
        blocks = [(min(s["validation"]), max(s["validation"])+1) for s in splits if s["protocol"] == protocol]
        table = pd.read_csv(output/protocol/"oof_predictions.csv")
        screen = pd.read_csv(args.screen/protocol/"oof_predictions.csv")
        for seed in plan["seeds"]:
            arrays = []
            for path, width in ((source/protocol/f"ordered_xgboost_frame_seed{seed}_probabilities.npy", 3),
                                (source/protocol/f"previous_ensemble_seed{seed}_probabilities.npy", 3),
                                (confirmed/protocol/f"fast_seed{seed}_probabilities.npy", 2)):
                full = np.full((len(raw), width), np.nan)
                full[public] = np.load(path)
                arrays.append(full)
            for method in plan["methods"]:
                name = f"{method}_seed{seed}"
                prediction = decode_refinement(*arrays, raw, blocks, method)[public]
                np.testing.assert_array_equal(prediction, table[name].to_numpy())
                if seed == 42:
                    np.testing.assert_array_equal(prediction, screen[name].to_numpy())
        with (output/protocol/"official.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT/"evaluation/run_evaluation.py"), "--gt", "public",
                            "--results", str(output/protocol/"scorer_inputs"),
                            "--out", str(output/protocol/"official"), "--bootstrap", "1000", "--no-detail"],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        for row in scores[scores.protocol == protocol].to_dict("records"):
            report = json.loads((output/protocol/"official/reports"/(row["model"]+".json")).read_text())
            if abs(report["spell_f1_mean"]-row["spellF1"]) > 1e-12 or abs(report["macro_f1"]-row["macroF1"]) > 1e-12:
                raise ValueError("Official scorer mismatch")
            official_rows.append(dict(protocol=protocol, model=row["model"],
                                      bootstrap90_low=report["spell_bootstrap_low"],
                                      bootstrap90_high=report["spell_bootstrap_high"],
                                      matched_iou50=report["spell_f1"]["0.5"]["n_matched"],
                                      spurious_iou50=report["spell_f1"]["0.5"]["n_spurious"],
                                      missed_iou50=report["spell_f1"]["0.5"]["n_missed"]))
    scores.merge(pd.DataFrame(official_rows), on=["protocol", "model"], validate="one_to_one").to_csv(
        output/"official_comparison.csv", index=False)
    folds = pd.read_csv(output/"fold_scores.csv")
    deltas = []
    for method in plan["methods"]:
        if method == "current":
            continue
        for seed in plan["seeds"]:
            a = folds[folds.model == f"current_seed{seed}"]
            b = folds[folds.model == f"{method}_seed{seed}"]
            paired = a.merge(b, on=["protocol", "minute"], suffixes=["_before", "_after"], validate="one_to_one")
            for row in paired.to_dict("records"):
                deltas.append(dict(protocol=row["protocol"], seed=seed, method=method, block=row["minute"],
                                   spellF1_before=row["spellF1_before"], spellF1_after=row["spellF1_after"],
                                   delta=row["spellF1_after"]-row["spellF1_before"]))
    pd.DataFrame(deltas).to_csv(output/"block_deltas.csv", index=False)
    result.update(official_score_agreement=True, shared_decoder_parity=True, seed42_screen_parity=True)
    (output/"promotion_checks.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
