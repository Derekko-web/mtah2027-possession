"""Frozen secondary-objective gates, decoder parity and official scoring."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from stoppage_consensus import decode_stoppage_consensus


def promotion_checks(scores, segments, plan):
    index = scores.set_index(["protocol", "model"])
    if len(scores) != len(plan["protocols"])*len(plan["seeds"])*len(plan["methods"]) or index.index.has_duplicates:
        raise ValueError("Incomplete or duplicate confirmation")
    pairs = []
    for protocol in plan["protocols"]:
        for seed in plan["seeds"]:
            old_name, new_name = f"current_seed{seed}", f"stoppage_consensus_seed{seed}"
            old, new = index.loc[(protocol, old_name)], index.loc[(protocol, new_name)]
            short = segments[(segments.protocol == protocol) & (segments.label != "D") & (segments.frames <= 125)]
            before = short[short.model == old_name].set_index(["start", "end", "label"])
            after = short[short.model == new_name].set_index(["start", "end", "label"])
            pd.testing.assert_index_equal(before.index, after.index)
            flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12),
                         macro_gain=bool(new.macroF1 >= old.macroF1+.005-1e-12),
                         false_dead=bool(new.false_dead_frames <= old.false_dead_frames),
                         dead_recall_gain=bool(new.dead_recall >= old.dead_recall+.05-1e-12),
                         stoppage_recovery=bool(new.missed_stoppages == 0 and new.missed_stoppages <= old.missed_stoppages),
                         prior_short_recovery=bool((after.recall[before.recall >= .5] >= .5).all()),
                         short_recall=bool(new.short_live_mean_recall >= old.short_live_mean_recall-1e-12))
            pairs.append(dict(protocol=protocol, seed=seed, checks=flags,
                              spellF1_delta=float(new.spellF1-old.spellF1),
                              macroF1_delta=float(new.macroF1-old.macroF1),
                              dead_recall_delta=float(new.dead_recall-old.dead_recall),
                              false_dead_delta=int(new.false_dead_frames-old.false_dead_frames)))
    passed = all(all(pair["checks"].values()) for pair in pairs)
    return dict(selected="stoppage_consensus" if passed else None, passed=passed, paired=pairs,
                original_primary_gain_criterion_met=False,
                objective="Separate prospectively frozen adaptive stoppage-recovery objective")


def main():
    output = ROOT/"data/control_renewal_v1/recovery_confirmation"
    plan_path = ROOT/"experiments/STOPPAGE_RECOVERY_CONFIRMATION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    if (output/"plan.json").read_bytes() != plan_path.read_bytes():
        raise ValueError("Confirmation plan changed")
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Frozen source changed: "+name)
    scores, segments = [pd.read_csv(output/name) for name in ("scores.csv", "segment_diagnostics.csv")]
    checks = promotion_checks(scores, segments, plan)
    raw = pd.read_pickle(ROOT/"data/control_renewal_v1/stoppage_inputs/raw.pkl")
    source = ROOT/"data/advanced_validation"
    splits = json.loads((source/"splits.json").read_text())
    public, reports = np.arange(11185, 18685), []
    for protocol in plan["protocols"]:
        blocks = [(min(s["validation"]), max(s["validation"])+1) for s in splits if s["protocol"] == protocol]
        table = pd.read_csv(output/protocol/"oof_predictions.csv")
        for seed in plan["seeds"]:
            arrays = []
            paths = [source/protocol/f"{name}_seed{seed}_probabilities.npy" for name in
                     ("ordered_xgboost_frame", "previous_ensemble")]
            paths.append(ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy")
            paths.append(output/protocol/f"heads_seed{seed}_probabilities.npy")
            for path, width in zip(paths, (3, 3, 2, 2)):
                full = np.full((len(raw), width), np.nan)
                full[public] = np.load(path)
                arrays.append(full)
            for name in plan["methods"]:
                prediction = decode_stoppage_consensus(*arrays, raw, blocks, name != "current")[public]
                np.testing.assert_array_equal(prediction, table[f"{name}_seed{seed}"].to_numpy())
        with (output/protocol/"official.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT/"evaluation/run_evaluation.py"), "--gt", "public",
                            "--results", str(output/protocol/"scorer_inputs"), "--out", str(output/protocol/"official"),
                            "--bootstrap", "1000", "--no-detail"], check=True, stdout=log, stderr=subprocess.STDOUT)
        for row in scores[scores.protocol == protocol].to_dict("records"):
            report = json.loads((output/protocol/"official/reports"/(row["model"]+".json")).read_text())
            if abs(report["spell_f1_mean"]-row["spellF1"]) > 1e-12 or abs(report["macro_f1"]-row["macroF1"]) > 1e-12:
                raise ValueError("Official metric mismatch")
            reports.append(dict(protocol=protocol, model=row["model"],
                                bootstrap90_low=report["spell_bootstrap_low"], bootstrap90_high=report["spell_bootstrap_high"],
                                matched_iou50=report["spell_f1"]["0.5"]["n_matched"],
                                spurious_iou50=report["spell_f1"]["0.5"]["n_spurious"],
                                missed_iou50=report["spell_f1"]["0.5"]["n_missed"]))
    scores.merge(pd.DataFrame(reports), on=["protocol", "model"], validate="one_to_one").to_csv(output/"official_comparison.csv", index=False)
    folds = pd.read_csv(output/"fold_scores.csv")
    deltas = []
    for seed in plan["seeds"]:
        before = folds[folds.model == f"current_seed{seed}"]
        after = folds[folds.model == f"stoppage_consensus_seed{seed}"]
        paired = before.merge(after, on=["protocol", "minute"], suffixes=["_before", "_after"], validate="one_to_one")
        for row in paired.to_dict("records"):
            deltas.append(dict(protocol=row["protocol"], seed=seed, block=row["minute"],
                               spellF1_before=row["spellF1_before"], spellF1_after=row["spellF1_after"],
                               delta=row["spellF1_after"]-row["spellF1_before"]))
    pd.DataFrame(deltas).to_csv(output/"block_deltas.csv", index=False)
    checks.update(shared_decoder_parity=True, official_score_agreement=True)
    (output/"promotion_checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
