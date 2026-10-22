"""Audit both frozen screens, official scores, and unchanged selected artifacts."""

import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from control_refinement import contact_support, predict_control
from evaluate_physics_context import apply_context
from phase_control import phase_control
from short_precision import ownership_evidence, refine_short_endpoints
from temporal_candidate import gap_sides
from weak_ownership import ownership_prediction
from xgboost_baseline import CLASSES


def candidate_gates(scores, segments, predictions, plan):
    index = scores.set_index(["protocol", "model"])
    if index.index.has_duplicates:
        raise ValueError("Duplicate screen scores")
    names = sorted(set(scores.model)-{"current_seed42"})
    result = {}
    for name in names:
        pairs = []
        for protocol in plan["screen"]["protocols"]:
            old, new = index.loc[(protocol, "current_seed42")], index.loc[(protocol, name)]
            short = segments[(segments.protocol == protocol) & (segments.label != "D") & (segments.frames <= 125)]
            a = short[short.model == "current_seed42"].set_index(["start", "end", "label"])
            b = short[short.model == name].set_index(["start", "end", "label"])
            pd.testing.assert_index_equal(a.index, b.index)
            table = predictions[protocol]
            flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12),
                         macro=bool(new.macroF1 >= old.macroF1-1e-12),
                         short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                         short_matches=bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50),
                         prior_short_recovery=bool((b.recall[a.recall >= .5] >= .5).all()),
                         identical_D_mask=bool(np.array_equal(table[name] == "D", table.current_seed42 == "D")))
            if protocol == "buffered_minutes":
                flags.update(substantial_primary=bool(new.spellF1 >= old.spellF1+.04-1e-12),
                             extra_short_match=bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50+1))
            pairs.append(dict(protocol=protocol, checks=flags, primary_delta=float(new.spellF1-old.spellF1),
                              macro_delta=float(new.macroF1-old.macroF1), short_iou_delta=float(new.short_live_mean_best_iou-old.short_live_mean_best_iou)))
        result[name] = dict(passed=all(all(pair["checks"].values()) for pair in pairs), pairs=pairs)
    return dict(eligible=[name for name,value in result.items() if value["passed"]], methods=result)


def main():
    root = ROOT/"data/control_precision_v1"
    baseline = json.loads((root/"baseline_manifest.json").read_text())
    # The ledger is checked against its pre-round archive after appending rows.
    for path, expected in baseline.items():
        if path != "experiments/EXPERIMENT_LEDGER.csv" and digest(ROOT/path) != expected:
            raise ValueError("Protected baseline changed: "+path)
    selected = json.loads((ROOT/"data/candidate_stoppage_consensus/manifest.json").read_text())
    for name, expected in selected["source_sha256"].items():
        path = ROOT/name
        if not path.is_file():
            path = ROOT/"experiments"/name
        if digest(path) != expected:
            raise ValueError("Selected model source changed: "+name)
    with tarfile.open(root/"experiments_before.tar.gz") as archive:
        import io
        before = pd.read_csv(io.BytesIO(archive.extractfile("experiments/EXPERIMENT_LEDGER.csv").read()))
    after = pd.read_csv(ROOT/"experiments/EXPERIMENT_LEDGER.csv")
    keys = ["round", "protocol", "model", "source"]
    prior = after.merge(before[keys], on=keys, how="inner", validate="one_to_one")
    pd.testing.assert_frame_equal(before.sort_values(keys).reset_index(drop=True),
                                  prior[before.columns].sort_values(keys).reset_index(drop=True))
    public = np.arange(11185, 18685)
    source = ROOT/"data/advanced_validation"
    raw = pd.read_pickle(source/"raw.pkl")
    side = gap_sides(raw)[public]
    splits = json.loads((source/"splits.json").read_text())
    phase = phase_control(len(raw))
    phase_support = phase[[prefix+"phase_control_support" for prefix in ("h__", "a__")]].to_numpy()[public]
    support = contact_support(raw)[public]
    audit = dict(protected_baseline_unchanged=True, prior_ledger_rows=len(before), prior_ledger_rows_preserved=True,
                 current_ledger_rows=len(after), selected_sources_match_manifest=True, screens={})
    for family, plan_name in (("screen", "SHORT_PRECISION_PLAN.json"), ("weak_ownership_screen", "WEAK_OWNERSHIP_PLAN.json")):
        folder = root/family
        manifest = json.loads((folder/"manifest.json").read_text())
        if (folder/"plan.json").read_bytes() != (ROOT/"experiments"/plan_name).read_bytes():
            raise ValueError("Frozen screen plan changed")
        for name, expected in manifest["source_sha256"].items():
            if digest(ROOT/"experiments"/name) != expected:
                raise ValueError("Screen source changed: "+name)
        for path, expected in manifest["input_sha256"].items():
            if digest(ROOT/path) != expected:
                raise ValueError("Screen input changed: "+path)
        plan = json.loads((folder/"plan.json").read_text())
        scores = pd.read_csv(folder/"scores.csv")
        spans = pd.read_csv(folder/"segment_diagnostics.csv")
        tables, official = {}, []
        model_count = 0
        for protocol in plan["screen"]["protocols"]:
            table = pd.read_csv(folder/protocol/"oof_predictions.csv")
            tables[protocol] = table
            original = pd.read_csv(root/"baseline_confirmation"/protocol/"oof_predictions.csv")
            np.testing.assert_array_equal(table.current_seed42, original.stoppage_consensus_seed42)
            reference = np.array([np.flatnonzero(CLASSES == x)[0] for x in table.current_seed42])
            folds = [s for s in splits if s["protocol"] == protocol]
            if family == "screen":
                previous = apply_context(np.load(source/protocol/"previous_ensemble_seed42_probabilities.npy"), side, np.zeros(len(public)), False)
                fast = np.load(ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/"fast_seed42_probabilities.npy")
                fast = apply_context(np.column_stack((fast, np.zeros(len(public)))), side, np.zeros(len(public)), False)[:, :2]
                fast /= fast.sum(axis=1, keepdims=True)
                for name, config in plan["screen"]["methods"].items():
                    prediction = reference.copy()
                    for fold in folds:
                        ix = np.array(fold["validation"])-public[0]
                        p = ownership_evidence(previous[ix], fast[ix], **{k:v for k,v in config.items() if k != "radius"})
                        prediction[ix] = refine_short_endpoints(reference[ix], p, config["radius"])
                    np.testing.assert_array_equal(CLASSES[prediction], table[f"{name}_seed42"])
            else:
                for name, config in plan["screen"]["models"].items():
                    x = pd.read_pickle(folder/("semantic_inputs.pkl" if config["semantic"] else "geometry_inputs.pkl"))
                    probability = np.full((len(public), 2), np.nan)
                    for fold in folds:
                        model = XGBClassifier()
                        model.load_model(folder/protocol/f"{name}_seed42_block{fold['block']}.ubj")
                        ix = np.array(fold["validation"])
                        probability[ix-public[0]] = predict_control(model, x.iloc[ix])
                        model_count += 1
                    np.testing.assert_array_equal(probability, np.load(folder/protocol/f"{name}_seed42_probabilities.npy"))
                    for mode in plan["screen"]["decoders"]:
                        prediction = reference.copy()
                        for fold in folds:
                            ix = np.array(fold["validation"])-public[0]
                            prediction[ix] = ownership_prediction(reference[ix], probability[ix], support[ix], phase_support[ix], mode)
                        np.testing.assert_array_equal(CLASSES[prediction], table[f"{name}_{mode}_seed42"])
            with (folder/protocol/"official.log").open("w") as log:
                subprocess.run([sys.executable, str(ROOT/"evaluation/run_evaluation.py"), "--gt", "public",
                                "--results", str(folder/protocol/"scorer_inputs"), "--out", str(folder/protocol/"official"),
                                "--bootstrap", "0", "--no-detail"], check=True, stdout=log, stderr=subprocess.STDOUT)
            for row in scores[scores.protocol == protocol].to_dict("records"):
                report = json.loads((folder/protocol/"official/reports"/(row["model"]+".json")).read_text())
                for a,b in (("spell_f1_mean", "spellF1"), ("macro_f1", "macroF1")):
                    if abs(report[a]-row[b]) > 1e-12:
                        raise AssertionError("Official metric mismatch")
                official.append(dict(protocol=protocol, model=row["model"], spellF1=report["spell_f1_mean"], macroF1=report["macro_f1"]))
        gates = candidate_gates(scores, spans, tables, plan)
        (folder/"promotion_checks.json").write_text(json.dumps(gates, indent=2)+"\n")
        pd.DataFrame(official).to_csv(folder/"official_comparison.csv", index=False)
        folds = pd.read_csv(folder/"fold_scores.csv")
        deltas = []
        for protocol in plan["screen"]["protocols"]:
            old = folds[(folds.protocol == protocol) & (folds.model == "current_seed42")]
            for name in sorted(set(scores.model)-{"current_seed42"}):
                new = folds[(folds.protocol == protocol) & (folds.model == name)]
                paired = old.merge(new, on=["protocol", "minute"], suffixes=["_before", "_after"], validate="one_to_one")
                for row in paired.to_dict("records"):
                    deltas.append(dict(protocol=protocol, method=name, block=row["minute"],
                                       spellF1_before=row["spellF1_before"], spellF1_after=row["spellF1_after"],
                                       delta=row["spellF1_after"]-row["spellF1_before"]))
        pd.DataFrame(deltas).to_csv(folder/"block_deltas.csv", index=False)
        if gates["eligible"]:
            raise RuntimeError("A screen finalist needs prospective confirmation; no automatic promotion")
        audit["screens"][family] = dict(rows=len(scores), official_scorer_agreement=True,
                                       shared_inference_parity=True, models_reload_exactly=model_count,
                                       all_D_masks_identical=True, no_screen_finalist=True,
                                       block_primary_regressions=sum(row["delta"] < -1e-12 for row in deltas),
                                       block_comparisons=len(deltas))
    (root/"integrity.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
