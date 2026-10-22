"""Audit the local contact-dynamics screen and enforce its prospective gates."""

import io
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
from contact_dynamics import geometry_features, impulse_features, load_observations, repair_ball
from control_refinement import contact_support, predict_control
from stoppage_consensus import decode_stoppage_consensus
from weak_ownership import ownership_prediction
from xgboost_baseline import CLASSES


def screen_checks(scores, spans, plan):
    index = scores.set_index(["protocol", "model"])
    if index.index.has_duplicates or len(scores) != 39:
        raise ValueError("Incomplete or duplicate frozen screen scores")
    checks = {}
    for name in sorted(set(scores.model)-{"current_seed42"}):
        pairs = []
        for protocol in plan["screen"]["protocols"]:
            old, new = index.loc[(protocol, "current_seed42")], index.loc[(protocol, name)]
            short = spans[(spans.protocol == protocol) & (spans.label != "D") & (spans.frames <= 125)]
            a = short[short.model == "current_seed42"].set_index(["start", "end", "label"])
            b = short[short.model == name].set_index(["start", "end", "label"])
            pd.testing.assert_index_equal(a.index, b.index)
            flags = dict(primary_no_regression=bool(new.spellF1 >= old.spellF1-1e-12),
                         macro=bool(new.macroF1 >= old.macroF1-.005-1e-12),
                         dead_recall=bool(new.dead_recall >= old.dead_recall-.02-1e-12),
                         false_dead=bool(new.false_dead_frames <= old.false_dead_frames+10),
                         stoppages=bool(new.missed_stoppages <= old.missed_stoppages),
                         prior_short_recovery=bool((b.recall[a.recall >= .5] >= .5).all()))
            if protocol == "buffered_minutes":
                flags.update(substantial_primary=bool(new.spellF1 >= old.spellF1+.04-1e-12),
                             short_iou=bool(new.short_live_mean_best_iou > old.short_live_mean_best_iou+1e-12),
                             extra_short_match=bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50+1))
            else:
                flags["positive_primary"] = bool(new.spellF1 > old.spellF1+1e-12)
            pairs.append(dict(protocol=protocol, flags=flags, primary_delta=float(new.spellF1-old.spellF1),
                              macro_delta=float(new.macroF1-old.macroF1), false_dead_delta=int(new.false_dead_frames-old.false_dead_frames)))
        checks[name] = dict(passed=all(all(pair["flags"].values()) for pair in pairs), pairs=pairs)
    return dict(eligible=[name for name,value in checks.items() if value["passed"]], methods=checks)


def main():
    root = ROOT/"data/control_dynamics_v1"
    cache, output = root/"inputs", root/"screen"
    plan_path = ROOT/"experiments/CONTACT_DYNAMICS_PLAN.json"
    plan = json.loads(plan_path.read_text())
    for folder in (cache, output):
        if (folder/"plan.json").read_bytes() != plan_path.read_bytes():
            raise ValueError("Frozen plan differs")
        manifest = json.loads((folder/"manifest.json").read_text())
        for name, expected in manifest["source_sha256"].items():
            if digest(ROOT/"experiments"/name) != expected:
                raise ValueError("Source changed: "+name)
        for name, expected in manifest.get("cache_sha256", {}).items():
            if digest(folder/name) != expected:
                raise ValueError("Feature cache changed: "+name)
        for name, expected in manifest.get("input_sha256", {}).items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            if digest(path) != expected:
                raise ValueError("Input changed: "+name)
    baseline = json.loads((root/"baseline_manifest.json").read_text())
    for name, expected in baseline.items():
        if name != "experiments/EXPERIMENT_LEDGER.csv" and digest(ROOT/name) != expected:
            raise ValueError("Selected artifact changed: "+name)
    selected = json.loads((ROOT/"data/candidate_stoppage_consensus/manifest.json").read_text())
    for name, expected in selected["source_sha256"].items():
        path = ROOT/name
        if not path.is_file():
            path = ROOT/"experiments"/name
        if digest(path) != expected:
            raise ValueError("Selected source changed: "+name)
    with tarfile.open(root/"experiments_before.tar.gz") as archive:
        before = pd.read_csv(io.BytesIO(archive.extractfile("experiments/EXPERIMENT_LEDGER.csv").read()))
    after = pd.read_csv(ROOT/"experiments/EXPERIMENT_LEDGER.csv")
    keys = ["round", "protocol", "model", "source"]
    preserved = after.merge(before[keys], on=keys, how="inner", validate="one_to_one")
    pd.testing.assert_frame_equal(before.sort_values(keys).reset_index(drop=True),
                                  preserved[before.columns].sort_values(keys).reset_index(drop=True))
    public = np.arange(11185, 18685)
    # Read additional observations to verify that the actual public features do
    # not depend on truncating the input at the cached padded prefix.
    xy, speeds, ball, teams = load_observations(19060)
    for method in ("median", "robust"):
        corrected = repair_ball(ball, method)
        full = geometry_features(xy, speeds, corrected, ball, teams)
        cached = pd.read_pickle(cache/f"{method}_geometry.pkl")
        pd.testing.assert_frame_equal(full.iloc[public], cached.iloc[public])
        if method == "robust":
            pd.testing.assert_frame_equal(impulse_features(corrected, full).iloc[public],
                                          pd.read_pickle(cache/"impulse.pkl").iloc[public])
    source = ROOT/"data/advanced_validation"
    raw, ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw", "ordered")]
    geometry = {name:pd.read_pickle(cache/f"{name}_geometry.pkl") for name in ("median", "robust")}
    views = dict(ordered_control=ordered, median_control=ordered.join(geometry["median"]),
                 robust_control=ordered.join(geometry["robust"]),
                 impulse_control=ordered.join(geometry["robust"]).join(pd.read_pickle(cache/"impulse.pkl")))
    support = {"ordered_control":contact_support(raw)}
    for name in ("median_control", "robust_control", "impulse_control"):
        view = geometry["median" if name == "median_control" else "robust"]
        support[name] = contact_support(view.rename(columns={p+"clean_distance_1":p+"distance_1" for p in ("h__", "a__")}))
    splits = json.loads((source/"splits.json").read_text())
    scores, spans = [pd.read_csv(output/name) for name in ("scores.csv", "segment_diagnostics.csv")]
    checks = screen_checks(scores, spans, plan)
    official, deltas, reloads = [], [], 0
    folds_table = pd.read_csv(output/"fold_scores.csv")
    for protocol in plan["screen"]["protocols"]:
        folds = [s for s in splits if s["protocol"] == protocol]
        blocks = [(min(s["validation"]), max(s["validation"])+1) for s in folds]
        table = pd.read_csv(output/protocol/"oof_predictions.csv")
        reference = np.array([np.flatnonzero(CLASSES == label)[0] for label in table.current_seed42])
        paths = [source/protocol/f"{name}_seed42_probabilities.npy" for name in ("ordered_xgboost_frame", "previous_ensemble")]
        paths += [ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/"fast_seed42_probabilities.npy",
                  root/"baseline_confirmation"/protocol/"heads_seed42_probabilities.npy"]
        arrays = []
        for path, width in zip(paths, (3, 3, 2, 2)):
            value = np.full((len(raw), width), np.nan)
            value[public] = np.load(path)
            arrays.append(value)
        np.testing.assert_array_equal(decode_stoppage_consensus(*arrays, raw, blocks)[public], table.current_seed42)
        for name in plan["screen"]["models"]:
            probability = np.full((len(raw), 2), np.nan)
            for fold in folds:
                model = XGBClassifier()
                model.load_model(output/protocol/f"{name}_seed42_block{fold['block']}.ubj")
                validation = np.array(fold["validation"])
                probability[validation] = predict_control(model, views[name].iloc[validation])
                reloads += 1
            np.testing.assert_array_equal(probability[public], np.load(output/protocol/f"{name}_seed42_probabilities.npy"))
            for mode in plan["screen"]["decoders"]:
                method = f"{name}_{mode}_seed42"
                if mode == "replace_fast":
                    prediction = decode_stoppage_consensus(arrays[0], arrays[1], probability, arrays[3], raw, blocks)[public]
                else:
                    prediction = np.empty(len(public), dtype="<U1")
                    for fold in folds:
                        validation = np.array(fold["validation"])
                        ix = validation-public[0]
                        prediction[ix] = CLASSES[ownership_prediction(reference[ix], probability[validation], support[name][validation],
                                                                     np.zeros((len(ix), 2)), mode)]
                    np.testing.assert_array_equal(prediction == "D", table.current_seed42.to_numpy() == "D")
                np.testing.assert_array_equal(prediction, table[method])
                a = folds_table[(folds_table.protocol == protocol) & (folds_table.model == "current_seed42")]
                b = folds_table[(folds_table.protocol == protocol) & (folds_table.model == method)]
                paired = a.merge(b, on=["protocol", "minute"], suffixes=["_before", "_after"], validate="one_to_one")
                for row in paired.to_dict("records"):
                    deltas.append(dict(protocol=protocol, method=method, block=row["minute"], before=row["spellF1_before"],
                                       after=row["spellF1_after"], delta=row["spellF1_after"]-row["spellF1_before"]))
        with (output/protocol/"official.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT/"evaluation/run_evaluation.py"), "--gt", "public", "--results",
                            str(output/protocol/"scorer_inputs"), "--out", str(output/protocol/"official"),
                            "--bootstrap", "0", "--no-detail"], check=True, stdout=log, stderr=subprocess.STDOUT)
        for row in scores[scores.protocol == protocol].to_dict("records"):
            report = json.loads((output/protocol/"official/reports"/(row["model"]+".json")).read_text())
            if abs(report["spell_f1_mean"]-row["spellF1"]) > 1e-12 or abs(report["macro_f1"]-row["macroF1"]) > 1e-12:
                raise AssertionError("Official scoring mismatch")
            official.append(dict(protocol=protocol, model=row["model"], spellF1=report["spell_f1_mean"], macroF1=report["macro_f1"]))
    pd.DataFrame(official).to_csv(output/"official_comparison.csv", index=False)
    pd.DataFrame(deltas).to_csv(output/"block_deltas.csv", index=False)
    (output/"promotion_checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    audit = dict(prior_ledger_rows=len(before), prior_ledger_rows_preserved=True, ledger_rows=len(after),
                 protected_candidate_unchanged=True, selected_source_hashes_match=True,
                 source_input_cache_hashes_match=True, actual_padded_feature_parity=True,
                 shared_inference_parity_rows=len(scores), official_score_agreement_rows=len(official),
                 model_probability_reload_parity_count=reloads, screen_eligible=checks["eligible"],
                 block_comparisons=len(deltas), block_primary_regressions=sum(row["delta"] < -1e-12 for row in deltas))
    (root/"integrity.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))
    if checks["eligible"]:
        raise RuntimeError("An unchanged screen finalist requires prospective confirmation, not automatic promotion")


if __name__ == "__main__":
    main()
