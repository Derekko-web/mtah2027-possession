"""Verify nested label isolation, checkpoints, timing estimates and official scores."""

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
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from control_refinement import fast_columns,predict_control
from endpoint_calibration import calibrate_endpoints,contiguous_components,estimate_correction,timing_examples
from games import require
from research_endpoint_calibration import padded
from stoppage_consensus import decode_stoppage_consensus
from temporal_candidate import predict_models
from xgboost_baseline import CLASSES


def load_model(path):
    model = XGBClassifier()
    model.load_model(path)
    return model


def verify_fit(folder,base,ordered,physical,restart,labels):
    meta = json.loads((folder/"fit.json").read_text())
    train,validation = np.array(meta["train"]),np.array(meta["validation"])
    import hashlib
    if hashlib.sha256(labels[train].tobytes()).hexdigest() != meta["eligible_target_sha256"]:
        raise ValueError("Eligible training targets differ")
    for name,expected in meta["model_sha256"].items():
        if digest(folder/name) != expected:
            raise ValueError("Nested checkpoint changed")
    actual = [predict_models([load_model(folder/"ordered_0.ubj")],ordered.iloc[validation]),
              predict_models([load_model(folder/f"previous_{i}.ubj") for i in (0,1)],base.iloc[validation]),
              predict_control(load_model(folder/"fast.ubj"),base[fast_columns(base)].iloc[validation]).astype(float),
              np.column_stack([load_model(folder/f"dead_{name}.ubj").predict_proba(x.iloc[validation])[:,1].astype(float)
                               for name,x in (("physical",physical),("restart",restart))])]
    cached = np.load(folder/"probabilities.npz")
    np.testing.assert_array_equal(validation,cached["frames"])
    for name,p in zip(("ordered","previous","fast","heads"),actual):
        np.testing.assert_array_equal(p,cached[name])
    return actual,train,validation


def promotion_checks(scores,segments,plan):
    index = scores.set_index(["protocol","model"])
    if index.index.has_duplicates or len(scores) != 15:
        raise ValueError("Incomplete frozen screen")
    result = {}
    for method in plan["screen"]["methods"]:
        pairs = []
        for protocol in plan["screen"]["protocols"]:
            old,new = index.loc[(protocol,"current_seed42")],index.loc[(protocol,f"{method}_seed42")]
            short = segments[(segments.protocol == protocol) & (segments.label != "D") & (segments.frames <= 125)]
            before = short[short.model == "current_seed42"].set_index(["start","end","label"])
            after = short[short.model == f"{method}_seed42"].set_index(["start","end","label"])
            pd.testing.assert_index_equal(before.index,after.index)
            flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12),
                         short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                         macro=bool(new.macroF1 >= old.macroF1-.005-1e-12),
                         prior_short_recovery=bool((after.recall[before.recall >= .5] >= .5).all()))
            if protocol == "buffered_minutes":
                flags.update(substantial_primary=bool(new.spellF1 >= old.spellF1+.04-1e-12),
                             extra_short_match=bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50+1))
            else:
                flags["positive_primary"] = bool(new.spellF1 > old.spellF1+1e-12)
            pairs.append(dict(protocol=protocol,checks=flags,primary_delta=float(new.spellF1-old.spellF1),
                              macro_delta=float(new.macroF1-old.macroF1),short_iou_delta=float(new.short_live_mean_best_iou-old.short_live_mean_best_iou)))
        result[method] = dict(passed=all(all(pair["checks"].values()) for pair in pairs),pairs=pairs)
    return dict(eligible=[name for name,value in result.items() if value["passed"]],methods=result)


def main():
    root = ROOT/"data/control_calibration_v1"
    output = root/"screen"
    manifest = json.loads((output/"manifest.json").read_text())
    plan_path = ROOT/"experiments/ENDPOINT_CALIBRATION_PLAN.json"
    if (output/"plan.json").read_bytes() != plan_path.read_bytes():
        raise ValueError("Frozen plan changed")
    plan = json.loads(plan_path.read_text())
    for name,expected in manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Source changed: "+name)
    for name,expected in manifest["input_sha256"].items():
        if digest(ROOT/name) != expected:
            raise ValueError("Input changed: "+name)
    cache = ROOT/"data/control_renewal_v1/stoppage_inputs"
    for name,expected in manifest["cache_sha256"].items():
        if digest(cache/name) != expected:
            raise ValueError("Weak cache changed")
    cache_manifest = json.loads((cache/"manifest.json").read_text())
    actual_inputs = {Path(path).name:digest(path) for path in require(["atd_meta","atd_raw","smart_tagging"])}
    if actual_inputs != cache_manifest["input_sha256"]:
        raise ValueError("Actual competition inputs differ from cached inputs")
    for name,expected in json.loads((root/"baseline_manifest.json").read_text()).items():
        if name != "experiments/EXPERIMENT_LEDGER.csv" and digest(ROOT/name) != expected:
            raise ValueError("Protected selected artifact changed: "+name)
    selected = json.loads((ROOT/"data/candidate_stoppage_consensus/manifest.json").read_text())
    for name,expected in selected["source_sha256"].items():
        path = ROOT/name
        if not path.is_file(): path = ROOT/"experiments"/name
        if digest(path) != expected: raise ValueError("Selected model source changed")
    with tarfile.open(root/"experiments_before.tar.gz") as archive:
        before = pd.read_csv(io.BytesIO(archive.extractfile("experiments/EXPERIMENT_LEDGER.csv").read()))
    after = pd.read_csv(ROOT/"experiments/EXPERIMENT_LEDGER.csv")
    keys = ["round","protocol","model","source"]
    preserved = after.merge(before[keys],on=keys,how="inner",validate="one_to_one")
    pd.testing.assert_frame_equal(before.sort_values(keys).reset_index(drop=True),
                                  preserved[before.columns].sort_values(keys).reset_index(drop=True))
    source = ROOT/"data/advanced_validation"
    raw,base,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","base","ordered")]
    physical,restart = [pd.read_pickle(cache/(name+".pkl")) for name in ("physical","restart")]
    gt = pd.read_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv")
    labels,public = gt.label.to_numpy(dtype="<U1"),np.arange(11185,18685)
    splits = json.loads((source/"splits.json").read_text())
    nested = json.loads((output/"nested_splits.json").read_text())
    calibrators = pd.read_csv(output/"calibrators.csv")
    preflight,train,validation = verify_fit(output/"preflight",base,ordered,physical,restart,labels)
    first = next(s for s in splits if s["protocol"] == "buffered_minutes" and s["block"] == 1)
    np.testing.assert_array_equal(train,first["train"])
    np.testing.assert_array_equal(validation,first["validation"])
    reloads,inner_count,example_count = 6,0,0
    preflight_paths = [source/"buffered_minutes"/f"{name}_seed42_probabilities.npy" for name in ("ordered_xgboost_frame","previous_ensemble")]
    preflight_paths += [ROOT/"data/control_boundary_v1/baseline_confirmation/buffered_minutes/fast_seed42_probabilities.npy",
                       root/"baseline_confirmation/buffered_minutes/heads_seed42_probabilities.npy"]
    for value,path in zip(preflight,preflight_paths):
        np.testing.assert_array_equal(value,np.load(path)[validation-public[0]])
    scores,segments,folds = [pd.read_csv(output/name) for name in ("scores.csv","segment_diagnostics.csv","fold_scores.csv")]
    official,deltas = [],[]
    for protocol in plan["screen"]["protocols"]:
        table = pd.read_csv(output/protocol/"oof_predictions.csv")
        selected_ref = pd.read_csv(root/"baseline_confirmation"/protocol/"oof_predictions.csv")
        np.testing.assert_array_equal(table.current_seed42,selected_ref.stoppage_consensus_seed42)
        reference = np.array([np.flatnonzero(CLASSES == x)[0] for x in table.current_seed42])
        rebuilt = {name:table.current_seed42.to_numpy().copy() for name in plan["screen"]["methods"]}
        for outer in [s for s in splits if s["protocol"] == protocol]:
            outer_train,outer_validation = np.array(outer["train"]),np.array(outer["validation"])
            outer_folder = output/protocol/f"outer{outer['block']}"
            cases = []
            records = [r for r in nested if r["protocol"] == protocol and r["outer"] == outer["block"]]
            if sorted(r["inner"] for r in records) != [1,2,3]: raise ValueError("Incomplete inner split audit")
            covered = []
            for record in records:
                a,b = 11185+(record["inner"]-1)*2500,11185+record["inner"]*2500
                expected_validation = outer_train[(outer_train >= a) & (outer_train < b)]
                expected_train = outer_train[(outer_train < a-250) | (outer_train >= b+250)]
                np.testing.assert_array_equal(record["train"],expected_train)
                np.testing.assert_array_equal(record["validation"],expected_validation)
                if np.intersect1d(expected_train,outer_validation).size or np.intersect1d(expected_validation,outer_validation).size:
                    raise AssertionError("Outer validation leaked into inner fitting/calibration")
                covered.extend(expected_validation)
                if record["skipped"] is not None:
                    if len(expected_validation) and set(labels[expected_train]) == {"H","A","D"}:
                        raise ValueError("Unexpected inner skip")
                    continue
                inner_folder = outer_folder/f"inner{record['inner']}"
                values,fit_train,fit_validation = verify_fit(inner_folder,base,ordered,physical,restart,labels)
                np.testing.assert_array_equal(fit_train,expected_train)
                np.testing.assert_array_equal(fit_validation,expected_validation)
                components = contiguous_components(expected_validation)
                predicted = decode_stoppage_consensus(*padded(values,expected_validation,len(raw)),raw,components)
                np.testing.assert_array_equal(predicted[expected_validation],np.load(inner_folder/"predictions.npy"))
                for start,end in components:
                    if not np.isin(np.arange(start,end),outer_train).all(): raise AssertionError("Target matching crosses excluded labels")
                    pc = np.array([np.flatnonzero(CLASSES == x)[0] for x in predicted[start:end]])
                    tc = np.array([np.flatnonzero(CLASSES == x)[0] for x in labels[start:end]])
                    found = timing_examples(pc,tc,start)
                    for case in found: case.update(inner=record["inner"],component_start=start,component_end_exclusive=end)
                    cases.extend(found)
                reloads += 6
                inner_count += 1
            np.testing.assert_array_equal(np.sort(covered),outer_train)
            if cases != json.loads((outer_folder/"timing_examples.json").read_text()):
                raise ValueError("Training-only calibration cases differ")
            example_count += len(cases)
            ix = outer_validation-public[0]
            for name,config in plan["screen"]["methods"].items():
                correction = estimate_correction(cases,config["minimum"])
                saved = calibrators[(calibrators.protocol == protocol) & (calibrators.outer == outer["block"]) & (calibrators.method == name)]
                if len(saved) != 1: raise ValueError("Missing/duplicate calibrator")
                saved = saved.iloc[0]
                for key in ("start","end","count","sufficient"):
                    if saved[key] != correction[key]: raise ValueError("Calibrator differs from training-only median")
                rebuilt[name][ix] = CLASSES[calibrate_endpoints(reference[ix],correction,config["both"])]
        for name,prediction in rebuilt.items():
            method = f"{name}_seed42"
            np.testing.assert_array_equal(prediction,table[method])
            np.testing.assert_array_equal(prediction == "D",table.current_seed42.to_numpy() == "D")
            a = folds[(folds.protocol == protocol) & (folds.model == "current_seed42")]
            b = folds[(folds.protocol == protocol) & (folds.model == method)]
            paired = a.merge(b,on=["protocol","minute"],suffixes=["_before","_after"],validate="one_to_one")
            for row in paired.to_dict("records"):
                deltas.append(dict(protocol=protocol,method=method,block=row["minute"],before=row["spellF1_before"],
                                   after=row["spellF1_after"],delta=row["spellF1_after"]-row["spellF1_before"]))
        with (output/protocol/"official.log").open("w") as log:
            subprocess.run([sys.executable,str(ROOT/"evaluation/run_evaluation.py"),"--gt","public","--results",str(output/protocol/"scorer_inputs"),
                            "--out",str(output/protocol/"official"),"--bootstrap","0","--no-detail"],check=True,stdout=log,stderr=subprocess.STDOUT)
        for row in scores[scores.protocol == protocol].to_dict("records"):
            report = json.loads((output/protocol/"official/reports"/(row["model"]+".json")).read_text())
            if abs(report["spell_f1_mean"]-row["spellF1"]) > 1e-12 or abs(report["macro_f1"]-row["macroF1"]) > 1e-12:
                raise AssertionError("Official score mismatch")
            official.append(dict(protocol=protocol,model=row["model"],spellF1=report["spell_f1_mean"],macroF1=report["macro_f1"]))
    checks = promotion_checks(scores,segments,plan)
    (output/"promotion_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    pd.DataFrame(official).to_csv(output/"official_comparison.csv",index=False)
    pd.DataFrame(deltas).to_csv(output/"block_deltas.csv",index=False)
    audit = dict(old_ledger_rows=len(before),old_ledger_rows_preserved=True,ledger_rows=len(after),selected_artifact_source_unchanged=True,
                 actual_inputs_source_cache_match=True,preflight_all_six_exact=True,nested_outer_label_isolation=True,
                 completed_inner_fits=inner_count,models_reload_exactly=reloads,training_case_rows=example_count,
                 calibration_reestimated_exactly=True,all_D_masks_unchanged=True,shared_outer_inference_parity_rows=len(scores),
                 official_agreement_rows=len(official),eligible=checks["eligible"],block_comparisons=len(deltas),
                 block_primary_regressions=sum(row["delta"] < -1e-12 for row in deltas))
    (root/"integrity.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()
