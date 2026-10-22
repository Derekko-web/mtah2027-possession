"""Nested six-model fitting for prospective learned short-endpoint corrections."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from control_refinement import fast_columns, fit_control, predict_control
from endpoint_calibration import (calibrate_endpoints, contiguous_components, estimate_correction,
                                  nested_folds, timing_examples)
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from stoppage_consensus import decode_stoppage_consensus
from stoppage_context import fit_context
from temporal_candidate import fit_models, predict_models
from xgboost_baseline import CLASSES


def fit_pipeline(base, ordered, physical, restart, weak, labels, train, validation, output, seed):
    """Every target and training weight is read through explicit train indices."""
    output.mkdir(parents=True, exist_ok=False)
    result = []
    for method,x,name in (("ordered_xgboost_frame",ordered,"ordered"), ("previous_ensemble",base,"previous")):
        models = fit_models(method,x,labels,train,seed)
        for index,model in enumerate(models):
            model.save_model(output/f"{name}_{index}.ubj")
        result.append(predict_models(models,x.iloc[validation]))
    fast_inputs = base[fast_columns(base)]
    model = fit_control(fast_inputs,labels,train,seed)
    model.save_model(output/"fast.ubj")
    result.append(predict_control(model,fast_inputs.iloc[validation]).astype(float))
    heads = []
    for name,x in (("physical",physical), ("restart",restart)):
        model = fit_context(x,labels,train,weak["frames"],weak["target"],.5,seed)
        model.save_model(output/f"dead_{name}.ubj")
        heads.append(model.predict_proba(x.iloc[validation])[:,1].astype(float))
    result.append(np.column_stack(heads))
    np.savez(output/"probabilities.npz", ordered=result[0], previous=result[1], fast=result[2], heads=result[3], frames=validation)
    (output/"fit.json").write_text(json.dumps(dict(seed=seed,train=np.asarray(train).tolist(),validation=np.asarray(validation).tolist(),
        eligible_target_sha256=hashlib.sha256(labels[train].tobytes()).hexdigest(),
        model_sha256={path.name:digest(path) for path in sorted(output.glob("*.ubj"))}),indent=2)+"\n")
    return result


def padded(arrays, frames, length):
    result = []
    for value in arrays:
        full = np.full((length,value.shape[1]),np.nan)
        full[frames] = value
        result.append(full)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/control_calibration_v1/screen")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True,exist_ok=False)
    plan_path = ROOT/"experiments/ENDPOINT_CALIBRATION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    source = ROOT/"data/advanced_validation"
    raw,base,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","base","ordered")]
    reference = ROOT/"data/control_calibration_v1/baseline_confirmation"
    cache = ROOT/"data/control_renewal_v1/stoppage_inputs"
    cache_expected = json.loads((reference/"manifest.json").read_text())["cache_sha256"]
    for name,expected in cache_expected.items():
        if digest(cache/name) != expected:
            raise ValueError("Selected weak-feature cache changed: "+name)
    for name,expected in json.loads((cache/"manifest.json").read_text())["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Cached feature source changed: "+name)
    physical,restart = [pd.read_pickle(cache/(name+".pkl")) for name in ("physical","restart")]
    weak = np.load(cache/"weak_samples.npz")
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public,np.arange(11185,18685))
    splits_path = source/"splits.json"
    splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in (gt,splits_path,cache/"manifest.json",source/"raw.pkl",source/"base.pkl",source/"ordered.pkl")}
    rows,details,segments,calibrators,nested_records,all_cases = [],[],[],[],[],[]
    preflight_done = False
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"],np.array(s["train"]),np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        ref_path = reference/protocol/"oof_predictions.csv"
        hashes[str(ref_path.relative_to(ROOT))] = digest(ref_path)
        original = pd.read_csv(ref_path)
        table = pd.DataFrame({"frame":public,"ground_truth":labels[public]})
        seed = 42
        paths = [source/protocol/f"{name}_seed{seed}_probabilities.npy" for name in ("ordered_xgboost_frame","previous_ensemble")]
        paths += [ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy",
                  reference/protocol/f"heads_seed{seed}_probabilities.npy"]
        arrays = []
        for path in paths:
            hashes[str(path.relative_to(ROOT))] = digest(path)
            arrays.append(np.load(path))
        blocks = [(int(v[0]),int(v[-1])+1) for _,_,v in folds]
        selected = decode_stoppage_consensus(*padded(arrays,public,len(raw)),raw,blocks)[public]
        np.testing.assert_array_equal(selected,original.stoppage_consensus_seed42)
        reference_states = np.array([np.flatnonzero(CLASSES == label)[0] for label in selected])
        predictions = {name:selected.copy() for name in plan["screen"]["methods"]}
        for outer,train,validation in folds:
            outer_labels = np.full(len(labels),"X",dtype="<U1")
            outer_labels[train] = labels[train]
            outer_folder = folder/f"outer{outer}"
            outer_folder.mkdir()
            if not preflight_done:
                print("Exact selected-model refit preflight",flush=True)
                fitted = fit_pipeline(base,ordered,physical,restart,weak,outer_labels,train,validation,output/"preflight",seed)
                for actual,expected in zip(fitted,arrays):
                    np.testing.assert_array_equal(actual,expected[validation-public[0]])
                preflight_done = True
                print("All six preflight raw probabilities match exactly",flush=True)
            cases = []
            for inner,inner_train,inner_validation in nested_folds(train):
                record = dict(protocol=protocol,outer=outer,inner=inner,train=inner_train.tolist(),validation=inner_validation.tolist())
                if not len(inner_validation) or set(outer_labels[inner_train]) != {"H","A","D"}:
                    record["skipped"] = "empty validation" if not len(inner_validation) else "missing fitting class"
                    nested_records.append(record)
                    continue
                print(protocol,"outer",outer,"inner",inner,"six models",flush=True)
                inner_folder = outer_folder/f"inner{inner}"
                fitted = fit_pipeline(base,ordered,physical,restart,weak,outer_labels,inner_train,inner_validation,inner_folder,seed)
                components = contiguous_components(inner_validation)
                predicted = decode_stoppage_consensus(*padded(fitted,inner_validation,len(raw)),raw,components)
                np.save(inner_folder/"predictions.npy",predicted[inner_validation])
                for start,end in components:
                    if np.any(outer_labels[start:end] == "X"):
                        raise AssertionError("Calibration matching touches outer-held or excluded labels")
                    pc = np.array([np.flatnonzero(CLASSES == label)[0] for label in predicted[start:end]])
                    tc = np.array([np.flatnonzero(CLASSES == label)[0] for label in outer_labels[start:end]])
                    found = timing_examples(pc,tc,start)
                    for case in found:
                        case.update(inner=inner,component_start=start,component_end_exclusive=end)
                    cases.extend(found)
                record.update(skipped=None,components=components)
                nested_records.append(record)
            (outer_folder/"timing_examples.json").write_text(json.dumps(cases,indent=2)+"\n")
            for case in cases:
                all_cases.append(dict(protocol=protocol,outer=outer,**case))
            print(protocol,"outer",outer,"matched timing cases",len(cases),flush=True)
            ix = validation-public[0]
            for name,config in plan["screen"]["methods"].items():
                correction = estimate_correction(cases,config["minimum"])
                predictions[name][ix] = CLASSES[calibrate_endpoints(reference_states[ix],correction,config["both"])]
                calibrators.append(dict(protocol=protocol,outer=outer,method=name,**config,**correction))
            pd.DataFrame(calibrators).to_csv(output/"calibrators.csv",index=False)
            pd.DataFrame(all_cases).to_csv(output/"timing_examples.csv",index=False)
            (output/"nested_splits.json").write_text(json.dumps(nested_records,indent=2)+"\n")
        for name,prediction in {"current":selected,**predictions}.items():
            method = f"{name}_seed42"
            table[method] = prediction
            np.testing.assert_array_equal(prediction == "D",selected == "D")
            record_predictions(output,protocol,method,public,labels,folds,prediction,rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,method)
        add_short_metrics(rows,segments)
        table.to_csv(folder/"oof_predictions.csv",index=False)
        pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
    names = ("research_endpoint_calibration.py","endpoint_calibration.py","temporal_candidate.py","build_candidate.py",
             "control_refinement.py","stoppage_context.py","stoppage_consensus.py","renewal_control.py")
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=digest(plan_path),input_sha256=hashes,
        source_sha256={name:digest(ROOT/"experiments"/name) for name in names},cache_sha256=cache_expected,
        preflight_all_six_raw_probability_parity=True,calibration_only_outer_training_labels=True,
        all_D_masks_identical=True,completed_inner_fits=sum(record["skipped"] is None for record in nested_records)),indent=2)+"\n")
    columns = ["protocol","model","spellF1","macroF1","short_live_mean_best_iou","short_live_matched_iou50"]
    print(pd.DataFrame(rows)[columns].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
