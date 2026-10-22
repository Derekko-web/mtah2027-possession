"""Prospective paired possession screen using excluded weak jersey-color models."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from control_refinement import fast_columns,fit_control,predict_control
from jersey_ownership import decode_jersey_ownership
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_control_renewal import fit_renewal
from research_stoppages import record_predictions
from temporal_candidate import predict_models
from xgboost_baseline import CLASSES


def codes(labels): return np.array([np.flatnonzero(CLASSES == value)[0] for value in labels])


def main():
    root = ROOT/"data/control_jersey_v1"; cache = root/"inputs"
    output = root/"screen"; output.mkdir(exist_ok=False)
    plan_path = ROOT/"experiments/JERSEY_OWNERSHIP_SCREEN_PLAN.json"; plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    cm = json.loads((cache/"manifest.json").read_text())
    if cm["plan_sha256"] != digest(plan_path) or not cm["actual_padded_prefix_parity"] or not cm["stream_seek_pixel_parity"]:
        raise ValueError("Jersey feature plan/parity differs")
    for name,sha in cm["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != sha: raise ValueError("Jersey source changed")
    for name,sha in cm["cache_sha256"].items():
        if digest(cache/name) != sha: raise ValueError("Jersey cache changed")
    source = ROOT/"data/advanced_validation"
    raw,base,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","base","ordered")]
    features = pd.read_pickle(cache/"jersey_inputs.pkl").iloc[:len(raw)]
    inputs = dict(jersey_fast=base[fast_columns(base)].join(features),jersey_joint_frame=ordered.join(features))
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"; labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X"); np.testing.assert_array_equal(public,np.arange(11185,18685))
    splits_path = source/"splits.json"; splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in (gt,splits_path,source/"raw.pkl",source/"base.pkl",source/"ordered.pkl",cache/"manifest.json")}
    rows,details,segments = [],[],[]
    for protocol in plan["protocols"]:
        folder = output/protocol; folder.mkdir()
        folds = [(s["block"],np.array(s["train"]),np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        original_path = root/"baseline_confirmation"/protocol/"oof_predictions.csv"
        hashes[str(original_path.relative_to(ROOT))] = digest(original_path)
        original = pd.read_csv(original_path); reference = original.broadcast_closure_seed42.to_numpy()
        states = codes(reference)
        table = pd.DataFrame(dict(frame=public,ground_truth=labels[public],current_seed42=reference))
        record_predictions(output,protocol,"current_seed42",public,labels,folds,reference,rows,details,segments)
        add_overlap_metrics(rows,segments,table,protocol,"current_seed42")
        for head in plan["heads"]:
            joint = head == "jersey_joint_frame"; probability = np.full((len(raw),3 if joint else 2),np.nan)
            for block,train,val in folds:
                print(protocol,head,"block",block,flush=True)
                model = fit_renewal(inputs[head],labels,train,"frame",42) if joint else fit_control(inputs[head],labels,train,42)
                model.save_model(folder/f"{head}_seed42_block{block}.ubj")
                probability[val] = predict_models([model],inputs[head].iloc[val]) if joint else predict_control(model,inputs[head].iloc[val])
            np.save(folder/f"{head}_seed42_probabilities.npy",probability[public])
            conditional = probability[:,:2]/probability[:,:2].sum(axis=1,keepdims=True)
            for mode in plan["decoders"]:
                prediction = reference.copy()
                for _,_,val in folds:
                    ix = val-public[0]
                    prediction[ix] = CLASSES[decode_jersey_ownership(states[ix],conditional[val],features.iloc[val],mode)]
                if not np.array_equal(prediction == "D",reference == "D"): raise AssertionError("Reference D differs")
                name = f"{head}_{mode}_seed42"; table[name] = prediction
                record_predictions(output,protocol,name,public,labels,folds,prediction,rows,details,segments)
                add_overlap_metrics(rows,segments,table,protocol,name)
        add_short_metrics(rows,segments)
        table.to_csv(folder/"oof_predictions.csv",index=False)
        pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
    names = ("research_jersey_ownership.py","jersey_ownership.py","weak_ownership.py","broadcast_intersection.py",
        "control_refinement.py","research_control_renewal.py")
    result = dict(plan_sha256=digest(plan_path),source_sha256={name:digest(ROOT/"experiments"/name) for name in names},
        input_sha256=hashes,feature_columns={head:list(value) for head,value in inputs.items()},reference_D_identical=True,
        public_train_only=True,selected_color_model_not_refit_on_possession_labels=True)
    (output/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol","model","spellF1","macroF1","short_live_mean_best_iou","short_live_matched_iou50"]].round(6).to_string(index=False),flush=True)


if __name__ == "__main__": main()
