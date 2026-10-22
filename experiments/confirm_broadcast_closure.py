"""Frozen incremental-primary confirmation of unchanged frame-head source closure."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from broadcast_refinement import decode_broadcast_closure
from build_temporal_candidate import digest
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_control_renewal import fit_renewal
from research_stoppages import record_predictions
from stoppage_consensus import decode_stoppage_consensus
from temporal_candidate import predict_models


def main():
    root = ROOT/"data/control_broadcast_v1"
    output = root/"closure_confirmation"
    output.mkdir(parents=True,exist_ok=False)
    plan_path = ROOT/"experiments/BROADCAST_CLOSURE_CONFIRMATION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    for name,expected in plan["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected: raise ValueError("Frozen confirmation source changed")
    for name,expected in plan["input_sha256"].items():
        if digest(ROOT/name) != expected: raise ValueError("Frozen confirmation input changed")
    source = ROOT/"data/advanced_validation"
    raw,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","ordered")]
    features = pd.read_pickle(root/"inputs/combined_inputs.pkl").iloc[:len(raw)]
    inputs = ordered.join(features)
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public,np.arange(11185,18685))
    splits_path = source/"splits.json"
    splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    rows,details,segments = [],[],[]
    for protocol in plan["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"],np.array(s["train"]),np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        blocks = [(int(v[0]),int(v[-1])+1) for _,_,v in folds]
        table = pd.DataFrame({"frame":public,"ground_truth":labels[public]})
        reference = pd.read_csv(root/"baseline_confirmation"/protocol/"oof_predictions.csv")
        for seed in plan["seeds"]:
            probability = np.full((len(raw),3),np.nan)
            for block,train,validation in folds:
                print(protocol,seed,"joint frame block",block,flush=True)
                model = fit_renewal(inputs,labels,train,"frame",seed)
                model.save_model(folder/f"broadcast_joint_frame_seed{seed}_block{block}.ubj")
                probability[validation] = predict_models([model],inputs.iloc[validation])
            np.save(folder/f"broadcast_joint_frame_seed{seed}_probabilities.npy",probability[public])
            if seed == 42:
                np.testing.assert_array_equal(probability[public],np.load(root/"screen"/protocol/"broadcast_joint_frame_seed42_probabilities.npy"))
            paths = [source/protocol/f"{name}_seed{seed}_probabilities.npy" for name in ("ordered_xgboost_frame","previous_ensemble")]
            paths += [ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy",
                      root/"baseline_confirmation"/protocol/f"heads_seed{seed}_probabilities.npy"]
            arrays = []
            for path,width in zip(paths,(3,3,2,2)):
                value = np.full((len(raw),width),np.nan); value[public] = np.load(path); arrays.append(value)
            for name in plan["methods"]:
                prediction = (decode_stoppage_consensus(*arrays,raw,blocks) if name == "current" else
                              decode_broadcast_closure(*arrays,probability,features,raw,blocks))[public]
                if name == "current": np.testing.assert_array_equal(prediction,reference[f"stoppage_consensus_seed{seed}"])
                elif seed == 42:
                    screen = pd.read_csv(root/"intersection_screen"/protocol/"oof_predictions.csv")
                    np.testing.assert_array_equal(prediction,screen.frame_closure_seed42)
                method = f"{name}_seed{seed}"
                table[method] = prediction
                record_predictions(output,protocol,method,public,labels,folds,prediction,rows,details,segments)
                add_overlap_metrics(rows,segments,table,protocol,method)
            add_short_metrics(rows,segments)
            table.to_csv(folder/"oof_predictions.csv",index=False)
            pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
            pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
            pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=digest(plan_path),source_sha256=plan["source_sha256"],
        input_sha256=plan["input_sha256"],feature_columns=list(inputs),seed42_raw_and_output_screen_parity=True),indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol","model","spellF1","macroF1","short_live_mean_best_iou","short_live_matched_iou50"]].round(6).to_string(index=False))


if __name__ == "__main__": main()
