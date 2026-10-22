"""Frozen39-row matched-camera temporal and pair-motion screen."""

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
from pair_candidate import decode_pair_candidate
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_control_renewal import fit_renewal
from research_stoppages import record_predictions
from temporal_candidate import predict_models


def reference_arrays(protocol,seed,n):
    source = ROOT/"data/advanced_validation"
    root = ROOT/"data/control_pair_motion_v1"
    paths = [source/protocol/f"{name}_seed{seed}_probabilities.npy" for name in ("ordered_xgboost_frame","previous_ensemble")]
    paths += [ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy",
        ROOT/"data/control_broadcast_v1/baseline_confirmation"/protocol/f"heads_seed{seed}_probabilities.npy",
        root/"baseline_confirmation"/protocol/f"broadcast_joint_frame_seed{seed}_probabilities.npy"]
    arrays = []
    for path,width in zip(paths,(3,3,2,2,3)):
        value = np.full((n,width),np.nan);value[11185:18685] = np.load(path);arrays.append(value)
    return arrays,{str(path.relative_to(ROOT)):digest(path) for path in paths}


def main():
    root = ROOT/"data/control_pair_motion_v1";cache = root/"inputs"
    output = root/"screen";output.mkdir(exist_ok=False)
    plan_path = ROOT/"experiments/PAIR_MOTION_PLAN.json";plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    cm = json.loads((cache/"manifest.json").read_text())
    if cm["plan_sha256"] != digest(plan_path) or not cm["actual_padded_prefix_parity"]: raise ValueError("Motion cache differs")
    for name,sha in cm["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != sha: raise ValueError("Motion source changed")
    for name,sha in cm["cache_sha256"].items():
        if digest(cache/name) != sha: raise ValueError("Motion cache changed")
    source = ROOT/"data/advanced_validation"
    raw,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","ordered")]
    bc = pd.read_pickle(ROOT/"data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:len(raw)]
    views = {name:ordered.join(pd.read_pickle(cache/(name+"_inputs.pkl")).iloc[:len(raw)]) for name in plan["heads"]}
    info = pd.read_pickle(cache/"motion_info.pkl").iloc[:len(raw)]
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv";labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X");np.testing.assert_array_equal(public,np.arange(11185,18685))
    split_path = source/"splits.json";splits = json.loads(split_path.read_text())
    (output/"splits.json").write_bytes(split_path.read_bytes())
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in (gt,split_path,source/"raw.pkl",source/"ordered.pkl",cache/"manifest.json")}
    rows,details,segments = [],[],[]
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol;folder.mkdir()
        folds = [(s["block"],np.array(s["train"]),np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        blocks = [(int(val[0]),int(val[-1])+1) for _,_,val in folds]
        arrays,artifacts = reference_arrays(protocol,42,len(raw));hashes.update(artifacts)
        reference = decode_broadcast_closure(*arrays,bc,raw,blocks)[public]
        old = root/"baseline_confirmation"/protocol/"oof_predictions.csv";hashes[str(old.relative_to(ROOT))] = digest(old)
        np.testing.assert_array_equal(reference,pd.read_csv(old).broadcast_closure_seed42)
        table = pd.DataFrame(dict(frame=public,ground_truth=labels[public],current_seed42=reference))
        record_predictions(output,protocol,"current_seed42",public,labels,folds,reference,rows,details,segments)
        add_overlap_metrics(rows,segments,table,protocol,"current_seed42")
        for head in plan["heads"]:
            probability = np.full((len(raw),3),np.nan)
            for block,train,val in folds:
                print(protocol,head,"block",block,flush=True)
                model = fit_renewal(views[head],labels,train,"frame",42)
                model.save_model(folder/f"{head}_seed42_block{block}.ubj")
                probability[val] = predict_models([model],views[head].iloc[val])
            np.save(folder/f"{head}_seed42_probabilities.npy",probability[public])
            for mode in plan["screen"]["decoders"]:
                prediction = decode_pair_candidate(*arrays,probability,bc,info,raw,blocks,head,mode)[public]
                name = f"{head}_{mode}_seed42";table[name] = prediction
                record_predictions(output,protocol,name,public,labels,folds,prediction,rows,details,segments)
                add_overlap_metrics(rows,segments,table,protocol,name)
        add_short_metrics(rows,segments)
        table.to_csv(folder/"oof_predictions.csv",index=False)
        pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
    names = ("research_pair_motion.py","pair_candidate.py","pair_motion_inference.py","pair_motion.py",
        "research_control_renewal.py","temporal_candidate.py","weak_ownership.py","stoppage_consensus.py","broadcast_refinement.py")
    manifest = dict(plan_sha256=digest(plan_path),source_sha256={name:digest(ROOT/"experiments"/name) for name in names},
        input_sha256=hashes,feature_columns={name:list(value) for name,value in views.items()},new_models=52,
        training_only_public_targets=True,original_seven_model_reference_reproduced=True)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol","model","spellF1","macroF1","false_dead_frames","short_live_mean_best_iou","short_live_matched_iou50"]].round(6).to_string(index=False),flush=True)


if __name__ == "__main__": main()
