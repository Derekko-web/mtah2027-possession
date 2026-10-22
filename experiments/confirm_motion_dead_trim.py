"""Frozen nine-pair confirmation of unchanged motion evidence with D-only inference."""

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
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_control_renewal import fit_renewal
from research_pair_motion import reference_arrays
from research_stoppages import record_predictions
from temporal_candidate import predict_models


def main():
    root = ROOT/"data/control_pair_motion_v1";output = root/"dead_trim_confirmation";output.mkdir(exist_ok=False)
    plan_path = ROOT/"experiments/MOTION_DEAD_TRIM_PLAN.json";plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    for name,sha in plan["input_sha256"].items():
        if digest(ROOT/name) != sha: raise ValueError("Frozen dead-trim input differs")
    source = ROOT/"data/advanced_validation";raw,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","ordered")]
    features = pd.read_pickle(root/"inputs/broadcast_motion_inputs.pkl").iloc[:len(raw)]
    inputs = ordered.join(features)
    if len(inputs.columns) != 261: raise ValueError("Motion model schema differs")
    info = pd.read_pickle(root/"inputs/motion_info.pkl").iloc[:len(raw)]
    bc = pd.read_pickle(ROOT/"data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:len(raw)]
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv";labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X");np.testing.assert_array_equal(public,np.arange(11185,18685))
    splits = json.loads((source/"splits.json").read_text());(output/"splits.json").write_bytes((source/"splits.json").read_bytes())
    rows,details,segments,hashes = [],[],[],dict(plan["input_sha256"])
    for protocol in plan["protocols"]:
        folder = output/protocol;folder.mkdir()
        folds = [(s["block"],np.array(s["train"]),np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        blocks = [(int(v[0]),int(v[-1])+1) for _,_,v in folds]
        table = pd.DataFrame(dict(frame=public,ground_truth=labels[public]))
        original = pd.read_csv(root/"baseline_confirmation"/protocol/"oof_predictions.csv")
        for seed in plan["seeds"]:
            probability = np.full((len(raw),3),np.nan)
            for block,train,val in folds:
                print(protocol,seed,"motion joint block",block,flush=True)
                model = fit_renewal(inputs,labels,train,"frame",seed)
                model.save_model(folder/f"broadcast_motion_seed{seed}_block{block}.ubj")
                probability[val] = predict_models([model],inputs.iloc[val])
            np.save(folder/f"broadcast_motion_seed{seed}_probabilities.npy",probability[public])
            if seed == 42:
                np.testing.assert_array_equal(probability[public],np.load(root/"screen"/protocol/"broadcast_motion_seed42_probabilities.npy"))
            arrays,source_hashes = reference_arrays(protocol,seed,len(raw));hashes.update(source_hashes)
            reference = decode_broadcast_closure(*arrays,bc,raw,blocks)[public]
            np.testing.assert_array_equal(reference,original[f"broadcast_closure_seed{seed}"])
            for name in plan["methods"]:
                prediction = reference if name == "current" else decode_motion_dead_trim(*arrays,probability,bc,info,raw,blocks)[public]
                method = f"{name}_seed{seed}";table[method] = prediction
                record_predictions(output,protocol,method,public,labels,folds,prediction,rows,details,segments)
                add_overlap_metrics(rows,segments,table,protocol,method)
            add_short_metrics(rows,segments)
            table.to_csv(folder/"oof_predictions.csv",index=False)
            pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
            pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
            pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
    names = ("confirm_motion_dead_trim.py","motion_dead_trim.py","pair_candidate.py","pair_motion.py","pair_motion_inference.py","research_control_renewal.py","research_pair_motion.py","broadcast_refinement.py")
    manifest = dict(plan_sha256=digest(plan_path),source_sha256={name:digest(ROOT/"experiments"/name) for name in names},
        input_sha256=hashes,feature_columns=list(inputs),seed42_probability_parity=True,reference_nonD_preserved=True)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol","model","spellF1","macroF1","false_dead_frames","dead_recall","short_live_mean_best_iou"]].round(6).to_string(index=False),flush=True)


if __name__ == "__main__": main()
