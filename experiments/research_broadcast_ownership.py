"""Prospective classifier screen using the observation-selected broadcast view."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from broadcast_consensus import decode_broadcast_consensus
from build_temporal_candidate import digest
from control_refinement import fast_columns,fit_control,predict_control
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_control_renewal import fit_renewal
from research_stoppages import record_predictions
from stoppage_consensus import decode_stoppage_consensus
from temporal_candidate import predict_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/control_broadcast_v1/screen")
    parser.add_argument("--inputs",type=Path,default=ROOT/"data/control_broadcast_v1/inputs")
    args = parser.parse_args()
    output,cache = args.output,args.inputs
    output.mkdir(parents=True,exist_ok=False)
    plan_path = ROOT/"experiments/BROADCAST_ALIGNMENT_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    cache_manifest = json.loads((cache/"manifest.json").read_text())
    if cache_manifest["plan_sha256"] != digest(plan_path): raise ValueError("Alignment plan changed")
    if cache_manifest["selected_profile"] is None: raise ValueError("Neither alignment profile passed physical feasibility")
    if not cache_manifest["no_target_labels_read"] or not cache_manifest["actual_padded_feature_parity"]:
        raise ValueError("Alignment did not pass prerequisite checks")
    for name,expected in cache_manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected: raise ValueError("Alignment source changed")
    for name,expected in cache_manifest["cache_sha256"].items():
        if digest(cache/name) != expected: raise ValueError("Alignment cache changed")
    source = ROOT/"data/advanced_validation"
    raw,base,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","base","ordered")]
    same,combined = [pd.read_pickle(cache/(name+"_inputs.pkl")).iloc[:len(raw)] for name in ("same","combined")]
    views = dict(broadcast_fast_same=base[fast_columns(base)].join(same),
                 broadcast_fast_cross=base[fast_columns(base)].join(combined),
                 broadcast_joint_frame=ordered.join(combined),broadcast_joint_runs=ordered.join(combined))
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(public,np.arange(11185,18685))
    reference = ROOT/"data/control_broadcast_v1/baseline_confirmation"
    segments = pd.read_csv(reference/"segment_diagnostics.csv")
    short = segments[(segments.protocol == "buffered_minutes") & (segments.model == "stoppage_consensus_seed42") & (segments.label != "D") & (segments.frames <= 125)]
    cases = [("all_public_live",np.flatnonzero(np.isin(labels[:len(raw)],["H","A"])))]
    cases += [(f"short_{int(s.frames)}",np.arange(int(s.start),int(s.end)+1)) for s in short.itertuples()]
    diagnostics = []
    for view,features in (("same",same),("cross",combined)):
        nearest = features[[prefix+f"bc_{view}_distance_1" for prefix in ("h__","a__")]].to_numpy()
        observed = np.isfinite(nearest).all(axis=1)
        prediction = np.where(nearest[:,0] <= nearest[:,1],"H","A")
        for case,frames in cases:
            seen = observed[frames]
            diagnostics.append(dict(view=view,case=case,observed_fraction=float(seen.mean()),
                correct_observed_fraction=float(np.mean(seen & (prediction[frames] == labels[frames]))),
                accuracy_when_observed=float(np.mean(prediction[frames][seen] == labels[frames][seen])) if seen.any() else None))
    pd.DataFrame(diagnostics).to_csv(output/"nearest_team_diagnostics.csv",index=False)
    print("Second-camera nearest-team diagnostics",flush=True)
    print(pd.DataFrame(diagnostics).round(4).to_string(index=False),flush=True)
    splits_path = source/"splits.json"
    splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in
              (gt,splits_path,source/"raw.pkl",source/"base.pkl",source/"ordered.pkl",cache/"manifest.json")}
    rows,details,spans = [],[],[]
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"],np.array(s["train"]),np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        blocks = [(int(v[0]),int(v[-1])+1) for _,_,v in folds]
        ref_path = reference/protocol/"oof_predictions.csv"
        hashes[str(ref_path.relative_to(ROOT))] = digest(ref_path)
        original = pd.read_csv(ref_path)
        table = pd.DataFrame({"frame":public,"ground_truth":labels[public]})
        paths = [source/protocol/f"{name}_seed42_probabilities.npy" for name in ("ordered_xgboost_frame","previous_ensemble")]
        paths += [ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/"fast_seed42_probabilities.npy",
                  reference/protocol/"heads_seed42_probabilities.npy"]
        arrays = []
        for path,width in zip(paths,(3,3,2,2)):
            hashes[str(path.relative_to(ROOT))] = digest(path)
            value = np.full((len(raw),width),np.nan)
            value[public] = np.load(path)
            arrays.append(value)
        selected = decode_stoppage_consensus(*arrays,raw,blocks)[public]
        np.testing.assert_array_equal(selected,original.stoppage_consensus_seed42)
        table["current_seed42"] = selected
        record_predictions(output,protocol,"current_seed42",public,labels,folds,selected,rows,details,spans)
        add_overlap_metrics(rows,spans,table,protocol,"current_seed42")
        for name in plan["screen"]["models"]:
            joint = "joint" in name
            x = views[name]
            probability = np.full((len(raw),3 if joint else 2),np.nan)
            for block,train,validation in folds:
                print(protocol,name,"block",block,flush=True)
                model = fit_renewal(x,labels,train,"runs" if name.endswith("runs") else "frame",42) if joint else fit_control(x,labels,train,42)
                model.save_model(folder/f"{name}_seed42_block{block}.ubj")
                probability[validation] = predict_models([model],x.iloc[validation]) if joint else predict_control(model,x.iloc[validation])
            np.save(folder/f"{name}_seed42_probabilities.npy",probability[public])
            for mode in plan["screen"]["decoders"]:
                method = f"{name}_{mode}_seed42"
                prediction = decode_broadcast_consensus(*arrays,probability,combined,raw,blocks,joint,mode)[public]
                table[method] = prediction
                record_predictions(output,protocol,method,public,labels,folds,prediction,rows,details,spans)
                add_overlap_metrics(rows,spans,table,protocol,method)
            add_short_metrics(rows,spans)
            table.to_csv(folder/"oof_predictions.csv",index=False)
            pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
            pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
            pd.DataFrame(spans).to_csv(output/"segment_diagnostics.csv",index=False)
    names = ("research_broadcast_ownership.py","broadcast_consensus.py","broadcast_features.py","broadcast_alignment.py",
             "research_control_renewal.py","control_refinement.py","stoppage_consensus.py")
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=digest(plan_path),input_sha256=hashes,
        source_sha256={name:digest(ROOT/"experiments"/name) for name in names},selected_profile=cache_manifest["selected_profile"],
        reference_decoder_parity=True,feature_columns={name:list(x) for name,x in views.items()}),indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol","model","spellF1","macroF1","false_dead_frames","dead_recall","short_live_mean_best_iou","short_live_matched_iou50"]].round(6).to_string(index=False))


if __name__ == "__main__": main()
