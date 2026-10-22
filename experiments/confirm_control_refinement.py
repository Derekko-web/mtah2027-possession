"""Frozen consensus variants against all current paired seeds and protocols."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from control_refinement import (compose, contact_support, control_evidence, fast_columns,
                                fit_control, predict_control, smooth)
from evaluate_physics_context import apply_context
from consensus_control import consensus_team
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import build_inputs, gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES

PLAN = ROOT/"experiments/CONTROL_CONFIRMATION_PLAN.json"


def main():
    source = ROOT/"data/advanced_validation"
    output = ROOT/"data/control_confirmation"
    output.mkdir(exist_ok=True)
    plan = json.loads(PLAN.read_text())
    (output/"plan.json").write_bytes(PLAN.read_bytes())
    (output/"plan_sha256.txt").write_text(hashlib.sha256(PLAN.read_bytes()).hexdigest()+"\n")
    print("Rebuilding observation context",flush=True)
    raw,base,ordered = build_inputs(18935)
    for name,table in (("raw",raw),("base",base),("ordered",ordered)):
        prior = pd.read_pickle(source/(name+".pkl"))
        pd.testing.assert_frame_equal(table,prior)
        table.to_pickle(output/(name+".pkl"))
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels!="X")
    side = gap_sides(raw)[public]
    support,evidence = contact_support(raw)[public],control_evidence(raw)[public]
    inputs = base[fast_columns(base)]
    splits = json.loads((source/"splits.json").read_text())
    rows,details,segments = [],[],[]
    for protocol in plan["protocols"]:
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        folds = [(s["block"],np.array(s["train"]),np.array(s["validation"]))
                 for s in splits if s["protocol"]==protocol]
        table = pd.DataFrame({"frame":public,"ground_truth":labels[public]})
        for seed in plan["seeds"]:
            team = np.full((7500,2),np.nan)
            for block,train,validation in folds:
                print(protocol,seed,"block",block,flush=True)
                model = fit_control(inputs,labels,train,seed)
                model.save_model(folder/f"fast_seed{seed}_block{block}.ubj")
                team[validation-public[0]] = predict_control(model,inputs.iloc[validation])
            np.save(folder/f"fast_seed{seed}_probabilities.npy",team)
            live = apply_context(np.column_stack((team,np.zeros(7500))),side,np.zeros(7500),False)[:,:2]
            live /= live.sum(axis=1,keepdims=True)
            p = np.load(source/protocol/f"ordered_xgboost_frame_seed{seed}_probabilities.npy")
            p = apply_context(p,side,np.zeros(7500),False)
            prev = np.load(source/protocol/f"previous_ensemble_seed{seed}_probabilities.npy")
            prev = apply_context(prev,side,np.zeros(7500),False)
            for name,config in {"current":None,**plan["finalists"]}.items():
                prediction = np.empty(7500,dtype="<U1")
                for _,_,validation in folds:
                    ix = validation-public[0]
                    slow,old,fast = smooth(p[ix],25),smooth(prev[ix],25),smooth(live[ix],5)
                    if config is None:
                        emissions = slow
                    else:
                        owner = consensus_team(slow,old,fast,support[ix],config["threshold"])
                        dead = slow[:,2]*(1-config["strength"]*evidence[ix].max(axis=1))
                        emissions = compose(dead,owner)
                    prediction[ix] = CLASSES[decode(emissions)]
                method = name+f"_seed{seed}"
                table[method] = prediction
                record_predictions(output,protocol,method,public,labels,folds,prediction,rows,details,segments)
            add_short_metrics(rows,segments)
            table.to_csv(folder/"oof_predictions.csv",index=False)
            pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
            pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
            pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":
    main()
