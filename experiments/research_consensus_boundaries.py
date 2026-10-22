"""Conserve current ownership at uncertain edges of accepted short proposals."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from consensus_control import consensus_team
from control_refinement import compose,contact_support,control_evidence,smooth
from evaluate_physics_context import apply_context
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES


def main():
    source=ROOT/"data/advanced_validation";confirmed=ROOT/"data/control_confirmation"
    output=ROOT/"data/consensus_boundary_research";output.mkdir(exist_ok=True)
    raw=pd.read_pickle(confirmed/"raw.pkl");side=gap_sides(raw)[11185:18685]
    support,evidence=contact_support(raw)[11185:18685],control_evidence(raw)[11185:18685]
    labels=pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public=np.arange(11185,18685);splits=json.loads((source/"splits.json").read_text())
    rows,details,segments=[],[],[]
    for protocol in ("buffered_minutes","purged_spells","buffered_100_seconds"):
        folder=output/protocol;folder.mkdir(exist_ok=True)
        folds=[(s["block"],np.array(s["train"]),np.array(s["validation"])) for s in splits if s["protocol"]==protocol]
        p=np.load(source/protocol/"ordered_xgboost_frame_seed42_probabilities.npy")
        prev=np.load(source/protocol/"previous_ensemble_seed42_probabilities.npy")
        fast=np.load(confirmed/protocol/"fast_seed42_probabilities.npy")
        p=apply_context(p,side,np.zeros(7500),False);prev=apply_context(prev,side,np.zeros(7500),False)
        fast=apply_context(np.column_stack((fast,np.zeros(7500))),side,np.zeros(7500),False)[:,:2]
        fast/=fast.sum(axis=1,keepdims=True)
        table=pd.DataFrame({"frame":public,"ground_truth":labels[public]})
        for mix in (.5,.75,.9,1.):
            prediction=np.empty(7500,dtype="<U1")
            for _,_,validation in folds:
                ix=validation-public[0];slow,old,short=smooth(p[ix],25),smooth(prev[ix],25),smooth(fast[ix],5)
                current=slow[:,:2]/slow[:,:2].sum(axis=1,keepdims=True)
                owner=consensus_team(slow,old,short,support[ix],.65)
                owner=mix*owner+(1-mix)*current
                dead=slow[:,2]*(1-.75*evidence[ix].max(axis=1))
                prediction[ix]=CLASSES[decode(compose(dead,owner))]
            name=f"mix{mix}";table[name]=prediction
            record_predictions(output,protocol,name,public,labels,folds,prediction,rows,details,segments)
        add_short_metrics(rows,segments);table.to_csv(folder/"oof_predictions.csv",index=False)
        pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":main()
