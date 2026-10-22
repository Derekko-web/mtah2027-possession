"""Build the frozen consensus refinement; preserve every previous candidate."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from consensus_control import CONFIG,decode_consensus
from control_refinement import fast_columns,fit_control,predict_control
from games import GAME,half_video_blocks,require,source_frames
from temporal_candidate import build_inputs,fit_models,predict_models

SOURCES = ("build_control_candidate.py","consensus_control.py","control_refinement.py",
           "temporal_candidate.py","temporal_control.py","rich_features.py","ball_kinematics.py",
           "relative_motion.py","train_factorized_baseline.py","build_candidate.py",
           "evaluate_physics_context.py","transition_penalty.py","xgboost_baseline.py",
           "requirements.txt","requirements-research.txt","CONTROL_CONFIRMATION_PLAN.json")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/candidate_control_consensus")
    args=parser.parse_args();output=args.output.resolve()
    protected=[(ROOT/"data"/folder).resolve() for folder in
               ("candidate","candidate_temporal_xgb","candidate_temporal")]
    if output in protected:
        parser.error("Choose a new directory to preserve the earlier candidates")
    plan=json.loads((ROOT/"experiments/CONTROL_CONFIRMATION_PLAN.json").read_text())
    if plan["finalists"][plan["preferred"]]!=CONFIG:
        raise ValueError("Implementation differs from frozen confirmation configuration")
    output.mkdir(parents=True,exist_ok=True)
    n=source_frames()
    print("Building full-match observation features",flush=True)
    raw,base,ordered=build_inputs(n)
    for name,value in (("base",base),("ordered",ordered)):
        reference=ROOT/"data/control_confirmation"/(name+".pkl")
        if reference.exists():
            pd.testing.assert_frame_equal(value.iloc[11185:18685],pd.read_pickle(reference).iloc[11185:18685])
    gt_path=ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels=pm.read_label_csv(gt_path).label.to_numpy();train=np.flatnonzero(labels!="X")
    np.testing.assert_array_equal(train,np.arange(11185,18685))
    print("Fitting retained model, previous proposal models and live-only fast model",flush=True)
    main_models=fit_models("ordered_xgboost_frame",ordered,labels,train,42)
    proposal_models=fit_models("previous_ensemble",base,labels,train,42)
    fast_inputs=base[fast_columns(base)]
    fast_model=fit_control(fast_inputs,labels,train,42)
    model_paths=[]
    for model,name in zip(main_models+proposal_models+[fast_model],
                          ("current_model","previous_frame","previous_segment","fast_model")):
        path=output/(name+".ubj");model.save_model(path);model_paths.append(path)
    current,previous,fast=np.empty((n,3)),np.empty((n,3)),np.empty((n,2))
    for start in range(0,n,20000):
        end=min(n,start+20000)
        current[start:end]=predict_models(main_models,ordered.iloc[start:end])
        previous[start:end]=predict_models(proposal_models,base.iloc[start:end])
        fast[start:end]=predict_control(fast_model,fast_inputs.iloc[start:end])
    retained=ROOT/"data/candidate_temporal_xgb/probabilities.npy"
    if retained.exists():
        np.testing.assert_array_equal(current,np.load(retained))
        print("Retained model probabilities are exactly unchanged",flush=True)
    blocks=[(start,end+1) for _,start,end in half_video_blocks(GAME)]
    prediction=decode_consensus(current,previous,fast,raw,blocks,**CONFIG)
    table=pd.DataFrame({"frame":np.arange(n),"label":prediction})
    reason=pm.validate_submission(table,n)
    if reason:raise ValueError(reason)
    path=output/"candidate.csv";table.to_csv(path,index=False)
    for name,value in (("current",current),("previous",previous),("fast",fast)):
        np.save(output/(name+"_probabilities.npy"),value)
    manifest=dict(model="Consensus-supported short control with retained ordered XGBoost dead/live model",
                  seed=42,config=CONFIG,features=dict(current=list(ordered),previous=list(base),fast=list(fast_inputs)),
                  public_training_frames=[11185,18684],public_training_count=7500,
                  fast_training_count=int(np.isin(labels[train],["H","A"]).sum()),
                  csv_sha256=digest(path),public_gt_sha256=digest(gt_path),
                  source_sha256={str(Path("experiments")/name):digest(ROOT/"experiments"/name) for name in SOURCES},
                  shared_source_sha256={name:digest(ROOT/name) for name in
                                        ("games.py","starter/baseline.py","starter/load_smart_tagging.py")},
                  input_sha256={Path(p).name:digest(p) for p in require(["atd_meta","atd_raw","smart_tagging"])},
                  model_sha256={p.name:digest(p) for p in model_paths},
                  output_class_counts=pd.Series(prediction).value_counts().to_dict(),
                  limitations=["Adaptive public development on the same 17 true possessions; no independent/private validation.",
                               "Short proposals and all final boundaries are algorithmic, not manual frame patches.",
                               "Some short possessions and one mostly missed visible stoppage remain.",
                               "Offline inference with future observations and full-half decoding.",
                               "Nothing has been submitted or uploaded."])
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Saved",path,flush=True);print("CSV SHA256",manifest["csv_sha256"],flush=True)
    print("Class counts",manifest["output_class_counts"],flush=True)


if __name__=="__main__":main()
