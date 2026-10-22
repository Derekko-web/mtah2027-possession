"""One controlled compact XGBoost experiment adding ball-motion detail."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from ball_kinematics import build_kinematics
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import gap_context, apply_context
from rich_features import swap_teams
from relative_motion import build_relative_motion
from train_factorized_baseline import compact_columns
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relative-motion',action='store_true')
    parser.add_argument('--segment-weights',action='store_true',
                        help='weight contiguous training segments equally instead of classes')
    args=parser.parse_args()
    folder='relative_motion_baseline' if args.relative_motion else 'kinematics_baseline'
    if args.segment_weights:folder+='_segment_weights'
    output = ROOT / 'data' / folder
    output.mkdir(exist_ok=True)
    x = pd.read_pickle(ROOT / "data/rich_baseline/input_features.pkl")
    inputs = x[compact_columns(x)].join(build_kinematics(len(x)))
    if args.relative_motion:
        inputs=inputs.join(build_relative_motion(len(x)))
    inputs.to_pickle(output / "input_features.pkl")
    gt = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    labels = gt.label.to_numpy()
    public = np.flatnonzero(labels != "X")
    folds = buffered_blocks(int(public[0]),int(public[-1]),25)
    p = np.full((len(public),3),np.nan)
    for minute, train, validation in folds:
        print("Kinematics fold",minute,flush=True)
        y = np.array([np.flatnonzero(CLASSES == label)[0] for label in labels[train]])
        train_x = pd.concat([inputs.iloc[train],swap_teams(inputs.iloc[train])])
        target = np.r_[y,np.where(y==0,1,np.where(y==1,0,2))]
        if args.segment_weights:
            # Derive segments strictly within this fold's eligible training rows.
            # Do not inspect validation labels to recover whole-possession lengths.
            boundaries=np.r_[True,(np.diff(train)!=1)|(y[1:]!=y[:-1])]
            segment=np.cumsum(boundaries)-1
            lengths=np.bincount(segment)
            original_weight=1/lengths[segment]
            weight=np.r_[original_weight,original_weight]
            weight/=weight.mean()
        else:
            weight = len(target)/(3*np.bincount(target)[target])
        model = XGBClassifier(objective="multi:softprob",num_class=3,eval_metric="mlogloss",
                              tree_method="hist",max_depth=2,n_estimators=120,learning_rate=.04,
                              reg_lambda=10,min_child_weight=15,subsample=.8,colsample_bytree=.9,
                              random_state=42,n_jobs=2)
        model.fit(train_x,target,sample_weight=weight)
        p[validation-public[0]]=(model.predict_proba(inputs.iloc[validation])+
            model.predict_proba(swap_teams(inputs.iloc[validation]))[:,[1,0,2]])/2
        model.save_model(output / f"fold_{minute}.ubj")
    np.save(output / "oof_probabilities.npy",p)
    side = gap_context(x)[public]
    predictions = pd.DataFrame({"frame":public,"ground_truth":labels[public]})
    scores,details=[],[]
    for momentum in [False,True]:
        probability=apply_context(p,side,np.zeros(len(p)),False) if momentum else p
        family='relative_motion' if args.relative_motion else 'kinematics'
        if args.segment_weights:family+='_segment_weights'
        name=family+'_momentum' if momentum else family
        prediction=np.empty(len(public),dtype="<U1")
        for minute,_,validation in folds:
            ix=validation-public[0]
            prediction[ix]=CLASSES[decode(probability_smoothing(probability[ix]))]
            result=pm.evaluate(pm.label_codes(labels[validation]),pm.label_codes(prediction[ix]))
            details.append(dict(model=name,minute=minute,spellF1=result["spell_f1_mean"]))
        predictions[name]=prediction
        result=pm.evaluate(pm.label_codes(labels[public]),pm.label_codes(prediction),bootstrap_n=1000)
        scores.append(dict(model=name,spellF1=result["spell_f1_mean"],macroF1=result["macro_f1"],possessions=result["n_pred_spells"]))
    predictions.to_csv(output / "oof_predictions.csv",index=False)
    pd.DataFrame(scores).to_csv(output / "oof_scores.csv",index=False)
    pd.DataFrame(details).to_csv(output / "fold_scores.csv",index=False)
    print(pd.DataFrame(details).pivot(index="model",columns="minute",values="spellF1").round(4))
    print(pd.DataFrame(scores).to_string(index=False))


if __name__=="__main__":
    main()
