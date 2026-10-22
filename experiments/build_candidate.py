"""Train the selected two-objective ensemble and generate a valid full-match CSV.

Uses only the public five-minute answer key for training. Do not interpret the
candidate's score on that same key as validation; use the saved out-of-fold
development report instead. No submission is uploaded by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from ball_kinematics import build_kinematics
from evaluate_physics_context import gap_context,apply_context
from games import GAME,half_video_blocks,source_frames
from relative_motion import build_relative_motion
from rich_features import build_features,swap_teams
from train_factorized_baseline import compact_columns
from transition_penalty import SWITCH_PENALTY,decode
from xgboost_baseline import CLASSES,probability_smoothing

PARAMS=dict(objective='multi:softprob',num_class=3,eval_metric='mlogloss',tree_method='hist',
            max_depth=2,n_estimators=120,learning_rate=.04,reg_lambda=10,min_child_weight=15,
            subsample=.8,colsample_bytree=.9,random_state=42,n_jobs=2)


def objective_weights(target,train,segment_balance):
    swapped=np.where(target==0,1,np.where(target==1,0,2))
    augmented=np.r_[target,swapped]
    if segment_balance:
        boundaries=np.r_[True,(np.diff(train)!=1)|(target[1:]!=target[:-1])]
        segment=np.cumsum(boundaries)-1
        weights=1/np.bincount(segment)[segment]
        weights=np.r_[weights,weights]
        weights/=weights.mean()
    else:
        weights=len(augmented)/(3*np.bincount(augmented)[augmented])
    return augmented,weights


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/candidate')
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    n_frames=source_frames()
    print('Building full-match features from provided tracking...',flush=True)
    raw=build_features(n_frames)
    inputs=raw[compact_columns(raw)].join(build_kinematics(n_frames)).join(build_relative_motion(n_frames))
    print('Feature table:',inputs.shape,flush=True)
    gt_path=ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    gt=pm.read_label_csv(gt_path);labels=gt.label.to_numpy();train=np.flatnonzero(labels!='X')
    if len(train)!=7500:raise ValueError('Expected exactly 7,500 public labelled training frames')
    target=np.array([np.flatnonzero(CLASSES==label)[0] for label in labels[train]])
    x_aug=pd.concat([inputs.iloc[train],swap_teams(inputs.iloc[train])],ignore_index=True)
    probabilities=[]
    for name,segment_balance in [('frame_model',False),('segment_model',True)]:
        print('Fitting',name,'on public labels only...',flush=True)
        augmented,weights=objective_weights(target,train,segment_balance)
        model=XGBClassifier(**PARAMS);model.fit(x_aug,augmented,sample_weight=weights)
        model.save_model(output/(name+'.ubj'))
        probability=np.empty((n_frames,3),dtype=np.float32)
        for start in range(0,n_frames,20000):
            block=inputs.iloc[start:start+20000]
            probability[start:start+len(block)]=(model.predict_proba(block)+
                model.predict_proba(swap_teams(block))[:,[1,0,2]])/2
        probabilities.append(probability)
    p=(probabilities[0]+probabilities[1])/2
    p=apply_context(p,gap_context(raw),np.zeros(n_frames),False)
    prediction=np.full(n_frames,'X',dtype='<U1')
    print('Decoding each half separately...',flush=True)
    for _,start,end in half_video_blocks(GAME):
        prediction[start:end+1]=CLASSES[decode(probability_smoothing(p[start:end+1]))]
    table=pd.DataFrame({'frame':np.arange(n_frames),'label':prediction})
    reason=pm.validate_submission(table,n_frames)
    if reason:raise ValueError(reason)
    candidate=output/'candidate.csv';table.to_csv(candidate,index=False)
    manifest=dict(
        model='Equal probability ensemble of frame-balanced and segment-balanced compact XGBoost',
        parameters=PARAMS,features=list(inputs.columns),
        public_training_frames=[int(train[0]),int(train[-1])],public_training_count=len(train),
        public_gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
        csv_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        training_classes=pd.Series(labels[train]).value_counts().to_dict(),
        output_class_counts=pd.Series(prediction).value_counts().to_dict(),
        postprocessing=dict(probability_smoothing_half_window_frames=25,switch_penalty=SWITCH_PENALTY,
                            occlusion_max_frames=237,occlusion_player_speed_threshold_m_s=1.5,
                            preceding_control_window_frames=13,context_mix_weight=.75),
        limitations=[
            'Full-match accuracy is unknown; private ground truth was not accessed.',
            'Reported validation uses buffered out-of-fold public predictions, not this training-fit file.',
            'The public five minutes were used repeatedly for adaptive development.',
            'This is a local candidate; nothing has been submitted or published.',
        ],
    )
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved',candidate)
    print('Class counts:',manifest['output_class_counts'])


if __name__=='__main__':main()
