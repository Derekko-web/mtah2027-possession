"""Select compact XGBoost complexity with inner buffered validation only.

Four fixed model configurations are compared on inner held-out minutes from the
outer training pool. The outer minute and its ten-second buffers are absent from
all inner training/validation. Choose mean log loss, not outer spellF1. Previously
developed feature/context/postprocessing choices are unchanged. Development is
adaptive; outer scores remain same-match evidence, not untouched confirmation.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import gap_context,apply_context
from rich_features import swap_teams
from transition_penalty import decode
from xgboost_baseline import CLASSES,probability_smoothing

CONFIGS={
    'depth2_balanced':dict(depth=2,trees=120,balance=True,child=15),
    'depth3_balanced':dict(depth=3,trees=160,balance=True,child=15),
    'depth3_unweighted':dict(depth=3,trees=160,balance=False,child=15),
    'depth4_balanced':dict(depth=4,trees=160,balance=True,child=25),
}


def fit(inputs,labels,train,config):
    y=np.array([np.flatnonzero(CLASSES==label)[0] for label in labels[train]])
    x=pd.concat([inputs.iloc[train],swap_teams(inputs.iloc[train])],ignore_index=True)
    target=np.r_[y,np.where(y==0,1,np.where(y==1,0,2))]
    if not np.array_equal(np.unique(target),np.arange(3)):
        raise ValueError('Inner training lacks a target class')
    weight=len(target)/(3*np.bincount(target)[target]) if config['balance'] else None
    model=XGBClassifier(objective='multi:softprob',num_class=3,eval_metric='mlogloss',tree_method='hist',
                        max_depth=config['depth'],n_estimators=config['trees'],learning_rate=.04,
                        reg_lambda=10,min_child_weight=config['child'],subsample=.8,colsample_bytree=.9,
                        random_state=42,n_jobs=2)
    model.fit(x,target,sample_weight=weight)
    return model


def predict(model,inputs,frames):
    return (model.predict_proba(inputs.iloc[frames])+
            model.predict_proba(swap_teams(inputs.iloc[frames]))[:,[1,0,2]])/2


def main():
    output=ROOT/'data/nested_baseline';output.mkdir(exist_ok=True)
    inputs=pd.read_pickle(ROOT/'data/kinematics_baseline/input_features.pkl')
    raw=pd.read_pickle(ROOT/'data/rich_baseline/input_features.pkl')
    gt=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv');labels=gt.label.to_numpy()
    public=np.flatnonzero(labels!='X');folds=buffered_blocks(int(public[0]),int(public[-1]),25)
    probability=np.full((len(public),3),np.nan);selection=[]
    for outer_minute,outer_train,outer_validation in folds:
        print('Outer fold',outer_minute,flush=True)
        losses={}
        for name,config in CONFIGS.items():
            inner_losses=[]
            for inner_minute,inner_train,inner_validation in folds:
                if inner_minute==outer_minute:continue
                train=np.intersect1d(outer_train,inner_train)
                validation=np.intersect1d(outer_train,inner_validation)
                if not len(train) or len(validation)<250:continue
                assert not np.intersect1d(train,outer_validation).size
                assert not np.intersect1d(validation,outer_validation).size
                model=fit(inputs,labels,train,config)
                p=predict(model,inputs,validation)
                target=np.array([np.flatnonzero(CLASSES==label)[0] for label in labels[validation]])
                inner_losses.append(log_loss(target,p,labels=[0,1,2]))
            losses[name]=float(np.mean(inner_losses))
        chosen=min(losses,key=losses.get)
        print('Selected',chosen,'inner losses',losses,flush=True)
        selection.append(dict(outer_minute=outer_minute,chosen=chosen,inner_log_losses=losses))
        model=fit(inputs,labels,outer_train,CONFIGS[chosen])
        probability[outer_validation-public[0]]=predict(model,inputs,outer_validation)
        model.save_model(output/f'fold_{outer_minute}.ubj')
    np.save(output/'oof_probabilities.npy',probability)
    side=gap_context(raw)[public]
    table=pd.DataFrame({'frame':public,'ground_truth':labels[public]});scores=[];detail=[]
    for momentum in [False,True]:
        p=apply_context(probability,side,np.zeros(len(public)),False) if momentum else probability
        prediction=np.empty(len(public),dtype='<U1');name='nested_momentum' if momentum else 'nested'
        for minute,_,validation in folds:
            ix=validation-public[0];prediction[ix]=CLASSES[decode(probability_smoothing(p[ix]))]
            result=pm.evaluate(pm.label_codes(labels[validation]),pm.label_codes(prediction[ix]))
            detail.append(dict(model=name,minute=minute,spellF1=result['spell_f1_mean']))
        table[name]=prediction
        result=pm.evaluate(pm.label_codes(labels[public]),pm.label_codes(prediction),bootstrap_n=1000)
        scores.append(dict(model=name,spellF1=result['spell_f1_mean'],macroF1=result['macro_f1'],possessions=result['n_pred_spells']))
    table.to_csv(output/'oof_predictions.csv',index=False);pd.DataFrame(scores).to_csv(output/'oof_scores.csv',index=False);pd.DataFrame(detail).to_csv(output/'fold_scores.csv',index=False)
    (output/'selection.json').write_text(json.dumps(dict(configs=CONFIGS,selection=selection),indent=2)+'\n')
    print(pd.DataFrame(detail).pivot(index='model',columns='minute',values='spellF1').round(4));print(pd.DataFrame(scores).to_string(index=False))


if __name__=='__main__':main()
