"""Use sustained near-ball contacts to maintain a team-control state.

One fixed hypothesis: a team change requires six consecutive frames with its
nearest player within 1.5 m and at least 0.5 m closer than the other team. Unknown
or airborne-ball frames retain the previous contact state. All state resets at
the held-out block boundary. Dead-ball probabilities remain learned out of fold.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import gap_context,apply_context
from transition_penalty import decode
from xgboost_baseline import CLASSES,probability_smoothing


def contact_state(features):
    dh=features.h__distance_1.to_numpy();da=features.a__distance_1.to_numpy()
    candidate=np.full(len(features),-1,dtype=int)
    candidate[(dh<=1.5)&(da-dh>=.5)]=0
    candidate[(da<=1.5)&(dh-da>=.5)]=1
    current=-1;pending=-1;count=0
    state=np.full(len(features),-1,dtype=int)
    for frame,team in enumerate(candidate):
        if team<0:
            pending=-1;count=0
        else:
            count=count+1 if team==pending else 1
            pending=team
            if count>=6:
                current=team
        state[frame]=current
    # First few observations can use the first confirmed contact, at most 1 s ahead.
    known=np.flatnonzero(state>=0)
    if len(known) and known[0]<=25:
        state[:known[0]]=state[known[0]]
    return state


def main():
    output=ROOT/'data/contact_control';output.mkdir(exist_ok=True)
    features=pd.read_pickle(ROOT/'data/rich_baseline/input_features.pkl')
    gt=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv');labels=gt.label.to_numpy()
    public=np.flatnonzero(labels!='X');folds=buffered_blocks(int(public[0]),int(public[-1]),25)
    compact=pd.read_csv(ROOT/'data/factorized_baseline/oof_predictions.csv')
    probability_sets={
        'kinematics':np.load(ROOT/'data/kinematics_baseline/oof_probabilities.npy'),
        'compact':compact[[f'compact_multiclass_probability_{s}' for s in CLASSES]].to_numpy(),
        'forest':np.load(ROOT/'data/tree_ensembles/forest_compact_oof_probabilities.npy'),
    }
    predictions=pd.DataFrame({'frame':public,'ground_truth':labels[public]});rows=[];details=[]
    for name,p in probability_sets.items():
        for combine in [False,True]:
            method=name+('_blended_contact' if combine else '_contact')
            prediction=np.empty(len(public),dtype='<U1')
            for minute,_,validation in folds:
                ix=validation-public[0];block=p[ix].copy()
                state=contact_state(features.iloc[validation])
                ph=block[:,0]/np.maximum(block[:,:2].sum(axis=1),1e-12)
                known=state>=0
                contact=np.where(state==0,.95,.05)
                ph[known]=.5*ph[known]+.5*contact[known] if combine else contact[known]
                block=np.column_stack(((1-block[:,2])*ph,(1-block[:,2])*(1-ph),block[:,2]))
                prediction[ix]=CLASSES[decode(probability_smoothing(block))]
                r=pm.evaluate(pm.label_codes(labels[validation]),pm.label_codes(prediction[ix]));details.append(dict(model=method,minute=minute,spellF1=r['spell_f1_mean']))
            predictions[method]=prediction
            r=pm.evaluate(pm.label_codes(labels[public]),pm.label_codes(prediction),bootstrap_n=1000)
            rows.append(dict(model=method,spellF1=r['spell_f1_mean'],macroF1=r['macro_f1'],possessions=r['n_pred_spells']))
    predictions.to_csv(output/'oof_predictions.csv',index=False);pd.DataFrame(rows).to_csv(output/'oof_scores.csv',index=False);pd.DataFrame(details).to_csv(output/'fold_scores.csv',index=False)
    print(pd.DataFrame(details).pivot(index='model',columns='minute',values='spellF1').round(4));print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
