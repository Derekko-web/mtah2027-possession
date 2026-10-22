"""Reproduce the selected equal-objective ensemble's validation and diagnostics."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import gap_context,apply_context
from transition_penalty import decode
from xgboost_baseline import CLASSES,probability_smoothing


def paired_possession_draws(truth,prediction,picks):
    runs=pm.valid_runs(truth);gt=pm.possession_spells(truth,runs)
    spans=pm.possession_spells(prediction,runs);overlap=pm.spell_overlaps(gt,spans)
    flags=np.vstack([pm.matched_gt_flags(overlap,len(gt),t) for t in pm.SPELL_IOUS])
    tp=flags[:,picks].sum(axis=2);fp=np.maximum(0,len(spans)-tp);fn=len(gt)-tp
    return np.mean(2*tp/np.maximum(2*tp+fp+fn,1),axis=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'data/selected_evaluation')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    output=ROOT/'data/selected_evaluation';output.mkdir(exist_ok=True)
    gt=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv');labels=gt.label.to_numpy()
    public=np.flatnonzero(labels!='X');truth=pm.label_codes(labels[public])
    folds=buffered_blocks(int(public[0]),int(public[-1]),25)
    raw=pd.read_pickle(ROOT/'data/rich_baseline/input_features.pkl')
    a=np.load(ROOT/'data/relative_motion_baseline/oof_probabilities.npy')
    b=np.load(ROOT/'data/relative_motion_baseline_segment_weights/oof_probabilities.npy')
    if a.shape!=(7500,3) or b.shape!=a.shape or not np.isfinite(a+b).all():
        raise ValueError('Incomplete out-of-fold probabilities')
    probability=apply_context((a+b)/2,gap_context(raw)[public],np.zeros(len(public)),False)
    prediction=np.empty(len(public),dtype='<U1')
    for _,_,validation in folds:
        ix=validation-public[0];prediction[ix]=CLASSES[decode(probability_smoothing(probability[ix]))]
    table=pd.DataFrame({'frame':public,'ground_truth':labels[public],'selected_ensemble':prediction})
    methods={'selected_ensemble':prediction}
    for variant in ['smooth50','smooth125']:
        methods['organiser_'+variant]=pm.read_label_csv(
            ROOT/f'evaluation/results/baseline_{variant}.csv').label.to_numpy()[public]
        table['organiser_'+variant]=methods['organiser_'+variant]
    table.to_csv(output/'oof_predictions.csv',index=False)
    scores=[];per_fold=[]
    for name,pred in methods.items():
        result=pm.evaluate(truth,pm.label_codes(pred),bootstrap_n=1000)
        for minute,_,validation in folds:
            ix=validation-public[0]
            detail=pm.evaluate(truth[ix],pm.label_codes(pred[ix]))
            per_fold.append(dict(model=name,minute=minute,spellF1=detail['spell_f1_mean']))
        scores.append(dict(model=name,stitched_oof_spellF1=result['spell_f1_mean'],
                           macroF1=result['macro_f1'],predicted_possessions=result['n_pred_spells']))
    folds_df=pd.DataFrame(per_fold)
    scores=pd.DataFrame(scores)
    scores['mean_minute_spellF1']=scores.model.map(folds_df.groupby('model').spellF1.mean())
    scores.to_csv(args.out/'selected-model-comparison.csv',index=False)
    folds_df.to_csv(args.out/'selected-model-folds.csv',index=False)
    rng=np.random.default_rng(42);n=17
    if len(pm.possession_spells(truth,pm.valid_runs(truth)))!=n:raise ValueError('Unexpected public possession count')
    picks=rng.integers(0,n,size=(20000,n));minute_picks=rng.integers(0,5,size=(20000,5))
    selected=paired_possession_draws(truth,pm.label_codes(prediction),picks)
    selected_folds=folds_df[folds_df.model=='selected_ensemble'].sort_values('minute').spellF1.to_numpy()
    intervals={}
    for name,pred in methods.items():
        if name=='selected_ensemble':continue
        delta=selected-paired_possession_draws(truth,pm.label_codes(pred),picks)
        ref_folds=folds_df[folds_df.model==name].sort_values('minute').spellF1.to_numpy()
        minute_delta=selected_folds-ref_folds
        intervals[name]=dict(
            paired_possession_delta_95=np.quantile(delta,[.025,.975]).tolist(),
            paired_minute_delta_95=np.quantile(minute_delta[minute_picks].mean(axis=1),[.025,.975]).tolist(),
            fold_deltas=minute_delta.tolist(),
        )
    (args.out/'selected-model-uncertainty.json').write_text(json.dumps(dict(
        bootstrap_draws=20000,seed=42,comparisons=intervals,
        warning='Adaptive same-match development, overlapping training sets, five folds and 17 possessions; '
                'intervals are diagnostic, not independent confirmatory significance tests.',
    ),indent=2)+'\n')
    # A valid full-length CSV containing only out-of-fold public predictions.
    oof=np.full(len(labels),'X',dtype='<U1');oof[public]=prediction
    oof_table=pd.DataFrame({'frame':np.arange(len(labels)),'label':oof})
    if pm.validate_submission(oof_table,len(labels)):raise ValueError('Invalid OOF format')
    oof_table.to_csv(output/'oof_public_only.csv',index=False)
    colors={'H':'#3178b5','A':'#e5a13d','D':'#c5cbd1'}
    fig,axes=plt.subplots(4,1,sharex=True,figsize=(13,5.3))
    for ax,column,label in zip(axes,['ground_truth','organiser_smooth50','organiser_smooth125','selected_ensemble'],
                               ['Answer key','Baseline smooth50','Baseline smooth125','Selected ensemble']):
        groups=table[column].ne(table[column].shift()).cumsum()
        for _,block in table.groupby(groups):
            left=(int(block.frame.iloc[0])-public[0])/25;right=(int(block.frame.iloc[-1])+1-public[0])/25
            state=block[column].iloc[0];ax.axvspan(left,right,color=colors[state],linewidth=0)
            if right-left>5:ax.text((left+right)/2,.5,state,ha='center',va='center',transform=ax.get_xaxis_transform(),fontsize=9)
        for second in [60,120,180,240]:ax.axvline(second,color='#222',linestyle=':',linewidth=.7)
        ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=12);ax.set_yticks([])
        ax.spines[['top','right','left']].set_visible(False);ax.set_xlim(0,300)
    axes[-1].set_xlabel('Seconds into the public five minutes; dotted lines separate held-out blocks')
    fig.suptitle('Out-of-fold predictions versus the supplied baselines',x=.03,ha='left',fontsize=17)
    fig.legend(handles=[Patch(color=colors[k],label=v) for k,v in [('H','Barcelona'),('A','Athletic'),('D','Stopped play')]],
               loc='upper right',bbox_to_anchor=(.98,.92),ncol=3,frameon=False)
    fig.text(.03,.015,'Every selected prediction comes from models trained outside its held-out minute and ten-second buffers. '
             'The same public window informed development.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.85));fig.savefig(args.out/'selected-model-timeline.png',dpi=160);plt.close(fig)
    print(scores.to_string(index=False));print(folds_df.pivot(index='model',columns='minute',values='spellF1').round(4))
    print('Diagnostic uncertainty:',json.dumps(intervals,indent=2))


if __name__=='__main__':main()
