"""R37 matched frozen-encoder feature transfer, checkpoint replay and gates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from player_representation import ROOT,ROUND,PLAN,PROTOCOLS,VIEWS,COLUMNS,check_inputs,audit_cache,digest,write_json,interpolate,live_overlay
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from build_candidate import objective_weights
from temporal_candidate import fit_models,predict_models
from research_ownership_graph import selected_arrays
from motion_dead_trim import decode_motion_dead_trim
from research_temporal_dead import assess
from research_stoppages import record_predictions
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from finalize_broadcast_closure import official_scores
from dfl_signal_audit import matching


def decode(arrays,probability,bc,info,raw,blocks):
    reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
    new=list(arrays);new[0]=probability
    return live_overlay(reference,decode_motion_dead_trim(*new,bc,info,raw,blocks))


def substantial(scores,segments,official,states,changes,seed):
    result=assess(scores,segments,official,states,changes,seed)
    for pair in result['pairs']:
        pair['checks'].pop('incremental_primary_gain',None)
        if pair['protocol']=='buffered_minutes':
            pair['checks']['substantial_primary_gain']=pair['historical_substantial_gain']
            pair['checks']['additional_short_match']=pair['historical_extra_short_match']
    eligible=[name for name in scores.model.unique() if name!=f'current_seed{seed}' and all(all(p['checks'].values()) for p in result['pairs'] if p['model']==name)]
    def key(name):
        pairs=[p for p in result['pairs'] if p['model']==name]
        return (-next(p['spellF1_delta'] for p in pairs if p['protocol']=='buffered_minutes'),-min(p['spellF1_delta'] for p in pairs if p['protocol']!='buffered_minutes'),sum(p['changed_frames'] for p in pairs),0 if name.startswith('pretrained') else 1)
    eligible.sort(key=key);result.update(passed=bool(eligible),eligible=eligible,selected=eligible[0] if eligible else None)
    return result


def run(seed,replay,override):
    check_inputs();audit_cache(ROUND/'inputs');audit_cache(ROUND/'encodings')
    views=list(VIEWS)
    if seed!=42:
        selection=json.loads((ROUND/'screen/selection.json').read_text())
        if not selection['passed']:raise ValueError('No eligible seed42 representation')
        views=[selection['selected'].rsplit('_seed',1)[0]]
    stage='screen' if seed==42 else f'confirmation_seed{seed}';saved=ROUND/stage
    if replay:audit_cache(saved)
    output=override or ROUND/(stage+'_replay' if replay else stage);output.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_bytes(PLAN.read_bytes())
    source=ROOT/'data/advanced_validation';raw=pd.read_pickle(source/'raw.pkl');ordered=pd.read_pickle(source/'ordered.pkl');n=len(raw)
    encoding=ROUND/'encodings';frames=np.load(encoding/'frames.npy');support=np.load(encoding/'eligible.npy')
    inputs={'control':ordered};eligible=None
    for view in views:
        z,mask=interpolate(frames,np.load(encoding/(view+'.npy')),support,n)
        if eligible is None:eligible=mask
        else:np.testing.assert_array_equal(mask,eligible)
        inputs[view]=ordered.join(pd.DataFrame(z,columns=COLUMNS,index=ordered.index))
    bc=pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:n]
    info=pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[:n]
    splits=json.loads((source/'splits.json').read_text());labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy();public=np.arange(11185,18685)
    np.testing.assert_array_equal(np.flatnonzero(labels!='X'),public)
    baselines={};probabilities={};models=0
    for protocol in PROTOCOLS:
        folds=[(s['block'],np.asarray(s['train']),np.asarray(s['validation'])) for s in splits if s['protocol']==protocol]
        blocks=[(int(val[0]),int(val[-1])+1) for _,_,val in folds];arrays,hashes=selected_arrays(protocol,seed,n)
        reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
        expected=pd.read_csv(ROOT/'data/control_pair_motion_v1/dead_trim_confirmation'/protocol/'oof_predictions.csv')[f'motion_dead_trim_seed{seed}'].to_numpy()
        np.testing.assert_array_equal(reference[public],expected);baselines[protocol]=(folds,blocks,arrays,reference)
        (output/protocol).mkdir()
    # Every original refit must pass before either new feature view is fitted.
    for view in ['control']+views:
        for protocol,(folds,blocks,arrays,reference) in baselines.items():
            probability=np.full((n,3),np.nan)
            for block,train,val in folds:
                if len(np.intersect1d(train,val)) or not np.isin(train,public).all():raise ValueError('Public exclusion split invalid')
                folder=output/protocol;filename=f'{view}_seed{seed}_block{block}.ubj'
                path=folder/filename;fitting=folder/(filename+'.fitting.npz')
                target=np.asarray([{'H':0,'A':1,'D':2}[v] for v in labels[train]])
                y,weight=objective_weights(target,train,False);masked=np.full_like(labels,'X');masked[train]=labels[train]
                target2=np.asarray([{'H':0,'A':1,'D':2}[v] for v in masked[train]]);y2,w2=objective_weights(target2,train,False)
                np.testing.assert_array_equal(y,y2);np.testing.assert_array_equal(weight,w2)
                np.savez_compressed(fitting,train=train,validation=val,target=y,weight=weight)
                if replay:
                    with np.load(saved/protocol/fitting.name) as prior:
                        for k,v in dict(train=train,validation=val,target=y,weight=weight).items():np.testing.assert_array_equal(prior[k],v)
                    model=XGBClassifier();model.load_model(saved/protocol/filename)
                else:
                    print('Fit',view,protocol,'block',block,flush=True)
                    model=fit_models('ordered_xgboost_frame',inputs[view],labels,train,seed)[0];model.save_model(path);models+=1
                    reloaded=XGBClassifier();reloaded.load_model(path)
                    np.testing.assert_array_equal(predict_models([model],inputs[view].iloc[val]),predict_models([reloaded],inputs[view].iloc[val]))
                probability[val]=predict_models([model],inputs[view].iloc[val])
            if view=='control':
                np.testing.assert_array_equal(probability[public],arrays[0][public])
                np.testing.assert_array_equal(decode(arrays,probability,bc,info,raw,blocks),reference)
            else:probability[public[~eligible[public]]]=arrays[0][public[~eligible[public]]]
            np.save(output/protocol/(view+'_probability.npy'),probability[public]);probabilities[(protocol,view)]=probability
            if replay:np.testing.assert_array_equal(probability[public],np.load(saved/protocol/(view+'_probability.npy')))
        if view=='control':print('All 13 exact original-head refits and complete pipeline identities pass',flush=True)
    rows=[];details=[];segments=[];states={};changes={}
    for protocol,(folds,blocks,arrays,reference) in baselines.items():
        table=pd.DataFrame(dict(frame=public,ground_truth=labels[public]));current=f'current_seed{seed}'
        predictions={current:reference}
        for view in ['control']+views:predictions[f'{view}_seed{seed}']=decode(arrays,probabilities[(protocol,view)],bc,info,raw,blocks)
        for name,prediction in predictions.items():
            np.testing.assert_array_equal(prediction[~np.isin(reference,['H','A'])],reference[~np.isin(reference,['H','A'])])
            table[name]=prediction[public]
            record_predictions(output,protocol,name,public,labels,folds,prediction[public],rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,name)
            if name!=current:
                states[(protocol,name)]=(matching(labels[public],reference[public]),matching(labels[public],prediction[public]));changes[(protocol,name)]=int((prediction[public]!=reference[public]).sum())
        table.to_csv(output/protocol/'oof_predictions.csv',index=False)
    add_short_metrics(rows,segments);scores=pd.DataFrame(rows);diagnostics=pd.DataFrame(segments)
    scores.to_csv(output/'scores.csv',index=False);diagnostics.to_csv(output/'segment_diagnostics.csv',index=False);pd.DataFrame(details).to_csv(output/'fold_scores.csv',index=False)
    official=official_scores(output,scores);selection=substantial(scores,diagnostics,official,states,changes,seed);write_json(output/'selection.json',selection)
    if replay:
        for name in ('scores.csv','segment_diagnostics.csv','fold_scores.csv','official_comparison.csv','selection.json'):
            if (output/name).read_bytes()!=(saved/name).read_bytes():raise ValueError('Score or gate replay differs: '+name)
        for protocol in PROTOCOLS:
            if (output/protocol/'oof_predictions.csv').read_bytes()!=(saved/protocol/'oof_predictions.csv').read_bytes():raise ValueError('OOF shared inference differs')
    else:write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),new_models=models,manual_labels_used=False,only_public_train_indices=True,control_identity_exact=True,
        input_sha256=json.loads(PLAN.read_text())['input_sha256'],cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    check_inputs()
    print(scores[['protocol','model','spellF1','macroF1','short_live_matched_iou50','false_dead_frames','dead_recall']].to_string(index=False),flush=True)
    print('Eligible:',selection['eligible'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43,44],default=42);p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args();run(args.seed,args.replay,args.output)
