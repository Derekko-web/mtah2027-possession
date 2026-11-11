"""Frozen no-fit false-stoppage bridge, with shared and official score replay."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from dead_bridge import bridge_dead
from dfl_signal_audit import matching
from finalize_broadcast_closure import audit_manifest,official_scores
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions
from research_temporal_dead import assess,write_json
from stoppage_features import restart_features,load_tags

ROUND=ROOT/'data/control_dead_bridge_v1'
PLAN=ROOT/'experiments/DEAD_BRIDGE_PLAN.json'


def check_inputs():
    freeze=ROUND/'protocol_freeze';plan=json.loads(PLAN.read_text())
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Plan changed')
    for names in (plan['input_sha256'],json.loads((freeze/'protected_artifacts.json').read_text()),json.loads((freeze/'implementation.json').read_text())):
        for name,expected in names.items():
            if digest(ROOT/name)!=expected:raise ValueError('Frozen input/source changed: '+name)
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((freeze/'EXPERIMENT_LEDGER.csv').read_bytes()):raise ValueError('Ledger prefix changed')
    for name,target in json.loads((freeze/'symlinks.json').read_text()).items():
        p=ROOT/name
        if not p.is_symlink() or str(p.readlink())!=target:raise ValueError('Source link changed')
    return plan


def evidence(raw):
    moving=raw.ball_displacement_1s.to_numpy()>4.5
    close=[]
    for own,other in (('h__','a__'),('a__','h__')):
        close.append((raw[own+'distance_1'].to_numpy()<2)&(raw[other+'distance_1'].to_numpy()-raw[own+'distance_1'].to_numpy()>.5)&moving)
    restart=restart_features(len(raw),load_tags()).restart_all_active.to_numpy()>0
    return raw.ball_seen.to_numpy(),np.column_stack(close),restart


def run(seed,replay=False,output_override=None):
    plan=check_inputs();stage='screen' if seed==42 else f'confirmation_seed{seed}'
    if seed!=42 and not json.loads((ROUND/'screen/selection.json').read_text())['passed']:raise ValueError('No eligible seed42 rule')
    saved=ROUND/stage;output=output_override or ROUND/(stage+'_replay' if replay else stage)
    if replay:
        for name,expected in json.loads((saved/'manifest.json').read_text())['cache_sha256'].items():
            if digest(saved/name)!=expected:raise ValueError('Saved cache changed')
    output.mkdir(parents=True,exist_ok=False);(output/'plan.json').write_bytes(PLAN.read_bytes())
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl');n=len(raw);public=np.arange(11185,18685)
    seen,contact,restart=evidence(raw)
    camera=ROOT/'data/control_broadcast_v1/inputs';motion=ROOT/'data/control_pair_motion_v1'
    for p in (camera,motion/'inputs',motion/'dead_trim_confirmation'):audit_manifest(p)
    bc=pd.read_pickle(camera/'combined_inputs.pkl').iloc[:n];info=pd.read_pickle(motion/'inputs/motion_info.pkl').iloc[:n]
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels!='X'),public)
    rows=[];details=[];segments=[];states={};changes={};hashes={};counts=[]
    for protocol in plan['protocols']:
        folder=output/protocol;folder.mkdir()
        folds=[(s['block'],np.asarray(s['train']),np.asarray(s['validation'])) for s in splits if s['protocol']==protocol]
        blocks=[(int(v[0]),int(v[-1])+1) for _,_,v in folds]
        arrays,h=selected_arrays(protocol,seed,n);hashes.update(h)
        reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
        expected=pd.read_csv(motion/'dead_trim_confirmation'/protocol/'oof_predictions.csv')[f'motion_dead_trim_seed{seed}'].to_numpy()
        np.testing.assert_array_equal(reference[public],expected)
        prediction,trace=bridge_dead(reference,arrays[2],seen,contact,restart,blocks)
        np.testing.assert_array_equal(prediction[reference!='D'],reference[reference!='D'])
        np.testing.assert_array_equal(prediction=='X',reference=='X')
        table=pd.DataFrame(dict(frame=public,ground_truth=labels[public]))
        old_name=f'current_seed{seed}';name=f'bridge_seed{seed}'
        for method,value in ((old_name,reference),(name,prediction)):
            table[method]=value[public]
            record_predictions(output,protocol,method,public,labels,folds,value[public],rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,method)
        table.to_csv(folder/'oof_predictions.csv',index=False);write_json(folder/'trace.json',trace)
        states[(protocol,name)]=(matching(labels[public],reference[public]),matching(labels[public],prediction[public]))
        changes[(protocol,name)]=int((reference[public]!=prediction[public]).sum())
        counts.append(dict(protocol=protocol,proposed=len(trace),accepted=sum(t['accepted'] for t in trace),changed_frames=changes[(protocol,name)]))
        if replay and (folder/'oof_predictions.csv').read_bytes()!=(saved/protocol/'oof_predictions.csv').read_bytes():raise ValueError('Inference replay differs')
    add_short_metrics(rows,segments);scores=pd.DataFrame(rows);diagnostics=pd.DataFrame(segments)
    scores.to_csv(output/'scores.csv',index=False);pd.DataFrame(details).to_csv(output/'fold_scores.csv',index=False)
    diagnostics.to_csv(output/'segment_diagnostics.csv',index=False)
    official=official_scores(output,scores);selection=assess(scores,diagnostics,official,states,changes,seed)
    for p in selection['pairs']:
        s=scores[scores.protocol==p['protocol']].set_index('model')
        p['checks']['false_D']=bool(s.loc[p['model'],'false_dead_frames']<=s.loc[f'current_seed{seed}','false_dead_frames'])
    selection['passed']=all(all(p['checks'].values()) for p in selection['pairs'])
    selection['eligible']=[f'bridge_seed{seed}'] if selection['passed'] else []
    selection['selected']=selection['eligible'][0] if selection['passed'] else None
    write_json(output/'selection.json',selection);write_json(output/'proposal_counts.json',counts)
    if replay:
        for name in ('scores.csv','fold_scores.csv','segment_diagnostics.csv','official_comparison.csv','selection.json','proposal_counts.json'):
            if (output/name).read_bytes()!=(saved/name).read_bytes():raise ValueError('Score replay differs: '+name)
    else:write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),input_sha256=hashes,new_models=0,manual_labels_used=False,
        cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    check_inputs()
    print(scores[['protocol','model','spellF1','macroF1','false_dead_frames','dead_recall']].to_string(index=False),flush=True)
    print('Bridge passes:',selection['passed'],'proposal totals:',counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.seed,a.replay,a.output)
