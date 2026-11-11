"""One dependent, frozen consensus of saved public-only live/dead heads."""
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
from dfl_signal_audit import matching
from finalize_broadcast_closure import official_scores
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions
from research_temporal_dead import assess,check_inputs as check_parent,predict_variant,write_json

ROUND=ROOT/'data/control_temporal_consensus_v1'
PLAN=ROOT/'experiments/TEMPORAL_CONSENSUS_PLAN.json'
PARENT=ROOT/'data/control_temporal_dead_v1/screen'


def check_inputs():
    check_parent();freeze=ROUND/'protocol_freeze';plan=json.loads(PLAN.read_text())
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Consensus plan changed')
    for group in (plan['input_sha256'],json.loads((freeze/'implementation.json').read_text())):
        for name,expected in group.items():
            if digest(ROOT/name)!=expected:raise ValueError('Frozen consensus dependency changed: '+name)
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((freeze/'EXPERIMENT_LEDGER.csv').read_bytes()):raise ValueError('Ledger prefix differs')
    return plan


def run(replay=False):
    plan=check_inputs();saved=ROUND/'screen';output=ROUND/('screen_replay' if replay else 'screen')
    output.mkdir(parents=True,exist_ok=False);(output/'plan.json').write_bytes(PLAN.read_bytes())
    if replay:
        for name,expected in json.loads((saved/'manifest.json').read_text())['cache_sha256'].items():
            if digest(saved/name)!=expected:raise ValueError('Saved consensus differs')
    raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl');n=len(raw);public=np.arange(11185,18685)
    bc=pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:n]
    info=pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[:n]
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text())
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels!='X'),public)
    rows=[];details=[];segments=[];states={};changes={};hashes={}
    for protocol in plan['protocols']:
        folder=output/protocol;folder.mkdir()
        folds=[(s['block'],np.asarray(s['train']),np.asarray(s['validation'])) for s in splits if s['protocol']==protocol]
        blocks=[(int(v[0]),int(v[-1])+1) for _,_,v in folds]
        arrays,h=selected_arrays(protocol,42,n);hashes.update(h)
        reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
        expected=pd.read_csv(PARENT/protocol/'oof_predictions.csv').current_seed42.to_numpy()
        np.testing.assert_array_equal(reference[public],expected)
        a=np.load(PARENT/protocol/'frame_dead_probability.npy');b=np.load(PARENT/protocol/'temporal_dead_probability.npy')
        if a.shape!=(7500,) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid frozen heads')
        dead=np.minimum(a,b);np.save(folder/'consensus_dead_probability.npy',dead)
        prediction=predict_variant(arrays,dead,bc,info,raw,blocks,'union',public)
        np.testing.assert_array_equal(prediction=='X',reference=='X')
        shared=np.isin(prediction,['H','A'])&np.isin(reference,['H','A'])
        np.testing.assert_array_equal(prediction[shared],reference[shared])
        np.testing.assert_array_equal(prediction[reference=='D'],reference[reference=='D'])
        table=pd.DataFrame(dict(frame=public,ground_truth=labels[public]));name='consensus_union_seed42'
        for method,value in (('current_seed42',reference),(name,prediction)):
            table[method]=value[public]
            record_predictions(output,protocol,method,public,labels,folds,value[public],rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,method)
        table.to_csv(folder/'oof_predictions.csv',index=False)
        states[(protocol,name)]=(matching(labels[public],reference[public]),matching(labels[public],prediction[public]))
        changes[(protocol,name)]=int((reference[public]!=prediction[public]).sum())
        if replay and (folder/'oof_predictions.csv').read_bytes()!=(saved/protocol/'oof_predictions.csv').read_bytes():raise ValueError('Consensus inference replay differs')
    add_short_metrics(rows,segments);scores=pd.DataFrame(rows);diagnostics=pd.DataFrame(segments)
    scores.to_csv(output/'scores.csv',index=False);pd.DataFrame(details).to_csv(output/'fold_scores.csv',index=False)
    diagnostics.to_csv(output/'segment_diagnostics.csv',index=False)
    official=official_scores(output,scores);selection=assess(scores,diagnostics,official,states,changes,42)
    write_json(output/'selection.json',selection)
    if replay:
        for name in ('scores.csv','fold_scores.csv','segment_diagnostics.csv','official_comparison.csv','selection.json'):
            if (output/name).read_bytes()!=(saved/name).read_bytes():raise ValueError('Consensus score replay differs')
    else:write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),input_sha256=hashes,new_models=0,manual_labels_used=False,
        cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    check_inputs()
    print(scores[['protocol','model','spellF1','macroF1','false_dead_frames','dead_recall']].to_string(index=False),flush=True)
    print('Consensus passes:',selection['passed'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replay',action='store_true')
    run(parser.parse_args().replay)
