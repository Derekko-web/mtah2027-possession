"""Frozen diagnostic of raw external D transfer; never modifies predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from research_temporal_dead import write_json
from research_dfl_dead_external import metrics


def main(output=None):
    folder=ROOT/'data/control_dfl_dead_signal_v1';plan_path=ROOT/'experiments/DFL_DEAD_SIGNAL_PLAN.json';freeze=folder/'protocol_freeze'
    plan=json.loads(plan_path.read_text())
    if plan_path.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Signal audit plan changed')
    for group in (plan['input_sha256'],json.loads((freeze/'protected_artifacts.json').read_text())):
        for name,expected in group.items():
            if digest(ROOT/name)!=expected:raise ValueError('Signal audit dependency changed')
    output=output or folder/'audit';output.mkdir(exist_ok=False)
    gt=pd.read_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv');public=np.flatnonzero(gt.label.to_numpy()!='X');labels=gt.label.to_numpy()[public]
    p=np.load(ROOT/'data/control_dfl_dead_v1/native_inputs/teacher_full.npy')[public];target=labels=='D'
    observations=np.load(ROOT/'data/control_dfl_dead_v1/native_inputs/native_observations.npz')
    seen=np.isfinite(observations['ball'][public]).all(axis=1)
    raw=metrics(np.where(target,0,1),p,seen);results={}
    for protocol in ('buffered_minutes','purged_spells','buffered_100_seconds'):
        table=pd.read_csv(ROOT/'data/control_dfl_dead_v1/public_trial'/protocol/'oof_predictions.csv')
        np.testing.assert_array_equal(table.frame,public);np.testing.assert_array_equal(table.ground_truth,labels)
        reference=table.current_seed42.to_numpy()=='D';proposed=p>=.8
        change=((p>=.8)|(p<=.2))&(proposed!=reference)
        corrected=int((change&(proposed==target)).sum());damage=int((change&(reference==target)).sum())
        precision=corrected/max(corrected+damage,1)
        results[protocol]=dict(proposals=int(change.sum()),corrected=corrected,damage=damage,net=corrected-damage,
            correction_precision=precision,passed=precision>=.75 and corrected-damage>=25)
    intervals=[];edges=np.flatnonzero(np.diff(np.r_[False,target,False]))
    for a,b in zip(edges[::2],edges[1::2]):intervals.append(dict(frames=int(b-a),raw_D_recall=float((p[a:b]>=.5).mean()),confident_D_fraction=float((p[a:b]>=.8).mean())))
    result=dict(passed=all(v['passed'] for v in results.values()),raw=raw,reference_corrections=results,all_D_interval_metrics=intervals,new_fits=0,new_possession_scores=0,diagnostic_only=True)
    write_json(output/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    main(p.parse_args().output)
