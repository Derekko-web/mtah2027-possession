"""Export qualified guarded boundaries over the preserved full-match recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from games import GAME,half_video_blocks,source_frames
from probability_guarded_boundaries import guarded_refine

ROUND=ROOT/'data/control_guarded_boundaries_v1'
PARENT=ROOT/'data/candidate_tabpfn_new_camera_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def build(output):
    plan=json.loads((ROUND/'plan.json').read_text())
    for name in ('experiments/probability_guarded_boundaries.py','experiments/possession_boundary_precision.py'):
        if digest(ROOT/name)!=plan['source_sha256'][name]:raise ValueError('Qualified inference source changed')
    if not json.loads((ROUND/'qualification_result.json').read_text())['passed']:raise ValueError('Incomplete all-nine boundary qualification')
    if json.loads((ROUND/'selection.json').read_text())['selected']!='guarded_release_approach25':raise ValueError('Unexpected finalist')
    reference=pd.read_csv(PARENT/'candidate.csv');n=source_frames()
    if len(reference)!=n or pm.validate_submission(reference,n):raise ValueError('Invalid parent candidate')
    key=str((PARENT/'candidate.csv').relative_to(ROOT))
    if digest(PARENT/'candidate.csv')!=plan['protected_sha256'][key]:raise ValueError('Frozen parent changed')
    paths=dict(raw=ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl',
        broadcast=ROOT/'data/control_remaining_away_v1/full_inputs/broadcast_extension.pkl',
        mapping=ROOT/'data/control_broadcast_v1/full_inputs/mapping.npz',
        fast=ROOT/'data/candidate_tabpfn_short_control_v1/fast_probabilities.npy',
        owner=ROOT/'data/candidate_tabpfn_short_control_v1/owner_probability.npy',
        recovery_events=PARENT/'recovery_events.json')
    raw,bc=pd.read_pickle(paths['raw']),pd.read_pickle(paths['broadcast'])
    fast,owner=np.load(paths['fast'],allow_pickle=False),np.load(paths['owner'],allow_pickle=False)
    old=np.load(paths['mapping'],allow_pickle=False)['frames']>=0
    events=json.loads(paths['recovery_events'].read_text())
    public=np.arange(11185,18685)
    prefix_raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl')
    prefix_bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl')
    pd.testing.assert_frame_equal(raw.iloc[public][['h__distance_1','a__distance_1']],prefix_raw.iloc[public][['h__distance_1','a__distance_1']],check_exact=True)
    pd.testing.assert_frame_equal(bc.iloc[public],prefix_bc.iloc[public],check_exact=True)
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    labels,changes=guarded_refine(reference.label.to_numpy(),fast,owner,raw,bc,old,blocks,events,approach=25)
    table=pd.DataFrame(dict(frame=np.arange(n),label=labels))
    if pm.validate_submission(table,n):raise ValueError('Invalid refined full candidate')
    for state in ('D','X'):np.testing.assert_array_equal(table.label==state,reference.label==state)
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'candidate.csv',index=False);write(output/'boundary_events.json',changes)
    write(output/'manifest.json',dict(model='tabpfn_guarded_boundaries_seed42',seed=42,full_frames=n,
        candidate_sha256=digest(output/'candidate.csv'),parent_candidate_path=key,parent_candidate_sha256=digest(PARENT/'candidate.csv'),
        parent_manifest_sha256=digest(PARENT/'manifest.json'),changed_frames=int(np.count_nonzero(labels!=reference.label)),events=len(changes),
        D_X_exact=True,topology_preserved=True,public_observation_parity_exact=True,
        all_nine_score_and_boundary_checks=True,public_OOF_spellF1=dict(main=.8972972972972972,purged=.854054054054054,longer=.7897435897435898),
        private_full_match_accuracy_unknown=True,new_model_fits=0,new_TabPFN_queries=0,
        source_sha256={name:digest(ROOT/name) for name in ('experiments/build_guarded_boundary_candidate.py','experiments/probability_guarded_boundaries.py','experiments/possession_boundary_precision.py')},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths.values()},
        qualification_plan_sha256=digest(ROUND/'plan.json'),qualification_result_sha256=digest(ROUND/'qualification_result.json')))
    print(json.dumps(dict(rows=n,changed_frames=int(np.count_nonzero(labels!=reference.label)),events=len(changes),sha256=digest(output/'candidate.csv')),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
