"""Export the qualified source-aware arrivals over the preserved full recipe."""
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
from leading_gap_arrivals import refine

ROUND=ROOT/'data/control_leading_gap_boundaries_v1'
PARENT=ROOT/'data/candidate_tabpfn_boundary_precision_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def build(output):
    plan=json.loads((ROUND/'plan.json').read_text())
    for name in ('experiments/leading_gap_arrivals.py','experiments/source_aware_arrivals.py','experiments/possession_boundary_precision.py'):
        if digest(ROOT/name)!=plan['source_sha256'][name]:raise ValueError('Qualified arrival source changed')
    if not json.loads((ROUND/'qualification_result.json').read_text())['passed']:raise ValueError('Incomplete all-nine qualification')
    if json.loads((ROUND/'selection.json').read_text())['selected']!='leading_camera13':raise ValueError('Unexpected finalist')
    reference=pd.read_csv(PARENT/'candidate.csv');n=source_frames()
    if len(reference)!=n or pm.validate_submission(reference,n):raise ValueError('Invalid parent CSV')
    key=str((PARENT/'candidate.csv').relative_to(ROOT))
    if digest(PARENT/'candidate.csv')!=plan['protected_sha256'][key]:raise ValueError('Frozen parent changed')
    paths=dict(raw=ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl',
        broadcast=ROOT/'data/control_remaining_away_v1/full_inputs/broadcast_extension.pkl',
        owner=ROOT/'data/candidate_tabpfn_short_control_v1/owner_probability.npy')
    raw,bc=pd.read_pickle(paths['raw']),pd.read_pickle(paths['broadcast']);owner=np.load(paths['owner'],allow_pickle=False)
    grand=json.loads((ROOT/'data/candidate_tabpfn_short_control_v1/manifest.json').read_text())
    if digest(paths['owner'])!=grand['cache_sha256']['owner_probability.npy']:raise ValueError('Full ownership probabilities changed')
    public=np.arange(11185,18685)
    prefix_raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl')
    prefix_bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl')
    pd.testing.assert_frame_equal(raw.iloc[public][['h__distance_1','a__distance_1']],prefix_raw.iloc[public][['h__distance_1','a__distance_1']],check_exact=True)
    pd.testing.assert_frame_equal(bc.iloc[public],prefix_bc.iloc[public],check_exact=True)
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    labels,events=refine(reference.label.to_numpy(),owner,raw,bc,blocks,camera=True,posterior_radius=13)
    table=pd.DataFrame(dict(frame=np.arange(n),label=labels))
    if pm.validate_submission(table,n):raise ValueError('Invalid full candidate')
    for state in ('D','X'):np.testing.assert_array_equal(table.label==state,reference.label==state)
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'candidate.csv',index=False);write(output/'arrival_events.json',events)
    write(output/'manifest.json',dict(model='tabpfn_source_aware_arrivals_seed42',seed=42,full_frames=n,
        candidate_sha256=digest(output/'candidate.csv'),parent_candidate_path=key,parent_candidate_sha256=digest(PARENT/'candidate.csv'),
        parent_manifest_sha256=digest(PARENT/'manifest.json'),changed_frames=int(np.count_nonzero(labels!=reference.label)),events=len(events),
        D_X_exact=True,topology_preserved=True,public_observations_exact=True,all_nine_score_and_home_boundary_checks=True,
        public_OOF_spellF1=dict(main=.918918918918919,purged=.8756756756756756,longer=.8),
        new_model_fits=0,new_TabPFN_queries=0,private_full_match_accuracy_unknown=True,
        source_sha256={p:digest(ROOT/p) for p in ('experiments/build_home_boundary_candidate.py','experiments/leading_gap_arrivals.py','experiments/source_aware_arrivals.py')},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths.values()},
        qualification_result_sha256=digest(ROUND/'qualification_result.json')))
    print(json.dumps(dict(rows=n,changed_frames=int(np.count_nonzero(labels!=reference.label)),events=len(events),sha256=digest(output/'candidate.csv')),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
