"""Export the fully qualified new-camera control stage over the full match."""
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
from new_camera_control import recover

ROUND=ROOT/'data/control_remaining_away_v1'
PARENT=ROOT/'data/candidate_tabpfn_broadcast_short_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def build(output):
    plan=json.loads((ROUND/'plan.json').read_text())
    if digest(ROOT/'experiments/new_camera_control.py')!=plan['source_sha256']['experiments/new_camera_control.py']:
        raise ValueError('Qualified control inference changed')
    if not json.loads((ROUND/'qualification_result.json').read_text())['passed']:
        raise ValueError('All-nine current-score and target gates must pass')
    if json.loads((ROUND/'selection.json').read_text())['stable_frames']!=13:raise ValueError('Unexpected qualified variant')
    features=ROUND/'full_inputs';manifest=json.loads((features/'manifest.json').read_text())
    if not manifest['passed'] or not manifest['public_full_mapping_and68_features_exact']:raise ValueError('Full/public feature parity missing')
    for name,sha in manifest['cache_sha256'].items():
        if digest(features/name)!=sha:raise ValueError('Full feature artifact changed: '+name)
    reference=pd.read_csv(PARENT/'candidate.csv');n=source_frames()
    if digest(PARENT/'candidate.csv')!=plan['protected_sha256'][str((PARENT/'candidate.csv').relative_to(ROOT))]:raise ValueError('Parent candidate changed')
    if len(reference)!=n or pm.validate_submission(reference,n):raise ValueError('Invalid full parent candidate')
    paths=dict(raw=ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl',
        motion=ROUND/'full_control_kernel/kernels.pkl',
        broadcast=features/'broadcast_extension.pkl',mapping=ROOT/'data/control_broadcast_v1/full_inputs/mapping.npz',
        fast=ROOT/'data/candidate_tabpfn_short_control_v1/fast_probabilities.npy')
    kernel_manifest=json.loads((ROUND/'full_control_kernel/manifest.json').read_text())
    if not kernel_manifest['passed'] or digest(paths['motion'])!=kernel_manifest['kernels_sha256']:
        raise ValueError('Full tactical kernel cache is not verified')
    raw,motion,broadcast=[pd.read_pickle(paths[k]) for k in ('raw','motion','broadcast')]
    fast=np.load(paths['fast'],allow_pickle=False);old=np.load(paths['mapping'],allow_pickle=False)['frames']>=0
    public=np.arange(11185,18685)
    prefix_raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl')
    prefix_motion=pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl')
    for full,prefix,columns in ((raw,prefix_raw,['h__distance_1','a__distance_1']),
                                (motion,prefix_motion,['h__tc_kernel','a__tc_kernel'])):
        pd.testing.assert_frame_equal(full.iloc[public][columns],prefix.iloc[public][columns],check_exact=True)
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    labels,events=recover(reference.label.to_numpy(),fast,raw,motion,broadcast,old,blocks,13)
    table=pd.DataFrame(dict(frame=np.arange(n),label=labels))
    if pm.validate_submission(table,n):raise ValueError('Invalid full recovered candidate')
    for state in ('D','X'):np.testing.assert_array_equal(table.label==state,reference.label==state)
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'candidate.csv',index=False);write(output/'recovery_events.json',events)
    write(output/'manifest.json',dict(model='tabpfn_new_camera_control_seed42',seed=42,full_frames=n,
        candidate_sha256=digest(output/'candidate.csv'),parent_candidate_path=str((PARENT/'candidate.csv').relative_to(ROOT)),
        parent_candidate_sha256=digest(PARENT/'candidate.csv'),parent_manifest_sha256=digest(PARENT/'manifest.json'),
        changed_frames=int(np.count_nonzero(labels!=reference.label)),events=len(events),D_X_exact=True,
        all_nine_current_scores_nonregressing=True,remaining_away_recovered_all_nine=True,
        public_OOF_spellF1=dict(main=.8648648648648649,purged=.8216216216216216,longer=.7487179487179488),
        public_feature_parity_exact=True,full_context_fits=0,additional_TabPFN_queries=0,
        private_full_match_accuracy_unknown=True,qualification_plan_sha256=digest(ROUND/'plan.json'),
        full_feature_manifest_sha256=digest(features/'manifest.json'),
        source_sha256={name:digest(ROOT/name) for name in ('experiments/new_camera_control.py','experiments/build_new_camera_candidate.py')},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths.values()}))
    print(json.dumps(dict(rows=n,changed_frames=int(np.count_nonzero(labels!=reference.label)),events=len(events),sha256=digest(output/'candidate.csv')),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
