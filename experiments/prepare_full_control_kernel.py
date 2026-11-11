"""Rebuild missing full tactical pair kernels with original scientific code."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pair_motion import native_pair_features

ROUND=ROOT/'data/control_remaining_away_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    source=ROUND/'full_inputs/tactical_observations.npz'
    manifest=json.loads((ROUND/'full_inputs/manifest.json').read_text())
    if digest(source)!=manifest['cache_sha256'][source.name]:raise ValueError('Verified full observations changed')
    folder=ROUND/'full_control_kernel';folder.mkdir(exist_ok=False)
    plan=dict(purpose='Repair full integration schema by computing the original native tactical kernels;no model or decoder change',
        source_sha256={p:digest(ROOT/p) for p in ('experiments/prepare_full_control_kernel.py','experiments/pair_motion.py')},
        input_sha256={str(source.relative_to(ROOT)):digest(source)},batch_queries=4096,public_parity_required=True,targets_loaded=False)
    (folder/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    with np.load(source,allow_pickle=False) as data:tactical=dict(data)
    n=len(tactical['positions']);result=np.full((n,2),np.nan,np.float32)
    for start in range(0,n,4096):
        end=min(n,start+4096)
        native=native_pair_features(tactical,np.arange(start,end))
        for k,team in enumerate(('h__','a__')):
            result[start:end,k]=np.minimum(native[team+'past5_control_kernel'],native[team+'future5_control_kernel'])
        print('Tactical control kernels',end,'/',n,flush=True)
    frame=pd.DataFrame(result,columns=['h__tc_kernel','a__tc_kernel'])
    prefix=pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl')
    pd.testing.assert_frame_equal(frame.iloc[11185:18685],prefix.iloc[11185:18685][list(frame.columns)],check_exact=True)
    frame.to_pickle(folder/'kernels.pkl')
    (folder/'manifest.json').write_text(json.dumps(dict(passed=True,full_frames=n,public_original_kernel_parity_exact=True,
        source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],kernels_sha256=digest(folder/'kernels.pkl'),
        original_native_pair_features_unchanged=True,model_fits=0,targets_loaded=False),indent=2,sort_keys=True)+'\n')
    print('Full tactical kernel parity passed.',flush=True)


if __name__=='__main__':prepare()
