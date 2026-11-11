"""Target-free two-arm native inference from immutable bounded input tensors."""
import argparse
import json
from pathlib import Path
import numpy as np
from pathcrf_runtime import FOLDER,ROOT,load_model,infer,sha


def main(output):
    plan_path=ROOT/'experiments/PATHCRF_SIGNAL_PLAN.json';plan=json.loads(plan_path.read_text())
    if plan_path.read_bytes()!=(FOLDER/'protocol_freeze_C/plan.json').read_bytes():raise ValueError('Signal plan changed')
    for name,expected in plan['input_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Native inference dependency changed')
    if not json.loads((FOLDER/'valid_input_checks.json').read_text())['passed']:raise ValueError('Input witnesses failed')
    output.mkdir(parents=True,exist_ok=False);model=load_model();values=np.load(FOLDER/'observation_support/windows.npy')
    frames=np.load(FOLDER/'observation_support/frames.npy');np.save(output/'frames.npy',frames)
    for arm in ('temporal','repeat'):
        parts=[]
        for start in range(0,len(values),8):
            value=values[start:start+8]
            if arm=='repeat':value=np.repeat(value[:,25:26],51,axis=1)
            result=infer(model,value)
            parts.append({k:result[k] for k in ('center_mass','sender','receiver','path')})
            if start%80==0:print('Target-free PathCRF',arm,start,'/',len(values),flush=True)
        np.savez_compressed(output/(arm+'.npz'),**{k:np.concatenate([p[k] for p in parts]) for k in parts[0]})
    metadata=dict(plan_sha256=sha(plan_path),source_sha256=sha(Path(__file__)),new_models=0,targets_loaded=False,private_data_loaded=False,
        windows=len(values),checkpoint='140',all_target_channels_none=True,independent_window_CRF=True,
        cache_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print('All paired raw outputs saved before any target join',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=FOLDER/'native_outputs')
    main(p.parse_args().output)
