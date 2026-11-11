"""Selected external model, target-free native outputs and actual-prefix parity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from masked_player_network import MaskedPlayer,initialize,probability
from masked_player_data import assemble
from masked_player_native_data import assemble_native

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/control_masked_player_v1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    plan_path=ROOT/'experiments/MASKED_PLAYER_NATIVE_PLAN.json';plan=json.loads(plan_path.read_text())
    if plan_path.read_bytes()!=(FOLDER/'protocol_freeze_C/plan.json').read_bytes():raise ValueError('Native source plan changed')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Native inference dependency changed')
    source=FOLDER/'native_inputs';manifest=json.loads((source/'manifest.json').read_text())
    for name,expected in manifest['cache_sha256'].items():
        if digest(source/name)!=expected:raise ValueError('Native inputs changed')
    if not json.loads((source/'summary.json').read_text())['passed']:raise ValueError('Native support failed')
    selection=json.loads((FOLDER/'external_assessment/selection.json').read_text());mode=selection['selected']
    if not selection['passed'] or mode!=plan['selected_external_arm']:raise ValueError('External arm selection differs')
    output.mkdir(parents=True,exist_ok=False);initialize();model=MaskedPlayer()
    path=FOLDER/'external_models'/mode/'model.pt';model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    x=np.load(source/'xy.npy',mmap_mode='r');mask=np.load(source/'observed.npy',mmap_mode='r');teams=np.load(source/'teams.npy',mmap_mode='r');half=np.load(source/'half_id.npy')
    frames=np.load(source/'frames.npy');rows=frames//5;results=[]
    prefix=[np.load(source/(n+'.npy'),mmap_mode='r') for n in ('public_xy','public_observed','public_teams')]
    for start in range(0,len(rows),64):
        index=np.arange(start,min(start+64,len(rows)));size=len(index)
        if size<64:index=np.r_[index,np.repeat(index[-1],64-size)]
        full=assemble_native(x,mask,teams,half,rows[index],repeat=mode=='frame')
        original=[np.array(v[index],copy=True) for v in prefix]
        if mode=='frame':
            original=[np.repeat(v[:,15:16],31,axis=1) for v in original]
            permitted=(np.arange(-15,16)[None]+rows[index,None]>=0)
            absolute=rows[index,None]+np.arange(-15,16)[None]
            permitted&=(absolute<len(half))&(half[np.clip(absolute,0,len(half)-1)]==half[rows[index,None]])
            original[0][~permitted]=0;original[1][~permitted]=False;original[2][~permitted]=0
        canonical=lambda values:[torch.from_numpy(np.ascontiguousarray(v)) for v in values]
        p=probability(model,*canonical(full));q=probability(model,*canonical(original))
        np.testing.assert_array_equal(p,q);results.append(p[:size])
    value=np.concatenate(results);np.save(output/'probability.npy',value);np.save(output/'frames.npy',frames)
    record=dict(selected_arm=mode,checkpoint_sha256=digest(path),plan_sha256=digest(plan_path),source_sha256=digest(Path(__file__)),
        native_prefix_probabilities_exact=True,new_fits=0,true_targets_loaded=False,human_labels_used=False,
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print('Target-free selected-model native outputs saved; all actual-prefix probabilities exact',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=FOLDER/'native_outputs')
    main(p.parse_args().output)
