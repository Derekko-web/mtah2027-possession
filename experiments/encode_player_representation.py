"""Frozen encoders, four reflections, and pre-classification hidden states."""
import argparse
from pathlib import Path
import numpy as np
import torch
from player_representation import ROOT,ROUND,PLAN,VIEWS,check_inputs,audit_cache,digest,write_json
from masked_player_network import MaskedPlayer,initialize
from masked_player_native_data import assemble_native


def extract(model,values):
    captured=[]
    hook=model.output.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    model.eval()
    try:
        with torch.inference_mode():
            p=torch.sigmoid(model(*[torch.from_numpy(np.ascontiguousarray(v)) for v in values])).numpy()
            z=captured[0].numpy();hidden=np.concatenate([z[:len(p)],z[len(p):]],axis=1)
    finally:hook.remove()
    return hidden,p


def reflected(model,values):
    results=[]
    for sx,sy in ((1,1),(-1,1),(1,-1),(-1,-1)):
        x=values[0]*np.asarray([sx,sy],np.float32);z,_=extract(model,(x,values[1],values[2]));results.append(z)
    return np.mean(np.stack(results).astype(np.float64),axis=0).astype(np.float32)


def main(output):
    check_inputs();source=ROUND/'inputs';audit_cache(source);output.mkdir(parents=True,exist_ok=False)
    frames=np.load(source/'frames.npy');half=np.load(source/'half_id.npy')
    x,mask,teams=[np.load(source/(name+'.npy'),mmap_mode='r') for name in ('xy','observed','teams')]
    prefix=[np.load(source/(name+'.npy'),mmap_mode='r') for name in ('prefix_xy','prefix_observed','prefix_teams')]
    for view in VIEWS:
        initialize();model=MaskedPlayer()
        if view=='pretrained_latent':model.load_state_dict(torch.load(ROOT/'data/control_masked_player_v1/external_models/temporal/model.pt',map_location='cpu',weights_only=True),strict=True)
        else:torch.save(model.state_dict(),output/'random_model.pt')
        parts=[];probabilities=[]
        for start in range(0,len(frames),64):
            index=np.arange(start,min(start+64,len(frames)));size=len(index)
            if size<64:index=np.r_[index,np.repeat(index[-1],64-size)]
            full=assemble_native(x,mask,teams,half,frames[index]//5);small=[np.array(v[index],copy=True) for v in prefix]
            z=reflected(model,full);q=reflected(model,small);np.testing.assert_array_equal(z,q);parts.append(z[:size])
            if view=='pretrained_latent':probabilities.append(extract(model,full)[1][:size])
            if start==0:
                swapped=reflected(model,(full[0],full[1],1-full[2]));np.testing.assert_array_equal(swapped,z[:,np.r_[32:64,0:32]])
                inverse=reflected(model,(full[0]*np.asarray([-1,1],np.float32),full[1],full[2]));np.testing.assert_array_equal(z,inverse)
        np.save(output/(view+'.npy'),np.concatenate(parts))
        if view=='pretrained_latent':
            np.testing.assert_array_equal(np.concatenate(probabilities)[:1500],np.load(ROOT/'data/control_masked_player_v1/native_outputs/probability.npy'))
        print(view,'hidden states and prefix/team/reflection controls pass',flush=True)
    np.save(output/'frames.npy',frames);np.save(output/'eligible.npy',np.load(source/'eligible.npy'))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),targets_loaded=False,prefix_hidden_exact=True,team_swap_exact=True,reflection_invariant=True,scalar_probability_used_as_feature=False,
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    check_inputs()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROUND/'encodings');main(p.parse_args().output)
