"""Matched ball-observed/masked external training on the fixed R36 schedule."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from player_representation import ROOT,digest,write_json
from ball_context_network import BallContext,probability,step
from ball_context_data import assemble_ball
from masked_player_network import initialize

FOLDER=ROOT/'data/control_ball_context_v1'


def infer(model,arrays,centers,views,control):
    result=[]
    for a in range(0,len(centers),64):
        c=np.asarray(centers[a:a+64]);v=np.asarray(views[a:a+64]);size=len(c)
        if size<64:c=np.r_[c,np.repeat(c[-1],64-size)];v=np.r_[v,np.repeat(v[-1],64-size)]
        values=assemble_ball(*arrays,c,v,control=control,trim=False)
        result.append(probability(model,[torch.from_numpy(np.ascontiguousarray(z)) for z in values])[:size])
    return np.concatenate(result)


def run(mode,replay,override):
    plan_path=ROOT/'experiments/BALL_CONTEXT_EXTERNAL_PLAN.json';plan=json.loads(plan_path.read_text());freeze=FOLDER/'protocol_freeze_B'
    if plan_path.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Ball context fitting plan changed')
    for group in (plan['input_sha256'],json.loads((freeze/'implementation.json').read_text())):
        for name,sha in group.items():
            if digest(ROOT/name)!=sha:raise ValueError('Ball context fitting dependency differs: '+name)
    source=ROOT/'data/control_masked_player_v1/corpus';ball_source=FOLDER/'corpus';manifest=json.loads((ball_source/'manifest.json').read_text())
    for name,sha in manifest['cache_sha256'].items():
        if digest(ball_source/name)!=sha:raise ValueError('Ball observations changed')
    arrays=[np.load(source/(name+'.npy'),mmap_mode='r') for name in ('xy','observed','teams','half_id')]+[np.load(ball_source/(name+'.npy'),mmap_mode='r') for name in ('ball','ball_observed')]
    saved=FOLDER/'external_models'/mode;output=override or FOLDER/('external_models_replay' if replay else 'external_models')/mode;output.mkdir(parents=True,exist_ok=False)
    initialize();model=BallContext();schedule=dict(np.load(source/'schedule.npz'));control=mode=='masked';cfg=plan['training']
    if replay:model.load_state_dict(torch.load(saved/'model.pt',map_location='cpu',weights_only=True),strict=True)
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);losses=[];start=time.monotonic()
        for i in range(cfg['steps']):
            args=[schedule[k][i] for k in ('centers','views','flip_x','flip_y')];values=assemble_ball(*arrays,*args,control=control,trim=True)
            losses.append(step(model,optimizer,[torch.from_numpy(np.ascontiguousarray(z)) for z in values],torch.from_numpy(schedule['target'][i])))
            if (i+1)%100==0:
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3
                if rss>2:raise MemoryError('R39 2GiB limit exceeded')
                print(mode,'step',i+1,'loss',round(float(np.mean(losses[-100:])),5),'seconds',round(time.monotonic()-start,1),'GiB',round(rss,3),flush=True)
        torch.save(model.state_dict(),output/'model.pt');np.save(output/'training_loss.npy',np.asarray(losses))
    validation=dict(np.load(source/'validation.npz'));centers=validation['centers']
    for view,which in (('clean',np.zeros(len(centers),int)),('masked',1+((validation['local']//10)%3))):
        p=infer(model,arrays,centers,which,control);np.save(output/(view+'_probability.npy'),p)
        loaded=BallContext();loaded.load_state_dict(torch.load((saved if replay else output)/'model.pt',map_location='cpu',weights_only=True),strict=True)
        np.testing.assert_array_equal(p,infer(loaded,arrays,centers,which,control))
        if replay:np.testing.assert_array_equal(p,np.load(saved/(view+'_probability.npy')))
    write_json(output/'manifest.json',dict(mode=mode,steps=cfg['steps'],parameters=sum(p.numel() for p in model.parameters()),schedule_sha256=digest(source/'schedule.npz'),strict_reload_exact=True,new_external_models=0 if replay else 1,native_targets_loaded=False,cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));print(mode,'complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['ball','masked'],required=True);p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args();run(args.mode,args.replay,args.output)
