"""Synthetic temporal capability, mask/symmetry tests and CPU training budget."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from masked_player_network import MaskedPlayer,initialize,probability,step

ROOT=Path(__file__).resolve().parents[1]


def toy(mode):
    labels=np.arange(64)%2;xy=np.zeros((64,31,4,2),np.float32)
    origin=np.array([[-.5,-.4],[-.3,.3],[.3,-.3],[.5,.4]],np.float32)
    xy[:]=origin
    sign=labels*2-1
    for t in range(31):
        xy[:,t,:2,0]+=(t-15)*.01*sign[:,None]
        xy[:,t,2:,0]-=(t-15)*.01*sign[:,None]
    if mode=='frame':xy=np.repeat(xy[:,15:16],31,axis=1)
    observed=np.ones(xy.shape[:-1],bool);teams=np.broadcast_to(np.array([0,0,1,1]),observed.shape).copy()
    return [torch.from_numpy(v) for v in (xy,observed,teams,labels.astype(np.float32))]


def fit_toy(mode):
    initialize();model=MaskedPlayer();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    data=toy(mode);rng=np.random.default_rng(42)
    for _ in range(200):
        indices=rng.integers(0,64,16);step(model,optimizer,*[v[indices] for v in data])
    return model,probability(model,*data[:3])


def main(output):
    output.mkdir(parents=True,exist_ok=False);scores={};temporal=None
    for mode in ('frame','temporal'):
        model,p=fit_toy(mode);scores[mode]=float(((p>=.5).astype(int)==np.arange(64)%2).mean())
        torch.save(model.state_dict(),output/(mode+'.pt'));loaded=MaskedPlayer();loaded.load_state_dict(torch.load(output/(mode+'.pt'),weights_only=True),strict=True)
        np.testing.assert_array_equal(p,probability(loaded,*toy(mode)[:3]));np.save(output/(mode+'_probability.npy'),p)
        if mode=='temporal':temporal=model
    if scores['temporal']<.95 or scores['frame']>.60:raise ValueError('Synthetic temporal capability gate failed')
    _,again=fit_toy('temporal');np.testing.assert_array_equal(again,np.load(output/'temporal_probability.npy'))
    xy,mask,teams,_=toy('temporal');p=probability(temporal,xy,mask,teams)
    flipped=probability(temporal,xy,mask,1-teams);np.testing.assert_allclose(p,1-flipped,rtol=1e-6,atol=1e-6)
    permutation=np.array([2,0,3,1]);np.testing.assert_allclose(p,probability(temporal,xy[:,:,permutation],mask[:,:,permutation],teams[:,:,permutation]),rtol=1e-5,atol=1e-5)
    missing=torch.zeros_like(mask);neutral=probability(temporal,torch.full_like(xy,float('nan')),missing,teams)
    np.testing.assert_array_equal(neutral,np.full(len(neutral),.5))
    masked=mask.clone();masked[:,:,0]=False;dirty=xy.clone();dirty[:,:,0]=float('nan');clean=xy.clone();clean[:,:,0]=0
    np.testing.assert_array_equal(probability(temporal,dirty,masked,teams),probability(temporal,clean,masked,teams))
    initialize();model=MaskedPlayer();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(7);big=torch.from_numpy(rng.uniform(-1,1,(32,31,46,2)).astype(np.float32));observed=torch.from_numpy(rng.random((32,31,46))>.2)
    groups=torch.from_numpy(np.broadcast_to(np.arange(46)%2,(32,31,46)).copy());target=torch.from_numpy((np.arange(32)%2).astype(np.float32))
    for _ in range(3):step(model,optimizer,big,observed,groups,target)
    times=[]
    for _ in range(10):
        start=time.monotonic();step(model,optimizer,big,observed,groups,target);times.append(time.monotonic()-start)
    seconds=float(np.median(times));projected=seconds*4000*2
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)
    plan=json.loads((ROOT/'experiments/MASKED_PLAYER_RUNTIME_PLAN.json').read_text());limits=plan['limits']
    summary=dict(passed=projected<=limits['projected_two_4000step_fits_seconds_max'] and rss<=limits['peak_RSS_GiB_max'],
        accuracy=scores,strict_reload_exact=True,deterministic_refit_exact=True,eval_team_swap=True,node_permutation=True,all_missing_neutral=True,masked_NaN_safe=True,
        parameters=sum(p.numel() for p in model.parameters()),seconds_per_train_step=seconds,projected_two_fits_seconds=projected,peak_RSS_GiB=rss,
        actual_observations_loaded=False,actual_targets_loaded=False,new_research_fits=0,torch=torch.__version__,numpy=np.__version__)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
