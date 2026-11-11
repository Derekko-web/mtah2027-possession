"""Ball-specific synthetic capability, controls, masks and bounded CPU budget."""
import argparse
from pathlib import Path
import json
import resource
import time
import numpy as np
import torch
from player_representation import ROOT,digest,write_json
from masked_player_network import initialize
from ball_context_network import BallContext,step,probability

FOLDER=ROOT/'data/control_ball_context_v1'


def toy(control=False):
    target=np.arange(64)%2;xy=np.zeros((64,31,4,2),np.float32);xy[:]=[[-.5,-.4],[-.3,.3],[.3,-.3],[.5,.4]]
    ball=np.zeros((64,31,2),np.float32);ball[:,:,0]=np.where(target[:,None],-.4,.4)
    observed=np.ones(xy.shape[:-1],bool);teams=np.broadcast_to([0,0,1,1],observed.shape).copy();ball_seen=np.full(ball.shape[:-1],not control,bool)
    return [torch.from_numpy(v) for v in (xy,observed,teams,ball,ball_seen)],torch.from_numpy(target.astype(np.float32))


def fit(control):
    initialize();model=BallContext();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);data,target=toy(control);rng=np.random.default_rng(42)
    for _ in range(200):
        rows=rng.integers(0,64,16);step(model,optimizer,[v[rows] for v in data],target[rows])
    return model,probability(model,data)


def main(output):
    plan_path=ROOT/'experiments/BALL_CONTEXT_RUNTIME_PLAN.json';plan=json.loads(plan_path.read_text());freeze=FOLDER/'protocol_freeze_A'
    if plan_path.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Ball context runtime plan changed')
    for name,sha in json.loads((freeze/'implementation.json').read_text()).items():
        if digest(ROOT/name)!=sha:raise ValueError('Ball context implementation changed')
    output.mkdir(parents=True,exist_ok=False);scores={}
    for mode,control in (('ball',False),('masked',True)):
        model,p=fit(control);scores[mode]=float(((p>=.5)==(np.arange(64)%2)).mean());torch.save(model.state_dict(),output/(mode+'.pt'));np.save(output/(mode+'_probability.npy'),p)
        loaded=BallContext();loaded.load_state_dict(torch.load(output/(mode+'.pt'),weights_only=True),strict=True);np.testing.assert_array_equal(p,probability(loaded,toy(control)[0]))
        if not control:trained=model;original=p
    if scores['ball']<.95 or scores['masked']>.6:raise ValueError('Ball-information capability control failed')
    _,again=fit(False);np.testing.assert_array_equal(again,original)
    values,target=toy();swapped=list(values);swapped[2]=1-values[2];np.testing.assert_allclose(probability(trained,values),1-probability(trained,swapped),rtol=1e-5,atol=1e-5)
    masked=list(values);masked[4]=torch.zeros_like(masked[4]);dirty=list(masked);dirty[3]=torch.full_like(dirty[3],float('nan'));np.testing.assert_array_equal(probability(trained,masked),probability(trained,dirty))
    missing=list(values);missing[1]=torch.zeros_like(missing[1]);np.testing.assert_array_equal(probability(trained,missing),np.full(64,.5))
    permutation=[2,0,3,1];permuted=[v[:,:,permutation] if i<3 else v for i,v in enumerate(values)];np.testing.assert_allclose(probability(trained,values),probability(trained,permuted),rtol=1e-5,atol=1e-5)
    initialize();model=BallContext();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);rng=np.random.default_rng(39)
    xy=torch.from_numpy(rng.uniform(-1,1,(32,31,46,2)).astype(np.float32));seen=torch.ones((32,31,46),dtype=torch.bool);groups=torch.from_numpy(np.broadcast_to(np.arange(46)%2,(32,31,46)).copy());ball=torch.from_numpy(rng.uniform(-1,1,(32,31,2)).astype(np.float32));ball_seen=torch.ones((32,31),dtype=torch.bool);target=torch.from_numpy((np.arange(32)%2).astype(np.float32));times=[]
    for i in range(13):
        start=time.monotonic();step(model,optimizer,[xy,seen,groups,ball,ball_seen],target)
        if i>=3:times.append(time.monotonic()-start)
    seconds=float(np.median(times));peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3;passed=seconds*8000<=2400 and peak<=2
    write_json(output/'summary.json',dict(passed=passed,accuracy=scores,parameters=sum(p.numel() for p in model.parameters()),strict_reload_exact=True,deterministic_refit_exact=True,teamswap=True,ball_masked_NaN_safe=True,all_players_missing_neutral=True,node_permutation=True,seconds_per_step=seconds,projected_two_4000step_fits_seconds=seconds*8000,peak_GiB=peak,actual_observations_loaded=False,actual_targets_loaded=False))
    write_json(output/'manifest.json',dict(plan_sha256=digest(plan_path),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));print((output/'summary.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=FOLDER/'synthetic');main(p.parse_args().output)
