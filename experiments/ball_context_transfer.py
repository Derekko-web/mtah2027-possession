"""R39 frozen native representation transfer; reuse the exact R37 CV engine."""
import argparse
import json
from pathlib import Path
import numpy as np
from player_representation import ROOT,digest,write_json,audit_cache
from masked_player_native_data import assemble_native

FOLDER=ROOT/'data/control_ball_context_v1/native_transfer'
PLAN=ROOT/'experiments/BALL_CONTEXT_TRANSFER_PLAN.json'
PLAYER=ROOT/'data/control_player_representation_v1/inputs'


def check_inputs():
    p=json.loads(PLAN.read_text());freeze=FOLDER/'protocol_freeze'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Ball transfer plan changed')
    for group in (p['input_sha256'],json.loads((freeze/'implementation.json').read_text()),json.loads((freeze/'protected_artifacts.json').read_text())):
        for name,sha in group.items():
            if digest(ROOT/name)!=sha:raise ValueError('Ball transfer dependency differs: '+name)
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((freeze/'EXPERIMENT_LEDGER.csv').read_bytes()):raise ValueError('Earlier score ledger changed')
    for name,target in json.loads((freeze/'symlinks.json').read_text()).items():
        path=ROOT/name
        if not path.is_symlink() or str(path.readlink())!=target:raise ValueError('Input source link changed')
    external=json.loads((ROOT/'data/control_ball_context_v1/external_assessment/selection.json').read_text())
    if not external['passed'] or external['selected']!='ball':raise ValueError('Relative-ball external source gate failed')
    return p


def clean_ball(ball,player_seen,pitch):
    value=np.asarray(ball,np.float32)[::5];pitch=np.asarray(pitch,np.float32)
    good=np.isfinite(value).all(1)&(value>=-5).all(1)&(value<=pitch+5).all(1)&player_seen[0].any(-1)
    return np.where(good[:,None],value/pitch*2-1,0)[None].astype(np.float32),good[None]


def ball_windows(ball,seen,half,centers):
    centers=np.asarray(centers,int)
    if (centers<0).any() or (centers>=ball.shape[1]).any() or (half[centers]<0).any():raise ValueError('Unknown native half center')
    index=centers[:,None]+np.arange(-15,16)[None];clipped=np.clip(index,0,ball.shape[1]-1)
    allowed=(index>=0)&(index<ball.shape[1])&(half[clipped]==half[np.asarray(centers),None])
    value=np.array(ball[0,clipped],copy=True);mask=np.array(seen[0,clipped],copy=True);value[~allowed]=0;mask[~allowed]=False
    return value,mask


def prepare():
    from broadcast_alignment import load_source
    from games import GAME,GAMES
    check_inputs();audit_cache(PLAYER);output=FOLDER/'inputs';output.mkdir(parents=True,exist_ok=False)
    frames=np.load(PLAYER/'frames.npy');half=np.load(PLAYER/'half_id.npy');players=np.load(PLAYER/'observed.npy',mmap_mode='r')
    with np.load(ROOT/'data/control_dfl_dead_v1/native_inputs/native_observations.npz') as source:ball=source['ball']
    first=None
    for n in (18935,19060,len(ball)):
        value=ball if n==len(ball) else load_source('tactical',n)['ball']
        b,m=clean_ball(value,players[:,:len(value[::5])],GAMES[GAME]['pitch']);windows=ball_windows(b,m,half[:b.shape[1]],frames//5)
        if first is None:first=windows
        else:
            for a,c in zip(first,windows):np.testing.assert_array_equal(a,c)
        if n==len(ball):np.save(output/'ball.npy',b);np.save(output/'ball_observed.npy',m)
    np.save(output/'prefix_ball.npy',first[0]);np.save(output/'prefix_ball_observed.npy',first[1]);np.save(output/'frames.npy',frames)
    write_json(output/'summary.json',dict(actual_prefixes=[18935,19060,len(ball)],ball_windows_exact=True,shared_player_windows_exact=True,targets_loaded=False,ball_visible_centers=int(first[1][:,15].sum()),centers=len(frames)))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));check_inputs()


def encode():
    import torch
    from ball_context_network import BallContext
    from masked_player_network import initialize
    from encode_player_representation import extract
    check_inputs();audit_cache(FOLDER/'inputs');audit_cache(PLAYER);output=FOLDER/'encodings';output.mkdir(parents=True,exist_ok=False)
    source=FOLDER/'inputs';x,mask,teams=[np.load(PLAYER/(n+'.npy'),mmap_mode='r') for n in ('xy','observed','teams')];half=np.load(PLAYER/'half_id.npy');frames=np.load(PLAYER/'frames.npy')
    ball=np.load(source/'ball.npy');ball_seen=np.load(source/'ball_observed.npy');prefix=[np.load(PLAYER/(n+'.npy'),mmap_mode='r') for n in ('prefix_xy','prefix_observed','prefix_teams')]+[np.load(source/'prefix_ball.npy'),np.load(source/'prefix_ball_observed.npy')]
    def hidden(model,values):
        results=[]
        for sx,sy in ((1,1),(-1,1),(1,-1),(-1,-1)):
            signs=np.asarray([sx,sy],np.float32);v=[values[0]*signs,values[1],values[2],values[3]*signs,values[4]];results.append(extract(model,v)[0])
        return np.mean(np.stack(results).astype(np.float64),axis=0).astype(np.float32)
    for view,mode in (('ball_latent','ball'),('masked_latent','masked')):
        initialize();model=BallContext();path=ROOT/'data/control_ball_context_v1/external_models'/mode/'model.pt';model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);parts=[]
        for start in range(0,len(frames),64):
            rows=np.arange(start,min(start+64,len(frames)));size=len(rows)
            if size<64:rows=np.r_[rows,np.repeat(rows[-1],64-size)]
            full=list(assemble_native(x,mask,teams,half,frames[rows]//5))+list(ball_windows(ball,ball_seen,half,frames[rows]//5));small=[np.array(v[rows],copy=True) for v in prefix]
            if mode=='masked':
                for v in (full,small):v[3][:]=0;v[4][:]=False
            a=hidden(model,full);b=hidden(model,small);np.testing.assert_array_equal(a,b);parts.append(a[:size])
            if start==0:
                swapped=list(full);swapped[2]=1-full[2];np.testing.assert_array_equal(hidden(model,swapped),a[:,np.r_[32:64,0:32]])
                flipped=list(full);flipped[0]=full[0]*[-1,1];flipped[3]=full[3]*[-1,1]
                # Keep canonical float32 inputs under the sign transform.
                flipped[0]=flipped[0].astype(np.float32);flipped[3]=flipped[3].astype(np.float32);np.testing.assert_array_equal(hidden(model,flipped),a)
        np.save(output/(view+'.npy'),np.concatenate(parts));print('Ball transfer hidden/prefix/team/reflection controls pass',view,flush=True)
    np.save(output/'frames.npy',frames);np.save(output/'eligible.npy',np.load(PLAYER/'eligible.npy'))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),targets_loaded=False,only_external_frozen_weights=True,scalar_probability_used_as_feature=False,cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));check_inputs()


def screen(seed,replay,output):
    # Parameterize only the experiment context and eligible view of the frozen
    # R37 engine. Fitting, decoding, scoring and every protection are identical.
    import research_player_representation as engine
    original_assess=engine.substantial
    def eligible_ball(*args):
        result=original_assess(*args);result['eligible']=[name for name in result['eligible'] if name.startswith('ball_latent_')];result['passed']=bool(result['eligible']);result['selected']=result['eligible'][0] if result['eligible'] else None;return result
    engine.ROUND=FOLDER;engine.PLAN=PLAN;engine.VIEWS=('ball_latent','masked_latent');engine.check_inputs=check_inputs;engine.substantial=eligible_ball
    engine.run(seed,replay,output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','encode','screen']);p.add_argument('--seed',type=int,choices=[42,43,44],default=42);p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
    if args.phase=='screen':screen(args.seed,args.replay,args.output)
    else:globals()[args.phase]()
