"""Actual-prefix clean native player-set tensors; no public-target or model reads."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from games import GAME,GAMES,half_video_blocks
from masked_player_data import observation_views,assemble
from masked_player_native_data import assemble_native
from research_temporal_dead import write_json

FOLDER=ROOT/'data/control_masked_player_v1'


def clean_inputs(positions,teams):
    # Clean view is independent of RNG; other predeclared views are discarded.
    x,mask,groups=observation_views(positions[::5],teams,np.asarray(GAMES[GAME]['pitch']),0,centered=False)
    frame=np.arange(0,len(positions),5);half=np.full(len(frame),-1,np.int16)
    for number,(_,a,b) in enumerate(half_video_blocks(GAME)):half[(frame>=a)&(frame<=b)]=number
    return x[:1],mask[:1],groups,half


def main(output):
    plan_path=ROOT/'experiments/MASKED_PLAYER_NATIVE_PLAN.json';plan=json.loads(plan_path.read_text())
    if plan_path.read_bytes()!=(FOLDER/'protocol_freeze_C/plan.json').read_bytes():raise ValueError('Native player-input plan changed')
    if not json.loads((FOLDER/'external_assessment/selection.json').read_text())['passed']:raise ValueError('External model did not pass')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Native input dependency changed')
    output.mkdir(parents=True,exist_ok=False);frames=np.arange(11185,18685,5);rows=frames//5
    with np.load(ROOT/'data/control_dfl_dead_v1/native_inputs/native_observations.npz') as data:full_positions=data['positions'];teams=data['teams']
    views=[]
    for n in (18935,19060,len(full_positions)):
        if n==len(full_positions):positions=full_positions
        else:
            source=load_source('tactical',n);positions=source['positions'];positions[~source['valid']]=np.nan
            np.testing.assert_array_equal(source['teams'],teams)
        x,mask,groups,half=clean_inputs(positions,teams)
        window=assemble_native(x,mask,groups,half,rows)
        if views:
            for a,b in zip(views[0],window):np.testing.assert_array_equal(a,b)
        else:views.append(window)
        if n==len(full_positions):
            for name,array in (('xy',x),('observed',mask),('teams',groups),('half_id',half)):np.save(output/(name+'.npy'),array)
    xy,observed,groups=views[0]
    for name,array in (('public_xy',xy),('public_observed',observed),('public_teams',groups)):np.save(output/(name+'.npy'),array)
    counts=np.stack([(observed&(groups==team)).sum(axis=-1) for team in (0,1)],axis=-1)
    eligible=(counts[:,15]>=4).all(axis=1)&(counts.mean(axis=1)>=4).all(axis=1)
    np.save(output/'frames.npy',frames);np.save(output/'eligible.npy',eligible)
    protocols={}
    for protocol in ('buffered_minutes','purged_spells','buffered_100_seconds'):
        baseline=pd.read_csv(ROOT/'data/control_dfl_dead_v1/public_trial'/protocol/'oof_predictions.csv',usecols=['frame','current_seed42'])
        np.testing.assert_array_equal(baseline.frame,np.arange(11185,18685))
        live=np.isin(baseline.current_seed42.to_numpy()[frames-11185],['H','A']);coverage=float((eligible&live).sum()/max(live.sum(),1))
        protocols[protocol]=dict(selected_live_centers=int(live.sum()),eligible_live_centers=int((eligible&live).sum()),coverage=coverage,passed=coverage>=.8)
    summary=dict(passed=all(v['passed'] for v in protocols.values()),protocols=protocols,all_centers=len(frames),eligible_centers=int(eligible.sum()),
        actual_prefixes=[18935,19060,len(full_positions)],complete_public_windows_exact=True,observation_radius=75,model_run=False,targets_loaded=False,ball_features_used=False)
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',dict(plan_sha256=digest(plan_path),input_sha256=plan['input_sha256'],
        source_sha256={name:digest(ROOT/'experiments'/name) for name in ('prepare_masked_player_native.py','masked_player_data.py','broadcast_alignment.py')},
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=FOLDER/'native_inputs')
    main(p.parse_args().output)
