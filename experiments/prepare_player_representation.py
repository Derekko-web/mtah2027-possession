"""Target-free actual-prefix windows, including the final interpolation endpoint."""
import argparse
import numpy as np
from player_representation import ROOT,ROUND,PLAN,check_inputs,digest,write_json
from broadcast_alignment import load_source
from prepare_masked_player_native import clean_inputs
from masked_player_native_data import assemble_native


def main(output):
    check_inputs();output.mkdir(parents=True,exist_ok=False);frames=np.arange(11185,18690,5)
    with np.load(ROOT/'data/control_dfl_dead_v1/native_inputs/native_observations.npz') as source:
        positions=source['positions'];teams=source['teams']
    first=None
    for n in (18935,19060,len(positions)):
        if n==len(positions):value=positions
        else:
            source=load_source('tactical',n);value=source['positions'];value[~source['valid']]=np.nan
            np.testing.assert_array_equal(source['teams'],teams)
        x,observed,groups,half=clean_inputs(value,teams)
        window=assemble_native(x,observed,groups,half,frames//5)
        if first is None:first=window
        else:
            for a,b in zip(first,window):np.testing.assert_array_equal(a,b)
        if n==len(positions):
            for name,array in (('xy',x),('observed',observed),('teams',groups),('half_id',half)):np.save(output/(name+'.npy'),array)
    xy,observed,groups=first
    for name,array in zip(('prefix_xy','prefix_observed','prefix_teams'),first):np.save(output/(name+'.npy'),array)
    counts=np.stack([(observed&(groups==team)).sum(-1) for team in (0,1)],axis=-1)
    eligible=(counts[:,15]>=4).all(1)&(counts.mean(1)>=4).all(1)
    np.save(output/'frames.npy',frames);np.save(output/'eligible.npy',eligible)
    write_json(output/'summary.json',dict(actual_prefixes=[18935,19060,len(positions)],all_windows_exact=True,targets_loaded=False,centers=len(frames),supported=int(eligible.sum()),radius=75,interpolation_radius=79))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    check_inputs();print('All 1501 prefix windows exact, no targets loaded',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=__import__('pathlib').Path,default=ROUND/'inputs');main(p.parse_args().output)
