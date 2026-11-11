"""Freeze ball corruption arrays without loading external or native targets."""
import json
import numpy as np
from player_representation import ROOT,digest,write_json
from ball_context_data import ball_views

FOLDER=ROOT/'data/control_ball_context_v1'
PLAN=ROOT/'experiments/BALL_CONTEXT_EXTERNAL_PLAN.json'


def validate():
    plan=json.loads(PLAN.read_text());freeze=FOLDER/'protocol_freeze_B'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Ball context external plan changed')
    for group in (plan['input_sha256'],json.loads((freeze/'implementation.json').read_text())):
        for name,sha in group.items():
            if digest(ROOT/name)!=sha:raise ValueError('Ball context dependency changed: '+name)
    if not json.loads((FOLDER/'synthetic/summary.json').read_text())['passed']:raise ValueError('Ball context synthetic gate failed')
    return plan


def main():
    validate();output=FOLDER/'corpus';output.mkdir(parents=True,exist_ok=False);original=ROOT/'data/control_masked_player_v1/corpus';segments=json.loads((original/'segments.json').read_text())
    xy=np.load(original/'xy.npy',mmap_mode='r');seen=np.load(original/'observed.npy',mmap_mode='r');n=xy.shape[1]
    ball=np.lib.format.open_memmap(output/'ball.npy',mode='w+',dtype=np.float32,shape=(4,n,2));mask=np.lib.format.open_memmap(output/'ball_observed.npy',mode='w+',dtype=bool,shape=(4,n))
    for s in segments:
        with np.load(ROOT/'data/control_dfl_transfer_v1/assessment_A'/(s['key']+'.npz')) as data:b=data['ball'][::5];pitch=data['pitch']
        a,z=s['start'],s['end'];v,m=ball_views(b,pitch,xy[:,a:z],seen[:,a:z],390000+100*s['match_index']+s['half_index']);ball[:,a:z]=v;mask[:,a:z]=m
    ball.flush();mask.flush();write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),targets_loaded=False,original_player_corpus=digest(original/'manifest.json'),new_ball_observations_only=True,cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));validate();print('Ball views saved before any target join',flush=True)


if __name__=='__main__':main()
