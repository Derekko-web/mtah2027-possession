"""Shared XY-only D evidence from actual tactical inputs; no target loading."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from dfl_dead_features import build
from games import GAME,GAMES,half_video_blocks,source_frames
from research_dfl_dead_external import validate as external_validate,ROUND
from research_temporal_dead import write_json


def observations(n):
    source=load_source('tactical',n);xy=source['positions'].copy();ball=source['ball'].copy();valid=source['valid']
    xy[~valid]=np.nan;ball[~valid]=np.nan;pitch=np.array(GAMES[GAME]['pitch'])
    ball[~(np.isfinite(ball).all(axis=1)&(ball>=-5).all(axis=1)&(ball<=pitch+5).all(axis=1))]=np.nan
    return xy,ball,source['teams']


def main(output):
    external_validate()
    if not json.loads((ROUND/'external_trial/summary.json').read_text())['passed']:raise ValueError('External D model failed its gates')
    plan=ROOT/'experiments/DFL_DEAD_PUBLIC_PLAN.json';frozen=ROUND/'protocol_freeze_C/plan.json'
    if plan.read_bytes()!=frozen.read_bytes():raise ValueError('Public integration plan changed')
    for name,expected in json.loads(plan.read_text())['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Public dependency changed: '+name)
    output.mkdir(parents=True,exist_ok=False);blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    views=[];source_sizes=(18935,19060,source_frames());teacher=XGBClassifier();teacher.load_model(ROUND/'external_trial/teacher_seed42.ubj')
    for n in source_sizes:
        print('Actual native XY features',n,flush=True)
        xy,ball,teams=observations(n);features=build(xy,ball,teams,blocks)
        p=teacher.predict_proba(features)[:,1]
        if not np.isfinite(p).all():raise ValueError('Invalid native teacher probabilities')
        views.append((features.iloc[11185:18685].copy(),p[11185:18685].copy()))
        if n==source_frames():
            features.to_pickle(output/'shared_full.pkl');np.save(output/'teacher_full.npy',p)
            np.savez_compressed(output/'native_observations.npz',positions=xy,ball=ball,teams=teams)
    for f,p in views[1:]:
        pd.testing.assert_frame_equal(views[0][0],f,check_exact=True);np.testing.assert_array_equal(views[0][1],p)
    write_json(output/'manifest.json',dict(plan_sha256=digest(plan),actual_input_sizes=list(source_sizes),public_features_exact=True,public_teacher_exact=True,
        targets_loaded=False,human_labels_used=False,input_sha256=json.loads(plan.read_text())['input_sha256'],
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()},
        source_sha256={name:digest(ROOT/'experiments'/name) for name in ('prepare_dfl_dead_native.py','dfl_dead_features.py','dfl_transfer_features.py','broadcast_alignment.py')}))
    print('Full/18935/19060 native features and D teacher outputs agree exactly',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROUND/'native_inputs')
    main(p.parse_args().output)
