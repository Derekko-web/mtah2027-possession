"""Target-blind native-player support audit; no PathCRF forward or scoring."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from games import GAME,half_video_blocks
from pathcrf_observations import window
from research_temporal_dead import write_json

FOLDER=ROOT/'data/control_pathcrf_v1'
PLAN=ROOT/'experiments/PATHCRF_SUPPORT_PLAN.json'


def validate():
    plan=json.loads(PLAN.read_text());freeze=FOLDER/'protocol_freeze_B'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Native support plan changed')
    for mapping in (plan['input_sha256'],json.loads((freeze/'implementation.json').read_text()),json.loads((FOLDER/'protocol_freeze_A/protected_artifacts.json').read_text())):
        for name,expected in mapping.items():
            if digest(ROOT/name)!=expected:raise ValueError('Support dependency changed: '+name)
    if not json.loads((FOLDER/'synthetic/summary.json').read_text())['passed']:raise ValueError('Runtime feasibility failed')
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((FOLDER/'protocol_freeze_A/EXPERIMENT_LEDGER.csv').read_bytes()):raise ValueError('Ledger prefix changed')
    return plan


def main(output):
    plan=validate();output.mkdir(parents=True,exist_ok=False);centers=np.arange(11185,18685,5)
    with np.load(ROOT/'data/control_dfl_dead_v1/native_inputs/native_observations.npz') as source:
        positions=source['positions'];teams=source['teams']
    # Only the declared public source neighborhood is compared/consumed.
    actual=[];public_context=np.arange(11185-150,18685+150)
    for n in (18935,19060):
        source=load_source('tactical',n);xy=source['positions'];xy[~source['valid']]=np.nan
        np.testing.assert_array_equal(xy[public_context],positions[public_context]);np.testing.assert_array_equal(source['teams'],teams)
        actual.append(xy)
    splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text());eligibility={};live={}
    for protocol in ('buffered_minutes','purged_spells','buffered_100_seconds'):
        baseline=pd.read_csv(ROOT/'data/control_dfl_dead_v1/public_trial'/protocol/'oof_predictions.csv',usecols=['frame','current_seed42'])
        np.testing.assert_array_equal(baseline.frame,np.arange(11185,18685))
        full=np.zeros(len(positions),bool);full[baseline.frame]=np.isin(baseline.current_seed42,['H','A'])
        live[protocol]=full[centers];eligible=np.zeros(len(centers),bool)
        for s in splits:
            if s['protocol']!=protocol:continue
            a,b=min(s['validation']),max(s['validation'])+1
            for i,c in enumerate(centers):
                if c-150>=a and c+150<b and full[c-125:c+126].all():eligible[i]=True
        for i,c in enumerate(centers):
            eligible[i]&=any(c-150>=a and c+150<=b for _,a,b in half_video_blocks(GAME))
        eligibility[protocol]=eligible
    candidates=np.logical_or.reduce(list(eligibility.values()));support=np.zeros(len(centers),bool)
    reasons=Counter();features=[];frames=[];left_home=[];witnesses=[]
    for i,c in enumerate(centers):
        if not candidates[i]:reasons['upstream_boundary_or_D']+=1;continue
        value,is_home,reason=window(positions,teams,int(c));reasons[reason]+=1
        if reason=='supported':
            support[i]=True;features.append(value);frames.append(c);left_home.append(is_home)
            if len(witnesses)<6:witnesses.append((int(c),value,is_home,reason))
        if len(witnesses)<9 and reason!='supported':witnesses.append((int(c),value,is_home,reason))
        if i%100==0:print('Target-blind support progress',i,'/',len(centers),flush=True)
    for xy in actual:
        for c,value,is_home,reason in witnesses:
            other,other_home,other_reason=window(xy,teams,c)
            if other_reason!=reason or other_home!=is_home:raise ValueError('Actual-prefix window support differs')
            if value is not None:np.testing.assert_array_equal(value,other)
    tensor=np.stack(features) if features else np.empty((0,51,26,8),np.float32)
    np.save(output/'windows.npy',tensor);np.save(output/'frames.npy',np.asarray(frames,int));np.save(output/'left_team_is_home.npy',np.asarray(left_home,bool))
    values={}
    for protocol,mask in eligibility.items():
        usable=support&mask;denominator=int(live[protocol].sum());coverage=int(usable.sum())/max(denominator,1)
        values[protocol]=dict(selected_live_centers=denominator,eligible_context_centers=int(mask.sum()),supported_centers=int(usable.sum()),coverage=coverage,passed=coverage>=.25)
        np.save(output/(protocol+'_eligible_frames.npy'),centers[usable])
    summary=dict(passed=all(v['passed'] for v in values.values()),protocols=values,reasons=dict(reasons),union_windows=len(features),
        source_context_radius=150,model_window_samples=51,actual_source_prefixes=[18935,19060],full_cache_rows=len(positions),
        public_source_rows_exact=True,feature_witnesses_exact=len(witnesses),true_targets_loaded=False,model_outputs_loaded=False,ball_features_used=False,new_scores=0,new_fits=0)
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),input_sha256=plan['input_sha256'],
        source_sha256=json.loads((FOLDER/'protocol_freeze_B/implementation.json').read_text()),
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    validate();print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=FOLDER/'observation_support')
    main(parser.parse_args().output)
