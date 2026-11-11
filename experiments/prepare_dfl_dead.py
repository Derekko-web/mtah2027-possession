"""No-fit support audit for direct external live/dead labels, with fixed gates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from research_temporal_dead import write_json

ROUND=ROOT/'data/control_dfl_dead_v1'
PLAN=ROOT/'experiments/DFL_DEAD_FEASIBILITY_PLAN.json'


def validate():
    plan=json.loads(PLAN.read_text());freeze=ROUND/'protocol_freeze_A'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Audit plan changed')
    for mapping in (plan['input_sha256'],json.loads((freeze/'implementation.json').read_text()),json.loads((freeze/'protected_artifacts.json').read_text())):
        for p,expected in mapping.items():
            if digest(ROOT/p)!=expected:raise ValueError('Frozen dependency changed: '+p)
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((freeze/'EXPERIMENT_LEDGER.csv').read_bytes()):raise ValueError('Ledger prefix changed')
    for name,target in json.loads((freeze/'symlinks.json').read_text()).items():
        p=ROOT/name
        if not p.is_symlink() or str(p.readlink())!=target:raise ValueError('Source link changed')
    return plan


def summary(status,ball,positions,teams):
    status=np.asarray(status);ball=np.asarray(ball);positions=np.asarray(positions);teams=np.asarray(teams)
    if ball.shape!=(len(status),2) or positions.shape!=(len(status),len(teams),2) or set(teams)!={0,1}:raise ValueError('Unaligned external observations')
    geometry=np.ones(len(status),bool)
    for team in (0,1):geometry &= np.isfinite(positions[:,teams==team]).all(axis=2).sum(axis=1)>=4
    seen=np.isfinite(ball).all(axis=1);dead=status==0;live=status==1
    intervals=int(np.sum(np.diff(np.r_[False,dead,False].astype(int))==1))
    return dict(frames=len(status),status_valid=int(np.isin(status,[0,1]).sum()),dead=int(dead.sum()),live=int(live.sum()),dead_intervals=intervals,
        dead_ball_xy=int((dead&seen).sum()),live_ball_xy=int((live&seen).sum()),dead_geometry=int((dead&geometry).sum()),live_geometry=int((live&geometry).sum()))


def run(output):
    plan=validate();output.mkdir(parents=True,exist_ok=False);games={};rule=plan['gates']
    source=ROOT/'data/control_dfl_transfer_v1/assessment_A'
    for game in plan['matches']:
        halves=[]
        for half in ('firstHalf','secondHalf'):
            with np.load(source/(game+'_'+half+'.npz')) as d:
                current=summary(d['status'],d['ball'],d['positions'],d['teams'])
                if not np.array_equal(d['status']==0,d['labels']=='D'):raise ValueError('Direct D target conversion differs')
                halves.append(current)
        total={key:sum(h[key] for h in halves) for key in halves[0]}
        ratios=dict(status_valid_fraction=total['status_valid']/total['frames'],dead_ball_xy_fraction=total['dead_ball_xy']/max(total['dead'],1),dead_player_geometry_fraction=total['dead_geometry']/max(total['dead'],1))
        checks=dict(dead_frames=total['dead']>=rule['per_match_minimum_dead_frames'],dead_intervals=total['dead_intervals']>=rule['per_match_minimum_dead_intervals'],
            status=ratios['status_valid_fraction']>=rule['per_match_status_valid_fraction'],dead_ball=ratios['dead_ball_xy_fraction']>=rule['per_match_dead_ball_xy_fraction'],
            dead_geometry=ratios['dead_player_geometry_fraction']>=rule['per_match_dead_player_geometry_fraction'])
        games[game]=dict(total=total,ratios=ratios,checks=checks,passed=all(checks.values()))
        print(game,json.dumps(games[game]),flush=True)
    result=dict(passed=all(g['passed'] for g in games.values()),matches=games,source='direct public DFL status only',new_models=0,new_competition_scores=0,manual_labels_used=False)
    write_json(output/'summary.json',result)
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),summary_sha256=digest(output/'summary.json')))
    validate();print('All external D support gates pass:',result['passed'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROUND/'assessment_A')
    run(parser.parse_args().output)
