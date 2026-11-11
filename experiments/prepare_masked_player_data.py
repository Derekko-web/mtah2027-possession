"""Build masked XY corpus before joining direct external team ownership targets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from masked_player_data import observation_views
from research_temporal_dead import write_json

FOLDER=ROOT/'data/control_masked_player_v1'
PLAN=ROOT/'experiments/MASKED_PLAYER_EXTERNAL_PLAN.json'


def validate():
    p=json.loads(PLAN.read_text())
    if PLAN.read_bytes()!=(FOLDER/'protocol_freeze_B/plan.json').read_bytes():raise ValueError('External masked-player plan changed')
    for name,expected in p['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Masked-player input changed: '+name)
    for group in ('protected_artifacts.json',):
        for name,expected in json.loads((FOLDER/'protocol_freeze_A'/group).read_text()).items():
            if digest(ROOT/name)!=expected:raise ValueError('Selected artifact changed')
    if not json.loads((FOLDER/'synthetic/summary.json').read_text())['passed']:raise ValueError('Synthetic runtime gate failed')
    for name,expected in json.loads((FOLDER/'protocol_freeze_B/implementation.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Masked-player source changed: '+name)
    return p


def main(output):
    plan=validate();output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/control_dfl_transfer_v1/assessment_A';segments=[];offset=0
    for game_index,game in enumerate(plan['matches']):
        for half_index,half in enumerate(('firstHalf','secondHalf')):
            key=game+'_'+half
            with np.load(source/(key+'.npz')) as d:n=(len(d['frames'])+4)//5
            segments.append(dict(key=key,match=game,match_index=game_index,half_index=half_index,start=offset,end=offset+n));offset+=n
    x=np.lib.format.open_memmap(output/'xy.npy',mode='w+',dtype=np.float32,shape=(4,offset,46,2))
    mask=np.lib.format.open_memmap(output/'observed.npy',mode='w+',dtype=bool,shape=(4,offset,46))
    teams=np.lib.format.open_memmap(output/'teams.npy',mode='w+',dtype=np.int8,shape=(offset,46))
    half_id=np.zeros(offset,np.int16)
    for i,s in enumerate(segments):
        with np.load(source/(s['key']+'.npz')) as d:
            positions=d['positions'][::5];membership=d['teams'];pitch=d['pitch']
        values,seen,groups=observation_views(positions,membership,pitch,360000+100*s['match_index']+s['half_index'])
        a,b=s['start'],s['end'];x[:,a:b]=values;mask[:,a:b]=seen;teams[a:b]=groups;half_id[a:b]=i
        print('Target-blind masked-player observations',s['key'],b-a,flush=True)
    x.flush();mask.flush();teams.flush();np.save(output/'half_id.npy',half_id)
    observation_hashes={n:digest(output/n) for n in ('xy.npy','observed.npy','teams.npy','half_id.npy')}
    write_json(output/'observations_before_targets.json',dict(sha256=observation_hashes,status_or_owner_loaded=False))
    labels=np.full(offset,-1,np.int8);buckets=[];validation=[];validation_groups=[];local_rows=[]
    for s in segments:
        with np.load(source/(s['key']+'.npz')) as d:status=d['status'][::5];owner=d['owner'][::5]
        target=np.where((status==1)&(owner==1),1,np.where((status==1)&(owner==2),0,-1)).astype(np.int8)
        a,b=s['start'],s['end'];labels[a:b]=target
    for game in plan['splits']['train']:
        for label in (0,1):
            rows=[]
            for s in segments:
                if s['match']!=game:continue
                eligible=np.arange(s['start']+15,s['end']-15);rows.extend(eligible[labels[eligible]==label])
            if not len(rows):raise ValueError('Missing external match/class training cell')
            buckets.append(np.asarray(rows,int))
    for s in segments:
        if s['match'] in plan['splits']['train']:continue
        rows=np.arange(s['start']+15,s['end']-15);rows=rows[labels[rows]>=0]
        validation.extend(rows);validation_groups.extend([s['match_index']]*len(rows));local_rows.extend(rows-s['start'])
    cfg=plan['training'];rng=np.random.default_rng(cfg['seed']);shape=(cfg['steps'],cfg['batch_size'])
    bucket=rng.integers(0,len(buckets),shape);centers=np.zeros(shape,np.int64)
    for i,rows in enumerate(buckets):
        use=bucket==i;centers[use]=rows[rng.integers(0,len(rows),int(use.sum()))]
    views=rng.integers(0,4,shape).astype(np.int8);flip_x=rng.random(shape)<.5;flip_y=rng.random(shape)<.5
    np.savez_compressed(output/'schedule.npz',centers=centers,views=views,flip_x=flip_x,flip_y=flip_y,target=labels[centers].astype(np.float32),bucket=bucket)
    np.savez_compressed(output/'validation.npz',centers=np.asarray(validation),match=np.asarray(validation_groups),target=labels[np.asarray(validation)].astype(int),local=np.asarray(local_rows))
    np.save(output/'labels.npy',labels);write_json(output/'segments.json',segments)
    for name,expected in observation_hashes.items():
        if digest(output/name)!=expected:raise ValueError('Target join altered observations')
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),source_sha256=json.loads((FOLDER/'protocol_freeze_B/implementation.json').read_text()),input_sha256=plan['input_sha256'],
        observations_before_targets=True,observation_sha256=observation_hashes,training_rows=sum(len(b) for b in buckets),validation_rows=len(validation),
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    validate();print('Corpus prepared; train centers',sum(len(b) for b in buckets),'validation centers',len(validation),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=FOLDER/'corpus')
    main(parser.parse_args().output)
