"""Fixed-budget isolated CPU worker; identical saved fitting schedules for both arms."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from masked_player_network import MaskedPlayer,initialize,probability,step
from masked_player_data import assemble

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/control_masked_player_v1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def infer(model,x,mask,teams,half_id,centers,views,mode):
    result=[]
    for a in range(0,len(centers),64):
        c=np.asarray(centers[a:a+64]);v=np.asarray(views[a:a+64]);size=len(c)
        if size<64:c=np.r_[c,np.repeat(c[-1],64-size)];v=np.r_[v,np.repeat(v[-1],64-size)]
        values=assemble(x,mask,teams,half_id,c,v,repeat=mode=='frame',trim=False)
        result.append(probability(model,*[torch.from_numpy(z) for z in values])[:size])
    return np.concatenate(result)


def run(mode,replay,output):
    plan_path=ROOT/'experiments/MASKED_PLAYER_EXTERNAL_PLAN.json';plan=json.loads(plan_path.read_text());cfg=plan['training'];source=FOLDER/'corpus'
    if plan_path.read_bytes()!=(FOLDER/'protocol_freeze_B/plan.json').read_bytes():raise ValueError('Masked-player fitting plan changed')
    manifest=json.loads((source/'manifest.json').read_text())
    for n,expected in manifest['cache_sha256'].items():
        if digest(source/n)!=expected:raise ValueError('Corpus artifact changed: '+n)
    for n,expected in manifest['source_sha256'].items():
        if digest(ROOT/n)!=expected:raise ValueError('Frozen implementation changed: '+n)
    output.mkdir(parents=True,exist_ok=False);initialize(cfg['seed']);model=MaskedPlayer()
    x=np.load(source/'xy.npy',mmap_mode='r');mask=np.load(source/'observed.npy',mmap_mode='r');teams=np.load(source/'teams.npy',mmap_mode='r');half_id=np.load(source/'half_id.npy')
    schedule=dict(np.load(source/'schedule.npz'));saved=FOLDER/'external_models'/mode
    if replay:model.load_state_dict(torch.load(saved/'model.pt',map_location='cpu',weights_only=True),strict=True)
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);losses=[];started=time.monotonic()
        for index in range(cfg['steps']):
            args=[schedule[k][index] for k in ('centers','views','flip_x','flip_y')]
            values=assemble(x,mask,teams,half_id,*args,repeat=mode=='frame',trim=True)
            loss=step(model,optimizer,*[torch.from_numpy(z) for z in values],torch.from_numpy(schedule['target'][index]));losses.append(loss)
            if (index+1)%100==0:
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)
                if rss>2:raise MemoryError('Frozen2GiB/model resource gate exceeded')
                print(mode,'step',index+1,'loss',round(float(np.mean(losses[-100:])),5),'seconds',round(time.monotonic()-started,1),'RSSGiB',round(rss,3),flush=True)
        torch.save(model.state_dict(),output/'model.pt');np.save(output/'training_loss.npy',np.asarray(losses))
    validation=dict(np.load(source/'validation.npz'));centers=validation['centers']
    for view,selection in (('clean',np.zeros(len(centers),int)),('masked',1+((validation['local']//10)%3))):
        value=infer(model,x,mask,teams,half_id,centers,selection,mode);np.save(output/(view+'_probability.npy'),value)
        loaded=MaskedPlayer();loaded.load_state_dict(torch.load((saved if replay else output)/'model.pt',map_location='cpu',weights_only=True),strict=True)
        np.testing.assert_array_equal(value,infer(loaded,x,mask,teams,half_id,centers,selection,mode))
        if replay:np.testing.assert_array_equal(value,np.load(saved/(view+'_probability.npy')))
    metadata=dict(mode=mode,steps=cfg['steps'],seed=cfg['seed'],parameters=sum(p.numel() for p in model.parameters()),
        corpus_manifest_sha256=digest(source/'manifest.json'),schedule_sha256=digest(source/'schedule.npz'),checkpoint_reload_exact=True,
        new_external_models=0 if replay else 1,manual_labels_used=False,competition_targets_loaded=False,
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print('Fixed-budget worker complete',mode,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['temporal','frame'],required=True);p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.mode,a.replay,a.output or FOLDER/('external_models_replay' if a.replay else 'external_models')/a.mode)
