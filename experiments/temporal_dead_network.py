"""Isolated, offline CPU PyTorch worker. Reads numeric public-only fit jobs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from temporal_dead import RADIUS,DILATIONS,WIDTH,STEPS,BATCH,windows


class TemporalDead(nn.Module):
    def __init__(self,channels,width=WIDTH):
        super().__init__()
        self.project=nn.Conv1d(channels,width,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,3,padding=d,dilation=d),nn.ReLU(),nn.Dropout(.1)) for d in DILATIONS])
        self.output=nn.Linear(width,1)

    def forward(self,value):
        value=torch.relu(self.project(value))
        for block in self.blocks:value=value+block(value)
        return self.output(value[:,:,RADIUS])[:,0]


def initialize(seed):
    if torch.__version__!='2.5.0':raise ValueError('Pinned existing Torch2.5.0 runtime required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def infer(model,values,frames,half_blocks,mode):
    result=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(frames),256):
            batch=windows(values,frames[start:start+256],half_blocks,repeat=mode=='frame')
            result.append(torch.sigmoid(model(torch.from_numpy(batch))).numpy())
    return np.concatenate(result)


def fit(values,train,target,weight,half_blocks,mode,seed,steps=STEPS):
    initialize(seed);model=TemporalDead(values.shape[1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed)
    schedule=rng.integers(0,len(train),size=(steps,BATCH))
    model.train();last_loss=None
    for selected in schedule:
        batch=windows(values,train[selected],half_blocks,repeat=mode=='frame')
        optimizer.zero_grad(set_to_none=True)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(batch)),torch.from_numpy(target[selected]),reduction='none')*torch.from_numpy(weight[selected])).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();last_loss=float(loss.detach())
    return model,schedule,last_loss


def run(job_path,output,replay=False):
    job=json.loads(job_path.read_text());output.mkdir(parents=True,exist_ok=False)
    with np.load(job['fitting']) as data:
        train,target,weight,validation=[data[k] for k in ('train','target','weight','validation')]
    values=np.load(job['normalized']);half_blocks=job['half_blocks'];seed=job['seed']
    records=[]
    modes=job.get('modes',['frame','temporal'])
    if not modes or len(set(modes))!=len(modes) or not set(modes)<={'frame','temporal'}:raise ValueError('Invalid frozen head selection')
    for mode in modes:
        started=time.monotonic();prefix=output/mode
        if replay:
            initialize(seed)
            state=torch.load(Path(job['parent'])/(mode+'.pt'),map_location='cpu',weights_only=True)
            model=TemporalDead(values.shape[1]);model.load_state_dict(state['state_dict'],strict=True)
            if state['mode']!=mode or state['seed']!=seed or state['steps']!=STEPS:raise ValueError('Checkpoint recipe differs')
        else:
            model,schedule,last_loss=fit(values,train,target,weight,half_blocks,mode,seed)
            torch.save(dict(state_dict=model.state_dict(),mode=mode,seed=seed,steps=STEPS,channels=values.shape[1]),str(prefix)+'.pt')
            np.save(str(prefix)+'_schedule.npy',schedule)
        probability=infer(model,values,validation,half_blocks,mode)
        if not replay:
            state=torch.load(str(prefix)+'.pt',map_location='cpu',weights_only=True)
            loaded=TemporalDead(values.shape[1]);loaded.load_state_dict(state['state_dict'],strict=True)
            np.testing.assert_array_equal(probability,infer(loaded,values,validation,half_blocks,mode))
        np.save(str(prefix)+'_probability.npy',probability)
        records.append(dict(mode=mode,parameters=sum(p.numel() for p in model.parameters()),checkpoint_reload_exact=True))
        print(('Replayed' if replay else 'Fitted')+' '+mode+' D head in '+str(round(time.monotonic()-started,2))+'s',flush=True)
    (output/'worker.json').write_text(json.dumps(dict(torch_version=torch.__version__,numpy_version=np.__version__,device='cpu',records=records),indent=2)+'\n')


def self_test(output):
    output.mkdir(parents=True,exist_ok=False);initialize(42)
    rng=np.random.default_rng(42)
    # Identical center observations; only past/future ordering supplies the label.
    centers=np.arange(40,40+128*70,70);values=np.zeros((centers[-1]+40,4),np.float32)
    target=np.arange(len(centers))%2
    for center,label in zip(centers,target):
        sign=1 if label else -1
        values[center-12:center,0]=sign;values[center+1:center+13,0]=-sign
    halves=[(0,len(values))];predictions={}
    for mode in ('frame','temporal'):
        model,schedule,loss=fit(values,centers,target.astype(np.float32),np.ones(len(centers),np.float32),halves,mode,42,steps=150)
        predictions[mode]=infer(model,values,centers,halves,mode)
        # Chunked inference / batch size and serialization must preserve values.
        torch.save(dict(state_dict=model.state_dict()),output/(mode+'.pt'))
        loaded=TemporalDead(4);loaded.load_state_dict(torch.load(output/(mode+'.pt'),weights_only=True)['state_dict'])
        np.testing.assert_array_equal(predictions[mode],infer(loaded,values,centers,halves,mode))
    accuracy={k:float(((v>=.5).astype(int)==target).mean()) for k,v in predictions.items()}
    if accuracy['temporal']<.95 or accuracy['frame']>.55:raise AssertionError('Synthetic order-only task failed')
    # Repeated deterministic training, not just repeated prediction.
    model,_,_=fit(values,centers,target.astype(np.float32),np.ones(len(centers),np.float32),halves,'temporal',42,steps=150)
    np.testing.assert_array_equal(predictions['temporal'],infer(model,values,centers,halves,'temporal'))
    (output/'summary.json').write_text(json.dumps(dict(passed=True,accuracy=accuracy,training_repeat_exact=True,checkpoint_reload_exact=True),indent=2)+'\n')
    print('Synthetic temporal order, deterministic refit and checkpoint checks pass',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test(args.output)
    else:run(args.job,args.output,args.replay)
