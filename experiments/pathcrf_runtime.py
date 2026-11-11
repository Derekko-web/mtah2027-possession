"""Pinned CPU PathCRF loader and synthetic bounded-window network/CRF benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'data/control_pathcrf_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model():
    if torch.__version__!='2.5.0':raise ValueError('Existing pinned Torch2.5.0 required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
    m=json.loads((FOLDER/'runtime_source_manifest.json').read_text())
    for name,expected in m['cache_sha256'].items():
        if sha(FOLDER/'inference_source'/name)!=expected:raise ValueError('Inference source changed')
    source=json.loads((FOLDER/'source_manifest.json').read_text())
    for record in source['source_files']:
        if record['path'] in ('saved/140/model/state_dict_best_acc.pt','saved/140/args.json'):
            if sha(FOLDER/'upstream'/record['path'])!=record['sha256']:raise ValueError('Model artifact changed')
    sys.path.insert(0,str(FOLDER/'inference_source'))
    from models.set_lstm import SetLSTM
    args=json.loads((FOLDER/'upstream/saved/140/args.json').read_text())
    model=SetLSTM(macro_type=args['macro_type'],micro_type=args['micro_type'],seq_model_type=args['seq_model'],crf_model_type=args['crf_model'],
        feat_dim=args['node_in_dim'],team_size=args['team_size'],macro_ppe_dim=args['coarse_dim'],macro_fpe_dim=args['coarse_dim'],
        micro_enc_dim=args['fine_dim'],macro_seq_dim=args['seq_dim'],micro_seq_dim=args['seq_dim'],crf_edge_dim=args['crf_edge_dim'],dropout=args['dropout'])
    state=torch.load(FOLDER/'upstream/saved/140/model/state_dict_best_acc.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);model.eval()
    if sum(p.numel() for p in model.parameters())!=3431031:raise ValueError('Pinned architecture parameter count differs')
    return model


def infer(model,value):
    with torch.inference_mode():
        macro,edge,embedding=model(torch.from_numpy(value),macro_target=None,micro_target=None,random_mask=None)
        if macro.shape!=(*value.shape[:3],2) or edge.shape!=(*value.shape[:2],676) or embedding.shape!=(*value.shape[:2],676,16):raise ValueError('Unexpected PathCRF interface')
        if not all(torch.isfinite(x).all() for x in (macro,edge,embedding)):raise ValueError('Nonfinite model output')
        selected=model.valid_orig_edge_ids
        path=model.crf.decode(edge[:,:,selected],embedding[:,:,selected],last_node_idx=None)
        center=value.shape[1]//2
        emission=torch.softmax(edge[:,center,selected],dim=-1)
        decoded=path[:,center]
        src=model.comp_src[decoded];dst=model.comp_dst[decoded]
        mass=torch.stack([emission[:,model.comp_src<11].sum(1),emission[:,(model.comp_src>=11)&(model.comp_src<22)].sum(1),emission[:,model.comp_src>=22].sum(1)],dim=-1)
        return dict(macro=macro.numpy(),edge=edge.numpy(),center_mass=mass.numpy(),sender=src.numpy(),receiver=dst.numpy(),path=path.numpy())


def synthetic():
    rng=np.random.default_rng(42);value=np.zeros((8,51,26,8),np.float32)
    for team,offset in ((0,0),(1,11)):
        origin=rng.uniform([15+team*25,5],[60+team*25,63],size=(8,11,2));velocity=rng.normal(0,.6,size=(8,11,2))
        origin[:,0]=[4 if team==0 else 101,34];velocity[:,0]=0
        value[:,:,offset:offset+11,0]=-1 if team==0 else 1;value[:,:,offset,1]=1
        value[:,:,offset:offset+11,2:4]=origin[:,None]+(np.arange(51)[None,:,None,None]-25)*.2*velocity[:,None]
        value[:,:,offset:offset+11,4:6]=velocity[:,None];value[:,:,offset:offset+11,6]=np.linalg.norm(velocity,axis=-1)[:,None]
    value[:,:,22:,2:4]=np.array([[0,34],[105,34],[52.5,0],[52.5,68]],np.float32)
    return value


def main(output):
    output.mkdir(parents=True,exist_ok=False);model=load_model();value=synthetic()
    first=infer(model,value);second=infer(model,value)
    for k in first:np.testing.assert_array_equal(first[k],second[k])
    for k,v in first.items():np.save(output/(k+'.npy'),v)
    # Fixed node relabeling, keeping left/right blocks and outside semantics.
    permutation=np.r_[np.arange(11)[::-1],11+np.arange(11)[::-1],np.arange(22,26)]
    other=infer(model,value[:,:,permutation]);undo=np.argsort(permutation)
    np.testing.assert_allclose(first['macro'],other['macro'][:,:,undo],rtol=1e-4,atol=1e-4)
    old_edges=first['edge'].reshape(8,51,26,26);new_edges=other['edge'].reshape(8,51,26,26)[:,:,undo][:,:,:,undo]
    np.testing.assert_allclose(old_edges,new_edges,rtol=1e-4,atol=1e-4)
    timings=[]
    for _ in range(3):
        start=time.monotonic();infer(model,value);timings.append(time.monotonic()-start)
    per_window=float(np.median(timings)/len(value));projected=per_window*1500*2
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3)
    plan=json.loads((ROOT/'experiments/PATHCRF_FEASIBILITY_PLAN.json').read_text());limits=plan['resource_gate']
    passed=rss<=limits['peak_RSS_GiB_max'] and projected<=limits['projected_two_arms_1500_native_windows_seconds_max']
    summary=dict(passed=passed,torch=torch.__version__,numpy=np.__version__,device='cpu',threads=2,parameters=3431031,
        strict_load=True,repeat_exact=True,permutation_logits_close=True,targets_passed=False,native_inputs_loaded=False,
        samples=51,batch=8,seconds_per_window=per_window,projected_two_arms_seconds=projected,peak_RSS_GiB=rss,
        plan_sha256=sha(ROOT/'experiments/PATHCRF_FEASIBILITY_PLAN.json'),source_sha256=sha(Path(__file__)))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
