"""Strict weights-only WASB loading and synthetic CPU resource feasibility."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from player_representation import ROOT,digest,write_json
from masked_player_network import initialize

FOLDER=ROOT/'data/control_wasb_v1'


class Config(dict):
    __getattr__=dict.__getitem__


def attributes(value):return Config({k:attributes(v) if isinstance(v,dict) else v for k,v in value.items()})


def load_model():
    source=FOLDER/'source';manifest=json.loads((FOLDER/'source_manifest.json').read_text())
    for name,sha in manifest['cache_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('WASB official source/checkpoint differs')
    spec=importlib.util.spec_from_file_location('official_wasb_hrnet',source/'src/models/hrnet.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=attributes(json.loads((source/'config.json').read_text()));model=module.HRNet(config)
    checkpoint=torch.load(source/'wasb_soccer_best.pth.tar',map_location='cpu',weights_only=True)
    state=checkpoint['model_state_dict']
    if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()):raise ValueError('Expected tensor-only model state')
    model.load_state_dict(state,strict=True);model.eval();return model


def main(output):
    plan_path=ROOT/'experiments/WASB_FEASIBILITY_PLAN.json';plan=json.loads(plan_path.read_text());freeze=FOLDER/'protocol_freeze_A'
    if plan_path.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('WASB runtime plan changed')
    for name,sha in json.loads((freeze/'implementation.json').read_text()).items():
        if digest(ROOT/name)!=sha:raise ValueError('Frozen WASB adapter differs')
    output.mkdir(parents=True,exist_ok=False);initialize();model=load_model()
    rng=np.random.default_rng(380001);value=rng.normal(0,1,(1,9,288,512)).astype(np.float32);tensor=torch.from_numpy(value);times=[];result=[]
    with torch.inference_mode():
        for i in range(4):
            start=time.monotonic();y=model(tensor)[0];times.append(time.monotonic()-start)
            if y.shape!=(1,3,288,512) or not torch.isfinite(y).all():raise ValueError('Wrong synthetic forward')
            result.append(torch.sigmoid(y).numpy())
    for a in result[1:]:np.testing.assert_array_equal(a,result[0])
    second=load_model()
    with torch.inference_mode():other=torch.sigmoid(second(tensor)[0]).numpy()
    np.testing.assert_array_equal(other,result[0]);np.save(output/'synthetic_heatmap.npy',result[0])
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3);seconds=max(times[1:]);projected=seconds*plan['projected_forward_count']
    passed=peak<=plan['max_GiB'] and projected<=plan['max_projected_seconds']
    write_json(output/'summary.json',dict(passed=passed,parameters=sum(p.numel() for p in model.parameters()),strict_weights_only_load=True,repeated_forward_exact=True,reloaded_forward_exact=True,steady_seconds=times[1:],maximum_steady_seconds=seconds,projected_forward_count=plan['projected_forward_count'],projected_seconds=projected,peak_GiB=peak,device='cpu',new_fits=0,competition_media_loaded=False))
    write_json(output/'manifest.json',dict(plan_sha256=digest(plan_path),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));print((output/'summary.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=FOLDER/'synthetic');main(p.parse_args().output)
