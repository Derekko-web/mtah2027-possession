"""R42 conditional CPU/MPS equivalence and speed; no possession targets."""
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from player_representation import ROOT,digest,write_json
from check_wasb_runtime import load_model
from masked_player_network import initialize
from wasb_components import normalize_warped,components

FOLDER=ROOT/'data/control_wasb_acceleration_v1'
PLAN=ROOT/'experiments/WASB_ACCELERATION_PLAN.json'
SOURCE=ROOT/'data/control_wasb_v1'


def forward(model,value,device):
    with torch.inference_mode():result=torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(value)).to(device))[0]).cpu().numpy()
    return result


def main():
    p=json.loads(PLAN.read_text());freeze=FOLDER/'protocol_freeze'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Acceleration plan changed')
    for group in (p['input_sha256'],json.loads((freeze/'implementation.json').read_text())):
        for name,sha in group.items():
            if digest(ROOT/name)!=sha:raise ValueError('Acceleration input differs')
    output=FOLDER/'assessment';output.mkdir(parents=True,exist_ok=False);initialize()
    if not torch.backends.mps.is_available():raise ValueError('No existing MPS device')
    cpu=load_model();mps=load_model().to('mps');rng=np.random.default_rng(380001);value=rng.normal(0,1,(1,9,288,512)).astype(np.float32)
    times={};pred={}
    for device,model in (('cpu',cpu),('mps',mps)):
        durations=[]
        for i in range(4):
            start=time.monotonic();y=forward(model,value,device);durations.append(time.monotonic()-start)
            if i==0:pred[device]=y.copy()
            else:np.testing.assert_array_equal(y,pred[device])
        times[device]=durations[1:]
    synthetic_difference=float(np.abs(pred['cpu']-pred['mps']).max());records=[];width=1920;height=1080
    for camera in ('ID-5','ID-6'):
        rgb=np.load(SOURCE/'external_inputs'/(camera+'_rgb.npy'),mmap_mode='r');decoded=np.load(SOURCE/'external_inputs'/(camera+'_decoded_frames.npy'));centers=np.load(SOURCE/'external_inputs'/(camera+'_centers.npy'))
        for i in np.rint(np.linspace(0,len(centers)-1,16)).astype(int):
            center=centers[i]
            for mode,frames in (('temporal',center+[-1,0,1]),('frame',np.repeat(center,3))):
                rows=np.searchsorted(decoded,frames);values=np.concatenate([normalize_warped(rgb[r]) for r in rows],axis=0)[None]
                a=forward(cpu,values,'cpu');b=forward(mps,values,'mps');np.testing.assert_array_equal(b,forward(mps,values,'mps'))
                delta=np.abs(a-b);first=components(a[0,1],width,height);second=components(b[0,1],width,height)
                visibility=bool(first)==bool(second);distance=None;mass=None
                if first and second:
                    x=max(first,key=lambda q:q[2]);y=max(second,key=lambda q:q[2]);distance=float(np.hypot(x[0]-y[0],x[1]-y[1]));mass=abs(x[2]-y[2])/max(x[2],1e-9)
                passed=visibility and len(first)==len(second) and float(delta.max())<=.002 and float(np.sqrt((delta**2).mean()))<=.00002 and (distance is None or distance<=.5) and (mass is None or mass<=.001)
                records.append(dict(camera=camera,mode=mode,sample_index=int(i),maximum_heatmap_difference=float(delta.max()),RMS_heatmap_difference=float(np.sqrt((delta**2).mean())),visibility_exact=visibility,component_count_exact=len(first)==len(second),best_center_distance_pixels=distance,relative_mass_difference=mass,passed=passed))
        rgb._mmap.close();print('CPU/MPS source-only equivalence samples complete',camera,flush=True)
    speedup=float(np.median(times['cpu'])/np.median(times['mps']));cpu_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3;mps_memory=torch.mps.driver_allocated_memory()/1024**3
    passed=synthetic_difference<=.002 and speedup>=2 and cpu_peak<=2.5 and mps_memory<=2 and all(r['passed'] for r in records)
    write_json(output/'summary.json',dict(passed=passed,synthetic_maximum_heatmap_difference=synthetic_difference,repeated_MPS_exact=True,cpu_steady_seconds=times['cpu'],mps_steady_seconds=times['mps'],median_speedup=speedup,cpu_peak_GiB=cpu_peak,mps_driver_GiB=mps_memory,external_records=records,targets_loaded=False,competition_media_loaded=False,new_fits=0))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),cache_sha256={q.name:digest(q) for q in sorted(output.iterdir()) if q.is_file()}));print('MPS equivalence gate',passed,'speedup',speedup,flush=True)


if __name__=='__main__':main()
