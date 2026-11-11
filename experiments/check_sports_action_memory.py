"""Synthetic changing-input endurance before restarting the bounded private stream."""
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
import numpy as np
from sports_action_runtime import ROOT,EVIDENCE,digest,load_model,verify_public_inputs
from sports_action_memory import allocate_inputs,copy_rgb,release_inactive


def main():
    import torch
    path=EVIDENCE/'memory_execution_plan.json';plan=json.loads(path.read_text())
    _,_,checkpoint,vendor=verify_public_inputs()
    output=EVIDENCE/'memory_endurance';output.mkdir()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.mps.set_per_process_memory_fraction(.6)
    pixels=np.random.default_rng(42).integers(0,256,size=(100,448,796,3),dtype=np.uint8)
    rows=[]

    def no_network(*a,**kw):raise RuntimeError('Network forbidden during action inference')

    with patch.object(socket.socket,'connect',no_network):
        model,keys=load_model(checkpoint,vendor);model.to('mps')
        small=torch.from_numpy(pixels[:,:32,:56,:].copy()).permute(0,3,1,2).unsqueeze(0).to('mps')
        with torch.inference_mode():
            a,_=model(small,inference=True)
            b,_=model(small.float(),inference=True)
        parity=max(float((a[k]-b[k]).abs().max().item()) for k in a)
        if parity>plan['normalization_parity_tolerance']:
            raise ValueError('uint8 input head parity failed')
        (output/'input_parity.json').write_text(json.dumps(dict(max_abs_head_difference=parity,passed=True))+'\n')
        del small,a,b
        release_inactive(torch)
        host,device=allocate_inputs(torch)
        for index in range(plan['synthetic_forwards']):
            np.add(pixels,np.uint8(1),out=pixels)
            x=copy_rgb(torch,pixels,host,device);started=time.monotonic()
            with torch.inference_mode():raw,_=model(x,inference=True)
            torch.mps.synchronize()
            finite=all(torch.isfinite(value).all().item() for value in raw.values())
            after=int(torch.mps.driver_allocated_memory());del raw,x
            release_inactive(torch)
            rows.append(dict(index=index,finite=finite,seconds=time.monotonic()-started,
                driver_after_forward=after,driver_after_clear=int(torch.mps.driver_allocated_memory())))
            (output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
            if not finite:raise ValueError('Nonfinite synthetic endurance output')
            if (index+1)%10==0 or index+1==plan['synthetic_forwards']:
                print('Synthetic bounded-memory forwards',index+1,'/',plan['synthetic_forwards'],flush=True)
    steady=[r['driver_after_clear'] for r in rows[8:]]
    span=max(steady)-min(steady)
    summary=dict(passed=len(rows)==plan['synthetic_forwards'] and all(r['finite'] for r in rows) and span<=64*1024*1024,
        forwards=len(rows),state_keys=keys,postwarmup_driver_span_bytes=span,
        driver_after_clear_max=max(r['driver_after_clear'] for r in rows),
        driver_after_forward_max=max(r['driver_after_forward'] for r in rows),
        plan_sha256=digest(path),memory_source_sha256=digest(ROOT/'experiments/sports_action_memory.py'),
        checkpoint_sha256=digest(checkpoint),competition_images_used=False,possession_targets_read=False,
        memory_fraction=.6,watermarks_not_disabled=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    if not summary['passed']:raise SystemExit('Bounded-memory endurance failed')


if __name__=='__main__':main()
