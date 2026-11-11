"""R40 prospective replication on image windows disjoint from the R38 pilot."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from player_representation import ROOT,digest,write_json
from wasb_adapter import warp
from wasb_decode import selection_expression
from wasb_components import normalize_warped,components

FOLDER=ROOT/'data/control_wasb_confirmation_v1'
ORIGINAL=ROOT/'data/control_wasb_v1'
PLAN=ROOT/'experiments/WASB_CONFIRMATION_PLAN.json'


def validate():
    p=json.loads(PLAN.read_text());freeze=FOLDER/'protocol_freeze'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('WASB confirmation plan changed')
    for group in (p['input_sha256'],json.loads((freeze/'implementation.json').read_text())):
        for name,sha in group.items():
            if digest(ROOT/name)!=sha:raise ValueError('WASB confirmation dependency differs: '+name)
    return p


def prepare():
    p=validate();output=FOLDER/'inputs';output.mkdir(parents=True,exist_ok=False);records=[]
    prior=json.loads((ORIGINAL/'external_inputs/summary.json').read_text())
    for camera in prior['records']:
        key=camera['camera'];meta=camera['video'];n=int(meta['nb_read_frames']);width=int(meta['width']);height=int(meta['height'])
        old=np.load(ORIGINAL/'external_inputs'/(key+'_centers.npy'));all_centers=np.arange(1,n-1)
        available=all_centers[(np.abs(all_centers[:,None]-old[None])>2).all(1)]
        centers=available[np.rint(np.linspace(0,len(available)-1,p['centers_per_camera'])).astype(int)]
        if len(np.unique(centers))!=p['centers_per_camera']:raise ValueError('Insufficient disjoint image windows')
        needed=np.unique((centers[:,None]+[-1,0,1]).ravel());old_frames=np.unique((old[:,None]+[-1,0,1]).ravel())
        if len(np.intersect1d(needed,old_frames)):raise ValueError('Pilot and confirmation pixels overlap')
        np.save(output/(key+'_centers.npy'),centers);np.save(output/(key+'_decoded_frames.npy'),needed)
        write_json(output/(key+'_selection_before_targets.json'),dict(centers_sha256=digest(output/(key+'_centers.npy')),disjoint_from_pilot_image_windows=True,labels_read=False,video=meta))
        rgb=np.lib.format.open_memmap(output/(key+'_rgb.npy'),mode='w+',dtype=np.uint8,shape=(len(needed),288,512,3))
        command=['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(ORIGINAL/'external_inputs'/(key+'.avi')),'-vf','select='+selection_expression(needed),'-vsync','0','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
        with (output/(key+'_decode.log')).open('wb') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
            try:
                for i in range(len(needed)):
                    value=bytearray();size=width*height*3
                    while len(value)<size:
                        part=process.stdout.read(size-len(value))
                        if not part:raise ValueError('Missing confirmation frame')
                        value.extend(part)
                    rgb[i]=warp(np.frombuffer(value,np.uint8).reshape(height,width,3))
                if process.stdout.read(1) or process.wait()!=0:raise ValueError('Unexpected confirmation decode count')
            finally:
                process.stdout.close()
                if process.poll() is None:process.terminate();process.wait()
        rgb.flush();records.append(dict(camera=key,video=meta,centers=len(centers),decoded=len(needed)));print('Disjoint target-blind confirmation RGB',key,flush=True)
    write_json(output/'summary.json',dict(records=records,targets_loaded=False,competition_media_loaded=False));write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),cache_sha256={q.name:digest(q) for q in sorted(output.iterdir()) if q.is_file()}));validate()


def predict(model,rgb,decoded,center,mode,width,height):
    import torch
    frames=center+np.asarray([-1,0,1]) if mode=='temporal' else np.repeat(center,3);rows=np.searchsorted(decoded,frames);np.testing.assert_array_equal(decoded[rows],frames)
    values=np.concatenate([normalize_warped(rgb[r]) for r in rows],axis=0)
    with torch.inference_mode():hm=torch.sigmoid(model(torch.from_numpy(values[None]))[0])[0,1].numpy()
    detected=components(hm,width,height)
    if detected:
        best=max(detected,key=lambda x:x[2]);return dict(visible=True,x=best[0],y=best[1],score=best[2],components=len(detected))
    return dict(visible=False,x=None,y=None,score=None,components=0)


def infer():
    import resource,time
    from masked_player_network import initialize
    from check_wasb_runtime import load_model
    p=validate();source=FOLDER/'inputs';m=json.loads((source/'manifest.json').read_text())
    for name,sha in m['cache_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('Confirmation input changed')
    output=FOLDER/'outputs';output.mkdir(parents=True,exist_ok=False);initialize();model=load_model();start=__import__('time').monotonic()
    for camera in json.loads((source/'summary.json').read_text())['records']:
        key=camera['camera'];meta=camera['video'];width=int(meta['width']);height=int(meta['height'])
        old_records=json.loads((ORIGINAL/'external_outputs'/(key+'_predictions.json')).read_text());old_centers=np.load(ORIGINAL/'external_inputs'/(key+'_centers.npy'));old_frames=np.load(ORIGINAL/'external_inputs'/(key+'_decoded_frames.npy'))
        for center in old_centers[[0,-1]]:
            rgb=np.load(ORIGINAL/'external_inputs'/(key+'_rgb.npy'),mmap_mode='r')
            for mode in ('temporal','frame'):
                actual=predict(model,rgb,old_frames,center,mode,width,height);expected=next(r for r in old_records if r['frame']==center and r['mode']==mode)
                if any(actual[k]!=expected[k] for k in actual):raise ValueError('Cross-process pilot detection replay differs')
            rgb._mmap.close()
        centers=np.load(source/(key+'_centers.npy'));decoded=np.load(source/(key+'_decoded_frames.npy'));records=[]
        for i,center in enumerate(centers):
            rgb=np.load(source/(key+'_rgb.npy'),mmap_mode='r')
            for mode in ('temporal','frame'):records.append(dict(camera=key,frame=int(center),mode=mode,**predict(model,rgb,decoded,center,mode,width,height)))
            rgb._mmap.close()
            if (i+1)%32==0:print(key,i+1,'/',len(centers),'confirmation pairs',flush=True)
        write_json(output/(key+'_predictions.json'),records)
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),targets_loaded=False,competition_media_loaded=False,pilot_detection_replays_exact=8,elapsed_seconds=__import__('time').monotonic()-start,peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,cache_sha256={q.name:digest(q) for q in sorted(output.iterdir()) if q.is_file()}))


def assess():
    from assess_wasb_external import labels
    p=validate();output=FOLDER/'assessment';output.mkdir(parents=True,exist_ok=False);rows=[];model=json.loads((FOLDER/'outputs/manifest.json').read_text())
    for name,sha in model['cache_sha256'].items():
        if digest(FOLDER/'outputs'/name)!=sha:raise ValueError('Immutable confirmation detections changed')
    for camera in json.loads((FOLDER/'inputs/summary.json').read_text())['records']:
        key=camera['camera'];truth,coverage=labels(ORIGINAL/'external_inputs'/(key+'.xml'),int(camera['video']['nb_read_frames']));records=json.loads((FOLDER/'outputs'/(key+'_predictions.json')).read_text())
        for mode in ('temporal','frame'):
            selected=[r for r in records if r['mode']==mode];tp=fp=fn=tn=wrong=visible=0
            for r in selected:
                x,y,v=truth.get(r['frame'],(-1.,-1.,False));visible+=v
                if v and r['visible']:
                    if np.hypot(r['x']-x,r['y']-y)<4:tp+=1
                    else:fp+=1;fn+=1;wrong+=1
                elif v:fn+=1
                elif r['visible']:fp+=1
                else:tn+=1
            recall=tp/(tp+fn) if tp+fn else 0.;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
            passed=len(selected)==512 and visible>=32 and recall>=.75 and f1>=.8
            rows.append(dict(camera=key,mode=mode,centers=len(selected),visible=visible,invisible=len(selected)-visible,tp=tp,fp=fp,fn=fn,tn=tn,wrong_location=wrong,recall=recall,f1=f1,passed=passed,annotation_coverage=coverage))
    passed=all(r['passed'] for r in rows if r['mode']=='temporal') and model['peak_GiB']<=2 and model['elapsed_seconds']<=1800
    write_json(output/'selection.json',dict(passed=passed,records=rows,pilot_R38_remains_failed_for_insufficient_support=True,new_image_windows_disjoint=True,model_thresholds_unchanged=True,resource=model));print((output/'selection.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer','assess']);args=p.parse_args();globals()[args.phase]()
