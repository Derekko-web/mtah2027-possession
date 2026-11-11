"""Immutable no-target detections for adjacent and repeated-frame public samples."""
import argparse
import json
from pathlib import Path
import resource
import numpy as np
import torch
from player_representation import ROOT,digest,write_json
from masked_player_network import initialize
from check_wasb_runtime import load_model
from wasb_components import components,normalize_warped
from prepare_wasb_external import validate

FOLDER=ROOT/'data/control_wasb_v1'


def main(output):
    validate()
    source=FOLDER/'external_inputs';manifest=json.loads((source/'manifest.json').read_text())
    for name,sha in manifest['cache_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('External public input changed')
    initialize();model=load_model();output.mkdir(parents=True,exist_ok=False);all_records=[]
    summary=json.loads((source/'summary.json').read_text())
    for camera in summary['records']:
        key=camera['camera'];meta=camera['video'];width=int(meta['width']);height=int(meta['height'])
        centers=np.load(source/(key+'_centers.npy'));decoded=np.load(source/(key+'_decoded_frames.npy'))
        records=[]
        for index,center in enumerate(centers):
            # Bound resident file pages to one three-frame sample.
            rgb=np.load(source/(key+'_rgb.npy'),mmap_mode='r')
            for mode,frames in (('temporal',center+np.asarray([-1,0,1])),('frame',np.repeat(center,3))):
                rows=np.searchsorted(decoded,frames);np.testing.assert_array_equal(decoded[rows],frames)
                values=np.concatenate([normalize_warped(rgb[r]) for r in rows],axis=0)
                with torch.inference_mode():hm=torch.sigmoid(model(torch.from_numpy(values[None]))[0])[0,1].numpy()
                detections=components(hm,width,height)
                if detections:best=max(detections,key=lambda x:x[2]);prediction=dict(visible=True,x=best[0],y=best[1],score=best[2])
                else:prediction=dict(visible=False,x=None,y=None,score=None)
                records.append(dict(camera=key,frame=int(center),mode=mode,components=len(detections),**prediction))
            rgb._mmap.close()
            if (index+1)%32==0:print(key,index+1,'/256 paired samples',flush=True)
        write_json(output/(key+'_predictions.json'),records);all_records.extend(records)
    write_json(output/'manifest.json',dict(targets_loaded=False,competition_media_loaded=False,new_fits=0,device='cpu',peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3),checkpoint_sha256=digest(FOLDER/'source/wasb_soccer_best.pth.tar'),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=FOLDER/'external_outputs');main(p.parse_args().output)
