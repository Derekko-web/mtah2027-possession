"""Guarded public camera5/6 ranges and zero-indexed, target-blind RGB samples."""
import io
import json
from pathlib import Path
import struct
import subprocess
import urllib.request
import zipfile
import zlib
import numpy as np
from player_representation import ROOT,digest,write_json
from wasb_adapter import warp
from wasb_decode import selection_expression

FOLDER=ROOT/'data/control_wasb_v1'
PLAN=ROOT/'experiments/WASB_EXTERNAL_PLAN.json'


def range_get(url,start,end,total):
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=120) as response:
        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':raise ValueError('Changed or ignored public ZIP range; body not read')
        value=response.read(end-start+2)
    if len(value)!=end-start+1:raise ValueError('Range length mismatch')
    return value


def validate():
    p=json.loads(PLAN.read_text());freeze=FOLDER/'protocol_freeze_B'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('External WASB plan changed')
    implementation=freeze/'implementation_execution_revision1.json'
    if not implementation.exists():implementation=freeze/'implementation.json'
    for group in (p['input_sha256'],json.loads(implementation.read_text())):
        for name,sha in group.items():
            if digest(ROOT/name)!=sha:raise ValueError('External WASB dependency changed: '+name)
    if not json.loads((FOLDER/'synthetic/summary.json').read_text())['passed']:raise ValueError('WASB synthetic gate failed')
    return p


def main():
    plan=validate();output=FOLDER/'external_inputs';output.mkdir(parents=True,exist_ok=False)
    # Central directory and local headers bind both future payload ranges.
    archive=plan['video_archive'];directory=range_get(archive['url'],306354987,306355560,archive['bytes']);(output/'zip_directory.bin').write_bytes(directory)
    for camera in plan['cameras']:
        header=range_get(archive['url'],camera['header_start'],camera['payload_start']-1,archive['bytes'])
        signature,version,flags,method,mt,md,crc,compressed,uncompressed,namelen,extra=struct.unpack('<IHHHHHIIIHH',header[:30])
        if signature!=0x04034b50 or flags!=0 or method!=8 or crc!=int(camera['crc32'],16) or compressed!=camera['compressed_bytes'] or uncompressed!=camera['bytes'] or extra!=0 or header[30:].decode()!=camera['member']:raise ValueError('Public camera local header changed')
        compressed=range_get(archive['url'],camera['payload_start'],camera['payload_end'],archive['bytes']);decoder=zlib.decompressobj(-15)
        value=decoder.decompress(compressed,camera['bytes']+1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(value)!=camera['bytes'] or zlib.crc32(value)!=int(camera['crc32'],16):raise ValueError('Camera DEFLATE/CRC/size failure')
        path=output/(camera['id']+'.avi');path.write_bytes(value);del value,compressed
        print('Verified only public',camera['id'],'video member',flush=True)
    # Complete small revised annotation ZIP, kept separate from blind frame preparation.
    with urllib.request.urlopen(plan['labels_url'],timeout=120) as response:labelzip=response.read(plan['labels_bytes']+1)
    if len(labelzip)!=plan['labels_bytes']:raise ValueError('Revised labels length mismatch')
    (output/'soccer_annos.zip').write_bytes(labelzip)
    with zipfile.ZipFile(io.BytesIO(labelzip)) as archivefile:
        names=sorted(archivefile.namelist())
        if names!=[f'annos/ID-{i}.xml' for i in range(1,7)] or archivefile.testzip() is not None:raise ValueError('Revised annotation paths/CRC changed')
        for camera in plan['cameras']:(output/(camera['id']+'.xml')).write_bytes(archivefile.read('annos/'+camera['id']+'.xml'))
    records=[]
    for camera in plan['cameras']:
        path=output/(camera['id']+'.avi')
        metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(path)]))['streams'][0]
        n=int(metadata['nb_read_frames']);width=int(metadata['width']);height=int(metadata['height'])
        if metadata['r_frame_rate']!='25/1' or n<258:raise ValueError('Unexpected public video frame schema')
        centers=np.rint(np.linspace(1,n-2,plan['centers_per_camera'])).astype(int)
        if len(np.unique(centers))!=len(centers):raise ValueError('Duplicate uniform center')
        needed=np.unique((centers[:,None]+[-1,0,1]).ravel());np.save(output/(camera['id']+'_centers.npy'),centers);np.save(output/(camera['id']+'_decoded_frames.npy'),needed)
        write_json(output/(camera['id']+'_selection_before_labels.json'),dict(centers_sha256=digest(output/(camera['id']+'_centers.npy')),video=metadata,targets_loaded=False,zero_based_frame_indices=True))
        pixels=np.lib.format.open_memmap(output/(camera['id']+'_rgb.npy'),mode='w+',dtype=np.uint8,shape=(len(needed),288,512,3))
        select=selection_expression(needed)
        command=['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(path),'-vf','select='+select,'-vsync','0','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
        with (output/(camera['id']+'_decode.log')).open('wb') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
            for i in range(len(needed)):
                size=width*height*3;parts=[];count=0
                while count<size:
                    piece=process.stdout.read(size-count)
                    if not piece:raise ValueError('Missing exact selected public frame')
                    parts.append(piece);count+=len(piece)
                image=np.frombuffer(b''.join(parts),np.uint8).reshape(height,width,3);pixels[i]=warp(image)
            if process.stdout.read(1) or process.wait()!=0:raise ValueError('Extra selected frames or decoder failure')
        pixels.flush();records.append(dict(camera=camera['id'],video=metadata,centers=len(centers),decoded=len(needed),selection_before_label_join=True))
        print('Prepared target-blind RGB samples',camera['id'],flush=True)
    write_json(output/'summary.json',dict(records=records,competition_media_loaded=False,targets_used_for_sampling=False))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));validate()


if __name__=='__main__':main()
