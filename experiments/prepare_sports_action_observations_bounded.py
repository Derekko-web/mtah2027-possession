"""Offline pretrained action heads from fixed local clips; no targets or frame exports."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
from sports_action_runtime import ROOT, EVIDENCE, load_model, verify_public_inputs
from sports_action_sampling import action_centers, clip_frames, contiguous_ranges
from sports_action_memory import allocate_inputs, copy_rgb, release_inactive

sys.path.insert(0,str(ROOT))
from games import GAME, half_video_blocks, require
WIDTH, HEIGHT = 796, 448


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def read_rgb(handle):
    data = bytearray()
    while len(data) < WIDTH*HEIGHT*3:
        part = handle.read(WIDTH*HEIGHT*3-len(data))
        if not part:
            raise ValueError('Action video stream ended early')
        data.extend(part)
    return np.frombuffer(data,np.uint8).reshape(HEIGHT,WIDTH,3)


def independent_frame(video, frame):
    command = ['ffmpeg','-nostdin','-v','error','-threads','1','-ss',f'{frame/25:.8f}',
        '-i',str(video),'-frames:v','1','-filter_threads','1','-vf',f'scale={WIDTH}:{HEIGHT}:flags=bilinear',
        '-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    result = subprocess.run(command,capture_output=True,check=True)
    if len(result.stdout) != WIDTH*HEIGHT*3:
        raise ValueError('Independent action decode has wrong size')
    return np.frombuffer(result.stdout,np.uint8).reshape(HEIGHT,WIDTH,3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true',help='Only after the previous producer is confirmed terminal')
    args = parser.parse_args()
    execution = EVIDENCE/'memory_execution_plan.json'
    endurance = EVIDENCE/'memory_endurance/summary.json'
    gate = json.loads(endurance.read_text())
    if not gate['passed'] or gate['forwards'] != 187 or gate['plan_sha256'] != file_sha(execution) or gate['memory_source_sha256'] != file_sha(ROOT/'experiments/sports_action_memory.py'):
        raise ValueError('Bounded-memory execution has not passed its frozen gate')
    plan_path = ROOT/'experiments/SPORTS_ACTION_OBSERVATION_PLAN.json'
    plan = json.loads(plan_path.read_text())
    plan_sha = file_sha(plan_path)
    for name,expected in plan['input_sha256'].items():
        if file_sha(ROOT/name) != expected:
            raise ValueError('Frozen action observation input changed')
    _, _, checkpoint, vendor = verify_public_inputs()
    if gate['checkpoint_sha256'] != file_sha(checkpoint):
        raise ValueError('Endurance checkpoint differs')
    video = Path(require(['video'],source='tactical')[0])
    identity = str(Path('data/tactical')/video.name)
    if (ROOT/identity).resolve() != video.resolve():
        raise ValueError('Portable action-video identity differs')
    color = json.loads((ROOT/'data/control_jersey_v1/feasibility/manifest.json').read_text())
    video_sha = color['input_sha256']['tactical/video']
    if file_sha(video) != video_sha:
        raise ValueError('Pinned action video changed')
    reserved = json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held = np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    centers = action_centers(19060,half_video_blocks(GAME),reserved,held)
    if not len(centers):
        raise SystemExit('No eligible action windows')
    output = EVIDENCE/'observations'
    output.mkdir(exist_ok=args.resume)
    (output/'clips').mkdir(exist_ok=args.resume)
    if (output/'plan.json').exists() and (output/'plan.json').read_bytes() != plan_path.read_bytes():
        raise ValueError('Resumed action observation plan differs')
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    names = ('prepare_sports_action_observations_bounded.py','sports_action_memory.py','sports_action_sampling.py','sports_action_runtime.py','../games.py')
    sources = {name:file_sha(ROOT/'experiments'/name) for name in names}
    source_path = output/'source_at_dispatch.json'
    if source_path.exists() and json.loads(source_path.read_text()) != sources:
        raise ValueError('Do not resume with changed source')
    source_path.write_text(json.dumps(sources,indent=2)+'\n')
    progress = output/'progress.json'
    records = json.loads(progress.read_text()) if progress.exists() else []
    done = {r['center'] for r in records}
    if not done <= set(centers) or len(done) != len(records):
        raise ValueError('Resumed action clip identities differ')
    for row in records:
        if file_sha(output/row['file']) != row['sha256']:
            raise ValueError('Saved action clip changed')
    remaining = [int(c) for c in centers if c not in done]
    decoder_checks = []
    if remaining:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        if not torch.backends.mps.is_available():
            raise SystemExit('Planned MPS action runtime unavailable')
        torch.mps.set_per_process_memory_fraction(.6)
        needed = np.unique(np.concatenate([clip_frames(c) for c in remaining]))
        first,last = int(needed[0]),int(needed[-1])
        clauses = [f'between(n,{a-first},{b-first})' for a,b in contiguous_ranges(needed)]
        expression = 'not(mod(n,2))*('+'+'.join(clauses)+')'
        command = ['ffmpeg','-nostdin','-v','error','-threads','2','-ss',f'{first/25:.8f}',
            '-i',str(video),'-t',f'{(last-first+1)/25:.8f}','-filter_threads','1',
            '-vf',f"select='{expression}',scale={WIDTH}:{HEIGHT}:flags=bilinear",'-fps_mode','passthrough',
            '-pix_fmt','rgb24','-f','rawvideo','pipe:1']
        log_index = len(list(output.glob('decoder_*.log')))
        checks = {int(needed[i]) for i in (0,len(needed)//2,len(needed)-1)}
        buffer, cursor = {}, 0

        def no_network(*a,**kw):
            raise RuntimeError('Network forbidden during private action inference')

        with patch.object(socket.socket,'connect',no_network):
            model, keys = load_model(checkpoint,vendor)
            if keys != 600:
                raise ValueError('Action checkpoint schema changed')
            model.to('mps')
            host, device = allocate_inputs(torch)
            with (output/f'decoder_{log_index:03d}.log').open('wb') as log:
                process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
                try:
                    for center in remaining:
                        frames = clip_frames(center)
                        for old in [f for f in buffer if f < frames[0]]:
                            del buffer[old]
                        while cursor < len(needed) and needed[cursor] <= frames[-1]:
                            frame = int(needed[cursor])
                            image = read_rgb(process.stdout)
                            if frame in checks:
                                if not np.array_equal(image,independent_frame(video,frame)):
                                    raise ValueError('Streaming/accurate-seek action pixels differ')
                                decoder_checks.append(dict(frame=frame,sha256=hashlib.sha256(memoryview(image)).hexdigest(),exact=True))
                            buffer[frame] = image
                            cursor += 1
                        pixels = np.stack([buffer[int(f)] for f in frames])
                        rgb_sha = hashlib.sha256(memoryview(pixels)).hexdigest()
                        x = copy_rgb(torch,pixels,host,device)
                        started = time.monotonic()
                        with torch.inference_mode():
                            raw, _ = model(x,inference=True)
                        torch.mps.synchronize()
                        values = {name:value[0].cpu().numpy() for name,value in raw.items()}
                        if {k:v.shape for k,v in values.items()} != dict(im_feat=(100,31),team_feat=(100,),displ_feat=(100,)) or not all(np.isfinite(v).all() for v in values.values()):
                            raise ValueError('Invalid action output')
                        filename = f'clips/clip_{center:06d}.npz'
                        np.savez_compressed(output/filename,frames=frames,**values)
                        records.append(dict(center=center,file=filename,sha256=file_sha(output/filename),input_rgb_sha256=rgb_sha,seconds=time.monotonic()-started))
                        progress.write_text(json.dumps(records,indent=2)+'\n')
                        del x,pixels,raw,values
                        release_inactive(torch)
                        if len(records)%10 == 0 or len(records)==len(centers):
                            print('Action clips processed',len(records),'/',len(centers),flush=True)
                    if cursor != len(needed) or process.stdout.read(1):
                        raise ValueError('Action video stream coverage differs')
                    if process.wait() != 0:
                        raise ValueError('Action video decoder failed')
                finally:
                    process.stdout.close()
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
        (output/f'decoder_checks_{log_index:03d}.json').write_text(json.dumps(decoder_checks,indent=2)+'\n')
    if file_sha(plan_path) != plan_sha or sources != {name:file_sha(ROOT/'experiments'/name) for name in names}:
        raise ValueError('Action producer source/plan changed during extraction')
    if file_sha(video) != video_sha:
        raise ValueError('Action video changed during extraction')
    if sorted(r['center'] for r in records) != centers.tolist():
        raise ValueError('Incomplete action observation schedule')
    inputs = dict(plan['input_sha256'])
    inputs[identity] = video_sha
    inputs[str(execution.relative_to(ROOT))] = file_sha(execution)
    inputs[str(endurance.relative_to(ROOT))] = file_sha(endurance)
    result = dict(plan_sha256=plan_sha,source_sha256=sources,input_sha256=inputs,
        cache_sha256={str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'},
        clips=len(records),targets_not_read=True,network_connections_forbidden=True,
        image_frames_not_saved=True,no_new_fitting=True,global_center_phase=0,full_frame_shape=[HEIGHT,WIDTH,3])
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Action observations complete:',len(records),'clips; no possession targets read',flush=True)


if __name__ == '__main__':
    main()
