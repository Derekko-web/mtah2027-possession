"""Run the unchanged sub-grid alignment over full-match local observations."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import load_source,reconcile_teams,map_anchors
from broadcast_features import aligned_features
from broadcast_provenance import verify_camera_inputs
from games import GAME,half_video_blocks
import subgrid_alignment as subgrid
from research_new_camera_control import ROUND,verify


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_npz(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare():
    verify()
    selected=json.loads((ROUND/'selection.json').read_text())
    if selected['stable_frames']!=13 or not json.loads((ROUND/'qualification_result.json').read_text())['passed']:
        raise ValueError('All-nine complete qualification is required')
    old=ROOT/'data/control_broadcast_v1/full_inputs'
    input_paths=[old/'anchors.npz',old/'mapping.npz',old/'manifest.json',
        ROOT/'data/continuation_20261021/annotation_freeze/windows.json',ROOT/'data/control_native_ball_v1/confirmation_frames.npy',
        ROOT/'data/control_subgrid_alignment_v1/assessment/extension_mapping_19060.npz',
        ROUND/'qualification_result.json',ROOT/'data/candidate_tabpfn_broadcast_short_v1/candidate.csv']
    sources=('experiments/prepare_new_camera_full.py','experiments/subgrid_alignment.py','experiments/broadcast_alignment.py',
             'experiments/broadcast_features.py','experiments/contact_dynamics.py','experiments/new_camera_control.py')
    plan=dict(scientific_method='Exact unchanged Round25refine_anchors and extension,now over full-match observations;same threshold/context/half/reserved/held guards.',
        progress_wrapper='Counts coarse_search calls and forwards identical arguments/results;restored after completion.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in input_paths},
        public_feature_parity_required=True,no_new_targets_or_model_fits=True,automatic_selection=False)
    with (ROUND/'full_feature_plan.json').open('x') as f:f.write(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    output=ROUND/'full_inputs';output.mkdir(exist_ok=False)
    prefix_manifest=json.loads((ROOT/'data/control_broadcast_v1/inputs/manifest.json').read_text())
    verify_camera_inputs(prefix_manifest['input_sha256'])
    print('Loading full tracking and camera geometry; no target labels.',flush=True)
    tactical,broadcast=load_source('tactical'),load_source('broadcast')
    broadcast['teams']=reconcile_teams(tactical['names'],broadcast['names'],broadcast['teams'])
    for label,source in (('tactical',tactical),('broadcast',broadcast)):
        reference=read_npz(ROOT/f'data/control_broadcast_v1/inputs/{label}_observations.npz')
        for key,value in reference.items():
            current=source[key] if key=='teams' else source[key][:len(value)]
            np.testing.assert_array_equal(current,value)
        np.savez_compressed(output/(label+'_observations.npz'),**{k:source[k] for k in reference})
    original=read_npz(old/'anchors.npz');baseline=map_anchors(original,tactical,broadcast,subgrid.STRICT)
    preserved=read_npz(old/'mapping.npz')
    for key,value in baseline.items():np.testing.assert_array_equal(value,preserved[key])
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    progress=ROUND/'full_feature_progress.json';count=0;started=time.monotonic()
    original_search=subgrid.coarse_search
    def reported_search(*args,**kwargs):
        nonlocal count
        result=original_search(*args,**kwargs);count+=1
        if count%250==0:
            write(progress,dict(coarse_queries=count,total_grid=len(original['tactical_frames']),elapsed_seconds=time.monotonic()-started))
            print('Full sub-grid anchors',count,'/',len(original['tactical_frames']),flush=True)
        return result
    subgrid.coarse_search=reported_search
    try:anchors,trace,candidate=subgrid.refine_anchors(tactical,broadcast,original,half_video_blocks(GAME),reserved,held)
    finally:subgrid.coarse_search=original_search
    extension=subgrid.extension(baseline,candidate)
    np.savez_compressed(output/'anchors.npz',**anchors)
    np.savez_compressed(output/'trace.npz',**trace)
    np.savez_compressed(output/'extension_mapping.npz',**extension)
    prefix=read_npz(ROOT/'data/control_subgrid_alignment_v1/assessment/extension_mapping_19060.npz')
    public=np.arange(11185,18685)
    for key in prefix:np.testing.assert_array_equal(extension[key][11135:18735],prefix[key][11135:18735])
    print('Full mapping public parity exact; building point features.',flush=True)
    _,features=aligned_features(tactical,broadcast,extension)
    for n in (18935,19060):
        prior=pd.read_pickle(ROUND/f'broadcast_extension_{n}.pkl')
        pd.testing.assert_frame_equal(features.iloc[public],prior.iloc[public],check_exact=True)
    features.to_pickle(output/'broadcast_extension.pkl')
    verify()
    for p,sha in plan['source_sha256'].items():
        if digest(ROOT/p)!=sha:raise ValueError('Full execution source changed')
    write(output/'manifest.json',dict(passed=True,public_full_mapping_and68_features_exact=True,old_mapping_rows_preserved=True,
        full_frames=len(features),new_mapping_rows=int((extension['origin']==2).sum()),coarse_queries=count,
        elapsed_seconds=time.monotonic()-started,targets_loaded=False,model_fits=0,
        source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    print('Full-match refined observations complete.',flush=True)


if __name__=='__main__':prepare()
