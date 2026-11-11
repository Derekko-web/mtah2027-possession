"""Build a confirmed public-only temporal D candidate in a fresh directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from games import GAME,half_video_blocks,source_frames
from motion_dead_trim import decode_motion_dead_trim
from research_temporal_dead import ROUND,PLAN,RUNTIME,check_inputs,predict_variant,write_json
from temporal_dead import normalization,transform


def main(output):
    plan=check_inputs();selection=json.loads((ROUND/'screen/selection.json').read_text())
    if not selection['passed']:raise ValueError('No eligible public-only candidate')
    head,mode,_=selection['selected'].split('_')
    for seed in (42,43,44):
        stage='screen' if seed==42 else f'confirmation_seed{seed}'
        gate=json.loads((ROUND/stage/'selection.json').read_text())
        name=f'{head}_{mode}_seed{seed}'
        if not all(all(p['checks'].values()) for p in gate['pairs'] if p['model']==name) or len([p for p in gate['pairs'] if p['model']==name])!=3:
            raise ValueError('All nine confirmation pairs must pass')
        if not (ROUND/(stage+'_replay')/'selection.json').exists():raise ValueError('Confirmation replay is required')
    output.mkdir(parents=True,exist_ok=False)
    baseline=ROOT/'data/candidate_motion_dead_trim';before=json.loads((ROUND/'protocol_freeze/protected_artifacts.json').read_text())
    for path,expected in before.items():
        if digest(ROOT/path)!=expected:raise ValueError('Selected baseline differs')
    full=ROOT/'data/control_renewal_v1/stoppage_inputs'
    raw=pd.read_pickle(full/'raw.pkl');features=pd.read_pickle(full/'physical.pkl')
    public=np.arange(11185,18685);prefix=pd.read_pickle(ROUND/'inputs/physical.pkl')
    if len(raw)!=source_frames() or len(features)!=len(raw):raise ValueError('Full provided-data cache differs')
    check=np.arange(11185-31,18685+31)
    pd.testing.assert_frame_equal(features.iloc[check],prefix.iloc[check],check_exact=True)
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    arrays=[np.load(baseline/(name+'_probabilities.npy')) for name in ('current','previous','fast','heads','broadcast','motion')]
    bc=pd.read_pickle(ROOT/'data/control_broadcast_v1/full_inputs/combined_inputs.pkl')
    info=pd.read_pickle(ROOT/'data/control_pair_motion_v1/full_inputs/motion_info.pkl')
    reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)
    np.testing.assert_array_equal(reference,pd.read_csv(baseline/'candidate.csv').label.to_numpy())
    print('Full selected8 identity and public feature-context parity pass',flush=True)
    labels=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels!='X'),public)
    matrix=features.to_numpy(np.float32);mean,scale=normalization(matrix,public)
    target=(labels[public]=='D').astype(np.float32)
    weight=(len(public)/(2*np.bincount(target.astype(int))[target.astype(int)])).astype(np.float32)
    normalized=transform(matrix,mean,scale)
    frames=np.concatenate([np.arange(a,b) for a,b in blocks])
    np.save(output/'normalized.npy',normalized)
    np.savez_compressed(output/'fitting.npz',train=public,target=target,weight=weight,validation=frames,mean=mean,scale=scale)
    job=dict(normalized=str(output/'normalized.npy'),fitting=str(output/'fitting.npz'),half_blocks=blocks,seed=42,modes=[head])
    write_json(output/'job.json',job)
    command=[str(RUNTIME),str(ROOT/'experiments/temporal_dead_network.py'),'--job',str(output/'job.json'),'--output',str(output/'worker')]
    with (output/'worker.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    dead=np.load(output/'worker'/(head+'_probability.npy'))
    prediction=predict_variant(arrays,dead,bc,info,raw,blocks,mode,frames)
    table=pd.DataFrame(dict(frame=np.arange(len(raw)),label=prediction));reason=pm.validate_submission(table,source_frames())
    if reason:raise ValueError(reason)
    table.to_csv(output/'candidate.csv',index=False)
    write_json(output/'manifest.json',dict(model='Public-only '+head+' live/dead correction',seed=42,head=head,decoder=mode,
        csv_sha256=digest(output/'candidate.csv'),reference_csv_sha256=digest(baseline/'candidate.csv'),plan_sha256=digest(PLAN),
        checkpoint_sha256=digest(output/'worker'/(head+'.pt')),selected_protected_sha256=before,
        source_sha256={name:digest(ROOT/name) for name in ('experiments/build_temporal_dead_candidate.py','experiments/research_temporal_dead.py','experiments/temporal_dead.py','experiments/temporal_dead_network.py')},
        feature_sha256=digest(full/'physical.pkl'),raw_sha256=digest(full/'raw.pkl'),public_training_rows=len(public),
        changed_frames=int((prediction!=reference).sum()),unknown_exact=bool(np.array_equal(prediction=='X',reference=='X')),
        conditional_ownership_preserved=True,manual_labels_used=False,private_targets_used=False,
        limitation='Full-match changed rows are predictions,not verified corrections; private accuracy unknown.'))
    print('Saved reproducible-candidate build:',output/'candidate.csv',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
