"""Reproduce and explain strict anchors; aggregate output only, no media or targets."""
import argparse
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from anchor_failures import (trace_anchors,trace_mapping,array_equal,attribution_summary,additional_consistency)
from build_temporal_candidate import digest
from games import GAME,half_video_blocks


def validate_inputs():
    path=ROOT/'experiments/ANCHOR_FAILURE_PLAN.json';plan=json.loads(path.read_text())
    root=ROOT/'data/control_anchor_failures_v1'
    if path.read_bytes()!=(root/'protocol_freeze/plan.json').read_bytes():raise ValueError('Frozen diagnostic plan changed')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen diagnostic input changed: '+name)
    for name,expected in json.loads((root/'protocol_freeze/protected_artifacts.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Protected selected/reference artifact changed')
    previous=ROOT/'data/control_broadcast_action_v1/prefix_replay'
    provenance=json.loads((previous/'summary.json').read_text())
    if not provenance['independently_recomputed_mappings']:raise ValueError('Independent prefix provenance absent')
    for name,expected in provenance['source_sha256'].items():
        if digest(ROOT/'experiments'/name)!=expected:raise ValueError('Inherited prefix source changed')
    for n in (18935,19060):
        name=f'mapping_{n}.npz'
        if digest(previous/name)!=provenance['cache_sha256'][name]:raise ValueError('Inherited prefix mapping changed')
    return plan


def require_equal(a,b,label):
    if not array_equal(a,b):raise ValueError('Diagnostic replay differs: '+label)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/control_anchor_failures_v1/assessment')
    args=parser.parse_args();plan=validate_inputs();output=args.output.resolve();output.mkdir(parents=True)
    path=ROOT/'experiments/ANCHOR_FAILURE_PLAN.json'
    sources=('prepare_anchor_failures.py','anchor_failures.py','broadcast_alignment.py','../games.py')
    hashes={name:digest(ROOT/'experiments'/name) for name in sources}
    (output/'source_at_dispatch.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (output/'plan.json').write_bytes(path.read_bytes())
    (output/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,requirements_sha256=digest(ROOT/'experiments/requirements-research.txt')),indent=2)+'\n')
    upstream=ROOT/'data/control_broadcast_v1/inputs'
    def load_source(name):
        with np.load(upstream/(name+'_observations.npz')) as data:
            return {key:data[key] for key in ('positions','teams','valid')}
    full,broadcast=load_source('tactical'),load_source('broadcast')
    profiles=json.loads((ROOT/'experiments/BROADCAST_ALIGNMENT_PLAN.json').read_text())['alignment']['profiles']
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    products={};summaries={}
    for n in (18935,19060):
        print('Diagnostic source prefix',n,flush=True)
        tactical={key:value[:n] if key in ('positions','valid') else value for key,value in full.items()}
        anchors,trace=trace_anchors(tactical,broadcast,profiles)
        mapping,frames=trace_mapping(anchors,trace,tactical,broadcast,profiles['strict'])
        # Fail closed without exposing any source arrays in an assertion error.
        with np.load(ROOT/f'data/control_broadcast_action_v1/prefix_replay/mapping_{n}.npz') as data:
            for key in mapping:require_equal(mapping[key],data[key],'inherited mapping field '+key)
        if n==19060:
            with np.load(upstream/'anchors.npz') as data:
                for key in anchors:require_equal(anchors[key],data[key],'cached original anchor field '+key)
            with np.load(upstream/'strict_mapping.npz') as data:
                for key in mapping:require_equal(mapping[key],data[key],'cached original strict field '+key)
        np.savez_compressed(output/f'anchors_{n}.npz',**anchors)
        np.savez_compressed(output/f'anchor_trace_{n}.npz',**trace)
        np.savez_compressed(output/f'mapping_{n}.npz',**mapping)
        np.savez_compressed(output/f'frame_trace_{n}.npz',**frames)
        table,extra=additional_consistency(anchors,trace,tactical,broadcast,half_video_blocks(GAME),reserved,held,plan)
        table.to_csv(output/f'consistency_{n}.csv',index=False)
        summary=attribution_summary(anchors,trace,mapping,frames,tactical,broadcast,plan['reporting'])
        summary['additional_consistency']=extra
        (output/f'summary_{n}.json').write_text(json.dumps(summary,indent=2)+'\n')
        products[n]=(anchors,trace,mapping,frames,table);summaries[n]=summary
    a,b=products[18935],products[19060]
    start,stop=plan['reporting']['public_start']-50,plan['reporting']['public_stop']+50
    anchor_masks=[(p[0]['tactical_frames']>=start)&(p[0]['tactical_frames']<stop) for p in (a,b)]
    for key in a[0]:require_equal(a[0][key][anchor_masks[0]],b[0][key][anchor_masks[1]],'anchor prefix field '+key)
    for key in a[1]:require_equal(a[1][key][anchor_masks[0]],b[1][key][anchor_masks[1]],'anchor reason prefix field '+key)
    for key in a[2]:
        if key=='anchor_allowed':require_equal(a[2][key][anchor_masks[0]],b[2][key][anchor_masks[1]],'allowed prefix')
        else:require_equal(a[2][key][start:stop],b[2][key][start:stop],'mapping prefix field '+key)
    for key in a[3]:
        if key.startswith('interval_'):
            masks=[(p[0]['tactical_frames'][:-1]>=start-25)&(p[0]['tactical_frames'][:-1]<stop) for p in (a,b)]
            require_equal(a[3][key][masks[0]],b[3][key][masks[1]],'interval prefix field '+key)
        else:require_equal(a[3][key][start:stop],b[3][key][start:stop],'frame prefix field '+key)
    if list(a[4])!=list(b[4]):raise ValueError('Consistency prefix schema differs')
    for key in a[4]:require_equal(a[4][key].to_numpy(),b[4][key].to_numpy(),'consistency prefix field '+key)
    if summaries[18935]!=summaries[19060]:raise ValueError('Aggregate diagnostic prefix differs')
    summary=summaries[19060]
    prior=json.loads((ROOT/'data/control_alignment_coverage_v1/assessment/summary.json').read_text())
    if summary['frame_first_failure']['mapped']!=prior['strict_mapping_coverage_public']:
        raise ValueError('Prior coverage denominator not reproduced')
    result=dict(round=24,scope='metadata_anchor_failure_diagnosis',plan_sha256=digest(path),diagnostics=summary,
                cached_original_anchors_exact=True,cached_strict_mapping_exact=True,
                both_actual_source_prefix_mappings_exact=True,attribution_and_consistency_prefix_parity=True,
                prefix_context_radius=50,new_images_decoded=0,new_neural_calls=0,new_action_outputs_consumed=0,
                new_possession_targets_read=0,new_fits=0,new_score_rows=0,
                no_new_alignment_or_gate_changes=True,selected_unchanged=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if hashes!={name:digest(ROOT/'experiments'/name) for name in sources}:raise ValueError('Diagnostic source changed during execution')
    validate_inputs()
    manifest=dict(plan_sha256=digest(path),source_sha256=hashes,input_sha256=plan['input_sha256'],
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
