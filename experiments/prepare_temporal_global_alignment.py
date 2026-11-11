"""Frozen Round 26 assessment; fresh outputs only and aggregate console output."""
import argparse
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import scipy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from broadcast_alignment import frame_usable, map_anchors
from temporal_global_alignment import (FIELDS, STRICT, equal, temporal_anchors, extension,
    checkpoints, inverse_schedule, metadata_gate)
from alignment_coverage import inverse_schedule as historical_schedule
from alignment_coverage import query_plan, synthetic_feature_view, coverage_diagnostics
from games import GAME, half_video_blocks

ROUND = ROOT/'data/control_temporal_global_v1'
PLAN = ROOT/'experiments/TEMPORAL_GLOBAL_PLAN.json'
VARIANTS = ('historical', 'baseline', 'candidate', 'extension')


def source_hashes():
    names = json.loads((ROUND/'protocol_freeze/implementation.json').read_text())
    return {name:digest(ROOT/name) for name in names}


def validate_inputs():
    plan = json.loads(PLAN.read_text())
    if PLAN.read_bytes() != (ROUND/'protocol_freeze/plan.json').read_bytes():
        raise ValueError('Frozen temporal protocol changed')
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected: raise ValueError('Frozen input changed: '+name)
    protected = json.loads((ROUND/'protocol_freeze/protected_artifacts.json').read_text())
    for name, expected in protected.items():
        if digest(ROOT/name) != expected: raise ValueError('Protected artifact changed: '+name)
    frozen = json.loads((ROUND/'protocol_freeze/implementation.json').read_text())
    if source_hashes() != frozen: raise ValueError('Frozen execution source changed')
    for base, manifest in (('control_broadcast_action_v1/prefix_replay','summary.json'),
                           ('control_anchor_failures_v1/assessment','manifest.json')):
        record = json.loads((ROOT/'data'/base/manifest).read_text())
        for name, expected in record['source_sha256'].items():
            if digest(ROOT/'experiments'/name) != expected: raise ValueError('Inherited prefix source changed')
        for name, expected in record['cache_sha256'].items():
            if digest(ROOT/'data'/base/name) != expected: raise ValueError('Inherited prefix cache changed')
    for name, target in json.loads((ROUND/'protocol_freeze/symlinks.json').read_text()).items():
        path=ROOT/name
        if not path.is_symlink() or str(path.readlink()) != target: raise ValueError('Source symlink changed')
    if (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes() != (ROUND/'protocol_freeze/ledger_before.csv').read_bytes():
        raise ValueError('Ledger bytes changed')
    return plan


def observations(name):
    with np.load(ROOT/f'data/control_broadcast_v1/inputs/{name}_observations.npz') as data:
        return {k:data[k] for k in ('positions','teams','valid')}


def load_npz(path):
    with np.load(path) as data: return dict(data)


def public_summary(summary):
    # Whole-prefix scheduling totals are separately reconstructed, not equated.
    whole = {'eligibility_counts','eligible_associations','unique_source_centers'}
    return dict(summary, variants={name:{k:v for k,v in values.items() if k not in whole}
                                  for name,values in summary['variants'].items()})


def prefix_parity(a,b,sa,sb):
    anchor_masks=[(p[0]['tactical_frames']>=11110)&(p[0]['tactical_frames']<18760) for p in (a,b)]
    for i in (0,1):
        for k in a[i]: equal(a[i][k][anchor_masks[0]],b[i][k][anchor_masks[1]],'prefix anchor/trace '+k)
    for name in VARIANTS:
        for k in a[2][name]:
            if k=='anchor_allowed':
                equal(a[2][name][k][anchor_masks[0]],b[2][name][k][anchor_masks[1]],'prefix allowed mask')
            else: equal(a[2][name][k][11110:18760],b[2][name][k][11110:18760],'prefix mapping '+k)
        for k in a[3][name]:equal(a[3][name][k][11135:18735],b[3][name][k][11135:18735],'prefix query '+k)
        if not a[4][name].iloc[11185:18685].equals(b[4][name].iloc[11185:18685]):
            raise ValueError('Full121 synthetic feature prefix parity failed')
    if not a[5].equals(b[5]) or public_summary(sa)!=public_summary(sb):
        raise ValueError('Public checkpoint or aggregate prefix parity failed')


def assess(output):
    plan=validate_inputs();output.mkdir(parents=True)
    hashes=source_hashes()
    (output/'source_at_dispatch.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (output/'plan.json').write_bytes(PLAN.read_bytes())
    (output/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,scipy=scipy.__version__,requirements_sha256=digest(ROOT/'experiments/requirements-research.txt')),indent=2)+'\n')
    full,broadcast=observations('tactical'),observations('broadcast')
    ub=frame_usable(broadcast)[:164487]
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    blocks=half_video_blocks(GAME);products={};summaries={}
    for n in (18935,19060):
        print('Temporal global actual source prefix',n,flush=True)
        tactical={k:v[:n] if k in ('positions','valid') else v for k,v in full.items()}
        old=load_npz(ROOT/f'data/control_anchor_failures_v1/assessment/anchors_{n}.npz')
        original=map_anchors(old,tactical,broadcast,STRICT)
        cached=load_npz(ROOT/f'data/control_broadcast_action_v1/prefix_replay/mapping_{n}.npz')
        for k in original:equal(original[k],cached[k],'original strict replay '+k)
        anchors,trace,candidate=temporal_anchors(tactical,broadcast,blocks,reserved,held,
            progress=lambda i,total:print('Coarse progress',n,i,total,flush=True))
        mappings=dict(historical=original,baseline=original,candidate=candidate,extension=extension(original,candidate))
        for name,value in (('anchors',anchors),('trace',trace)):np.savez_compressed(output/f'{name}_{n}.npz',**value)
        queries={};features={};diagnostics={}
        for name,mapping in mappings.items():
            np.savez_compressed(output/f'{name}_mapping_{n}.npz',**mapping)
            rows,counts=(historical_schedule(n,mapping,blocks,reserved,held,ub,25) if name=='historical' else
                         inverse_schedule(n,mapping,blocks,reserved,held,ub))
            q=query_plan(rows,mapping,n);queries[name]=q
            features[name]=synthetic_feature_view(q,mapping,n,blocks)
            if features[name].shape[1]!=121:raise ValueError('Synthetic schema changed')
            (output/f'{name}_schedule_{n}.json').write_text(json.dumps(rows,indent=2)+'\n')
            np.savez_compressed(output/f'{name}_queries_{n}.npz',**q)
            features[name].to_pickle(output/f'{name}_synthetic_features_{n}.pkl')
            diagnostic=coverage_diagnostics(q,queries['historical'],rows,counts)
            diagnostic['mapped_public_frames']=int((mapping['frames'][11185:18685]>=0).sum())
            relevant=[r for r in rows if 11110<=r['tactical_center']<18760]
            diagnostic['public_feature_associations']=len(relevant)
            diagnostic['public_feature_source_centers']=len({r['broadcast_center'] for r in relevant})
            if name in ('candidate','extension'):
                current=q['seen'][11185:18685];paired=queries['baseline']['seen'][11185:18685]
                diagnostic['net_vs_paired_strict100']=int(current.sum()-paired.sum())
                diagnostic['retention_vs_paired_strict100']=float((current&paired).sum()/paired.sum()) if paired.any() else None
            diagnostics[name]=diagnostic
        if diagnostics['historical']['observed_public_frames']!=2328 or diagnostics['historical']['mapped_public_frames']!=3359:
            raise ValueError('Historical baseline not reproduced')
        inherited=load_npz(ROOT/f'data/control_alignment_coverage_v1/assessment/inverse25_{n}_queries.npz')
        for k in inherited:equal(queries['historical'][k],inherited[k],'historical inverse125 query '+k)
        table,quality=checkpoints(dict(baseline=original,candidate=candidate,extension=mappings['extension']),tactical,broadcast,blocks,reserved,held)
        table.to_csv(output/f'checkpoints_{n}.csv',index=False)
        mask=(anchors['tactical_frames']>=11185)&(anchors['tactical_frames']<18685)
        oldcoarse=old['coarse'][mask];newcoarse=anchors['coarse'][mask]
        finite=np.isfinite(oldcoarse[:,:3]).all(axis=1)&np.isfinite(newcoarse[:,:3]).all(axis=1)
        overlap=(original['frames'][11185:18685]>=0)&(candidate['frames'][11185:18685]>=0)
        summary=dict(variants=diagnostics,checkpoints=quality,public_anchors=int(mask.sum()),
            safe_public_anchors=int(trace['safe'][mask].sum()),
            candidate_anchor_first_failure={str(k):int(v) for k,v in zip(*np.unique(trace['reason'][mask],return_counts=True))},
            changed_global_public_winners=int(((newcoarse[finite,0]!=oldcoarse[finite,0])|(newcoarse[finite,1]!=oldcoarse[finite,1])).sum()),
            global_public_comparable_winners=int(finite.sum()),
            candidate_changes_on_old_overlap=int((overlap&((candidate['frames'][11185:18685]!=original['frames'][11185:18685])|
                (candidate['orientation'][11185:18685]!=original['orientation'][11185:18685]))).sum()))
        (output/f'summary_{n}.json').write_text(json.dumps(summary,indent=2)+'\n')
        products[n]=(anchors,trace,mappings,queries,features,table);summaries[n]=summary
    prefix_parity(products[18935],products[19060],summaries[18935],summaries[19060])
    s=summaries[19060]
    gate=metadata_gate(s['variants']['extension'],s['checkpoints'],plan['opportunity_gate'],plan['checkpoints'],True)
    result=dict(round=26,plan_sha256=digest(PLAN),diagnostics=s,provisional_opportunity_gate=gate,
        independent_actual_prefixes=[18935,19060],original_strict_replay_all_fields=True,
        candidate_trace_mapping_query_prefix_parity=True,full121_synthetic_feature_prefix_parity=True,
        mapping_prefix_context_radius=75,query_prefix_context_radius=50,
        all_old_per_frame_mapping_fields_exact=True,extension_has_no_composite_anchor_mask=True,
        tactical_dependency_radius_bound=250,image_content_radius_bound=175,global_broadcast_search=True,
        actual_action_feature_parity_established=False,selected_unchanged=True,new_images=0,
        new_neural_calls=0,new_action_outputs_consumed=0,new_possession_targets_read=0,new_fits=0,new_score_rows=0)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    validate_inputs()
    if hashes!=source_hashes():raise ValueError('Execution source changed during assessment')
    manifest=dict(plan_sha256=digest(PLAN),source_sha256=hashes,input_sha256=plan['input_sha256'],
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(round=26,gate=gate,coverage={k:v['observed_public_frames'] for k,v in s['variants'].items()},
        checkpoints=s['checkpoints']['new_only'],changed_global_public_winners=s['changed_global_public_winners']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROUND/'assessment')
    assess(parser.parse_args().output.resolve())
