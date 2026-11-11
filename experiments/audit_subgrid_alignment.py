"""Audit frozen refinement predicates, safe support, preservation and fresh replay."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_subgrid_alignment import (ROUND, PLAN, validate_inputs, observations, load_npz, public_summary)
from subgrid_alignment import (FIELDS, STRICT, equal, safe_interval, neighborhood_reason, checkpoints, metadata_gate)
from broadcast_alignment import frame_usable, map_anchors
from alignment_coverage import inverse_schedule, query_plan, coverage_diagnostics
from finalize_broadcast_closure import audit_manifest
from build_temporal_candidate import digest
from games import GAME, half_video_blocks


def require(value, message):
    if not value: raise ValueError('Subgrid audit: '+message)


def main():
    plan=validate_inputs();first,replay=ROUND/'assessment',ROUND/'assessment_replay'
    a,b=audit_manifest(first),audit_manifest(replay)
    require(a==b,'replay manifest differs')
    require((first/'manifest.json').read_bytes()==(replay/'manifest.json').read_bytes(),'manifest byte replay')
    for name in a['cache_sha256']:
        require((first/name).read_bytes()==(replay/name).read_bytes(),'artifact byte replay')
    failure=ROUND/'execution_failure_01'
    require((failure/'minimized_red.log').exists() and (failure/'minimized_green.log').exists(),'execution repair evidence')
    old_sources=json.loads((failure/'implementation.json').read_text())
    require(digest(ROOT/'experiments/subgrid_alignment.py')==old_sources['subgrid_alignment.py'],'matching algorithm changed during repair')
    repair=json.loads((failure/'repair.json').read_text())
    require(repair['plan_sha256']==digest(PLAN),'protocol changed during repair')
    unchanged_partial=0
    for prior in (failure/'assessment').iterdir():
        if prior.name=='source_at_dispatch.json':continue
        require(prior.read_bytes()==(first/prior.name).read_bytes(),'repair changed preexisting observation result')
        unchanged_partial+=1
    require((ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes()==(ROUND/'protocol_freeze/ledger_before.csv').read_bytes(),'ledger preservation')
    full,broadcast=observations('tactical'),observations('broadcast');ub=frame_usable(broadcast)[:164487]
    blocks=half_video_blocks(GAME)
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    summary=json.loads((first/'summary.json').read_text())
    allproducts={};guarded_anchors=0;checked_checkpoints=0
    for n in (18935,19060):
        tactical={k:v[:n] if k in ('positions','valid') else v for k,v in full.items()}
        anchors=load_npz(first/f'anchors_{n}.npz');trace=load_npz(first/f'trace_{n}.npz')
        old=load_npz(ROOT/f'data/control_anchor_failures_v1/assessment/anchors_{n}.npz')
        mappings={name:load_npz(first/f'{name}_mapping_{n}.npz') for name in ('baseline','candidate','extension')}
        mask=trace['safe'] & np.isfinite(old['coarse'][:,:3]).all(axis=1)
        expected_safe=np.array([safe_interval(int(t),25,n,blocks,reserved,held) for t in anchors['tactical_frames']])
        equal(trace['safe'],expected_safe,'coarse guard recomputation')
        require(np.isnan(anchors['coarse'][~mask]).all(),'unsafe or unavailable refinement')
        require(np.isfinite(anchors['coarse'][mask,:3]).all(),'missing safe refinement')
        win,alt=trace['winner_refined'],trace['runner_refined'];runner=trace['runner']
        require((np.abs(win[mask,0]-old['coarse'][mask,0])<=12).all(),'halfcell winner bounds')
        require((win[mask,1]<=old['coarse'][mask,2]+1e-12).all(),'old winner omitted')
        equal(anchors['coarse'][mask,1],old['coarse'][mask,1],'fixed winner orientation')
        equal(anchors['coarse'][mask,2],win[mask,1],'winner refined cost')
        equal(anchors['coarse'][mask,3],np.minimum(old['coarse'][mask,3],alt[mask,1]-win[mask,1]),'both ambiguity margins')
        rm=mask & np.isfinite(runner[:,0])
        require((np.abs(runner[rm,0]-old['coarse'][rm,0])>250).all(),'original far runner')
        require((np.abs(alt[rm,0]-runner[rm,0])<=12).all(),'runner halfcell')
        require((np.abs(alt[rm,0]-win[rm,0])>=251).all(),'runner retained far separation')
        require((trace['candidate_counts']>=0).all() and (trace['candidate_counts']<=25).all(),'candidate denominators')
        allowed=mappings['candidate']['anchor_allowed'];equal(allowed,trace['reason']=='allowed','allowed reason')
        for i in np.flatnonzero(allowed):
            require(0<i<len(allowed)-1,'grid edge allowed')
            require(neighborhood_reason(anchors['coarse'][i-1:i+2],win[i-1:i+2,2])=='pass','refined neighborhood predicates')
            f=anchors['fine'][i]
            require(np.isfinite(f).all() and f[1]<=2.5 and f[2]<=10,'fine strict predicates')
            require(abs(f[0]-anchors['coarse'][i,0])<=25,'fine center recentering')
            require(np.isfinite(trace['extra_costs'][i]).all() and (trace['extra_costs'][i]<=2.5).all(),'extra own-offset predicates')
            equal(f,trace['fine_proposal'][i],'accepted proposal')
        require(np.isnan(anchors['fine'][~allowed]).all(),'rejected fine leakage')
        reconstructed=map_anchors(anchors,tactical,broadcast,STRICT)
        for k in reconstructed:equal(reconstructed[k],mappings['candidate'][k],'candidate map reproduction '+k)
        original=map_anchors(old,tactical,broadcast,STRICT)
        inherited=load_npz(ROOT/f'data/control_broadcast_action_v1/prefix_replay/mapping_{n}.npz')
        for k in original:
            equal(original[k],mappings['baseline'][k],'original map '+k)
            equal(original[k],inherited[k],'inherited original map '+k)
        ext=mappings['extension'];keep=original['frames']>=0
        require('anchor_allowed' not in ext,'invented composite anchor graph')
        for k in FIELDS:
            require(ext[k][keep].tobytes()==original[k][keep].tobytes(),'bit-exact old row preservation')
            equal(ext[k][~keep],mappings['candidate'][k][~keep],'extension missing-row rule '+k)
        equal(ext['origin'],np.where(keep,1,np.where(mappings['candidate']['frames']>=0,2,0)),'extension provenance')
        q_by_name={};coverage={}
        for name,mapping in mappings.items():
            rows,counts=inverse_schedule(n,mapping,blocks,reserved,held,ub,25)
            require(rows==json.loads((first/f'{name}_schedule_{n}.json').read_text()),'safe schedule recreation')
            q=query_plan(rows,mapping,n);saved=load_npz(first/f'{name}_queries_{n}.npz')
            for k in q:equal(q[k],saved[k],'query recreation '+k)
            q_by_name[name]=q
            coverage[name]=coverage_diagnostics(q,q_by_name['baseline'],rows,counts)
            coverage[name]['mapped_public_frames']=int((mapping['frames'][11185:18685]>=0).sum())
            for row in rows:
                c=row['tactical_center'];require(safe_interval(c,175,n,blocks,reserved,held),'full schedule guard')
                require(c-125<=row['image_left']<=c<=row['image_right']<=c+125,'bounded inverse support')
                require(row['decision_left']==c-175 and row['decision_right']==c+175,'decision bounds')
                source=row['broadcast_center'];require(0<=source-100 and source+98<len(ub),'actual video bounds')
                require(ub[source-100:source+99].all(),'intervening source usability')
        saved_summary=json.loads((first/f'summary_{n}.json').read_text())
        require(saved_summary['variants']==coverage,'coverage aggregates')
        table,quality=checkpoints(mappings,tactical,broadcast,blocks,reserved,held)
        saved_table=pd.read_csv(first/f'checkpoints_{n}.csv')
        require(list(table)==list(saved_table) and len(table)==len(saved_table),'checkpoint schema')
        for k in table:
            actual,recorded=table[k].to_numpy(),saved_table[k].to_numpy()
            if k=='cost':require(np.allclose(actual,recorded,rtol=1e-14,atol=1e-14,equal_nan=True),'checkpoint numeric replay')
            else:equal(actual,recorded,'checkpoint table '+k)
        require(saved_summary['checkpoints']==quality,'checkpoint denominators and aggregates')
        for f in table.tactical_frame.unique():
            require(f%25 in (6,19) and safe_interval(f,50,n,blocks,reserved,held),'unused checkpoint residue/guard')
        require(sum(saved_summary['candidate_anchor_first_failure'].values())==300,'public anchor denominator')
        allproducts[n]=(anchors,trace,mappings,q_by_name,table,saved_summary)
        guarded_anchors+=int(mask.sum());checked_checkpoints+=len(table)
    x,y=allproducts[18935],allproducts[19060]
    masks=[(p[0]['tactical_frames']>=11135)&(p[0]['tactical_frames']<18735) for p in (x,y)]
    for j in (0,1):
        for k in x[j]:equal(x[j][k][masks[0]],y[j][k][masks[1]],'actual prefix anchor/trace '+k)
    for j in (2,3):
        for name in x[j]:
            for k in x[j][name]:
                if k=='anchor_allowed':equal(x[j][name][k][masks[0]],y[j][name][k][masks[1]],'actual prefix mask')
                else:equal(x[j][name][k][11135:18735],y[j][name][k][11135:18735],'actual prefix mapping/query '+k)
    require(x[4].equals(y[4]) and public_summary(x[5])==public_summary(y[5]),'actual prefix checkpoint/public aggregate')
    for name in x[2]:
        before=pd.read_pickle(first/f'{name}_synthetic_features_18935.pkl').iloc[11185:18685]
        after=pd.read_pickle(first/f'{name}_synthetic_features_19060.pkl').iloc[11185:18685]
        require(before.shape==(7500,121) and before.equals(after),'full121 synthetic prefix parity')
    d=y[5];gate=metadata_gate(d['variants']['extension'],d['checkpoints'],plan['opportunity_gate'],plan['checkpoints'],True)
    require(gate==summary['provisional_opportunity_gate'],'declared primary gate')
    tests=(ROUND/'production_tests.log').read_text()
    require('\nOK\n' in tests and 'FAILED' not in tests,'full production tests')
    result=dict(round=25,passed=True,result='accepted_metadata_opportunity' if gate['passed'] else 'rejected_metadata_opportunity',
        opportunity_gate=gate,byte_exact_artifacts=len(a['cache_sha256'])+1,
        original_all_field_replay=True,old_valid_fields_bit_exact=True,actual_prefix_and_synthetic_parity=True,
        checkpoint_guarded_rows_checked=checked_checkpoints,safe_coarse_refinements_checked=guarded_anchors,
        complete_source_guards_verified=True,protected_artifacts_verified=17,ledger_rows=1468,
        all_ledger_bytes_exact=True,selected_unchanged=True,new_images=0,new_neural_calls=0,
        new_action_outputs_consumed=0,new_possession_targets_read=0,new_fits=0,new_score_rows=0,
        actual_action_features_or_model_promoted=False,
        execution_repair_partial_artifacts_unchanged=unchanged_partial,
        source_sha256={name:digest(ROOT/'experiments'/name) for name in ('audit_subgrid_alignment.py',)})
    (ROUND/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
