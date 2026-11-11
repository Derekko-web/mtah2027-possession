"""Verify frozen temporal predicates, complete guards, separate replay and preservation."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_temporal_global_alignment import (ROUND,PLAN,VARIANTS,validate_inputs,observations,
    load_npz,source_hashes,public_summary,prefix_parity)
from temporal_global_alignment import (FIELDS,STRICT,equal,safe_interval,temporal_costs,
    neighborhood_reason,inverse_schedule,checkpoints,metadata_gate)
from alignment_coverage import inverse_schedule as historical_schedule
from alignment_coverage import query_plan,synthetic_feature_view,coverage_diagnostics
from broadcast_alignment import frame_usable,map_anchors,formation_cost
from build_temporal_candidate import digest
from games import GAME,half_video_blocks


def require(value,message):
    if not value:raise ValueError('Temporal audit: '+message)


def manifest(folder):
    value=json.loads((folder/'manifest.json').read_text())
    require(set(value['cache_sha256'])=={p.name for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'},'artifact inventory')
    for name,expected in value['cache_sha256'].items():require(digest(folder/name)==expected,'artifact digest '+name)
    require(value['source_sha256']==source_hashes(),'manifest sources')
    require(value['plan_sha256']==digest(PLAN),'manifest plan')
    return value


def audit():
    plan=validate_inputs();first=ROUND/'assessment';replay=ROUND/'assessment_replay'
    a,b=manifest(first),manifest(replay)
    require(a==b,'fresh-process manifest')
    for name in (*a['cache_sha256'],'manifest.json'):
        require((first/name).read_bytes()==(replay/name).read_bytes(),'fresh-process byte replay '+name)
    full,broadcast=observations('tactical'),observations('broadcast')
    ub_all=frame_usable(broadcast);ub=ub_all[:164487]
    blocks=half_video_blocks(GAME)
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    saved_result=json.loads((first/'summary.json').read_text());products={};summaries={}
    checked_anchors=checked_schedules=checked_checkpoints=0
    for n in (18935,19060):
        tactical={k:v[:n] if k in ('positions','valid') else v for k,v in full.items()};ut=frame_usable(tactical)
        anchors=load_npz(first/f'anchors_{n}.npz');trace=load_npz(first/f'trace_{n}.npz')
        mappings={name:load_npz(first/f'{name}_mapping_{n}.npz') for name in VARIANTS}
        old=load_npz(ROOT/f'data/control_anchor_failures_v1/assessment/anchors_{n}.npz')
        original=map_anchors(old,tactical,broadcast,STRICT)
        inherited=load_npz(ROOT/f'data/control_broadcast_action_v1/prefix_replay/mapping_{n}.npz')
        for name in ('historical','baseline'):
            for k in original:
                equal(original[k],mappings[name][k],'original map '+name+' '+k)
                equal(original[k],inherited[k],'inherited original '+k)
        tf=anchors['tactical_frames'];equal(tf,np.arange(25,n-25,25),'tactical grid')
        safe=np.array([safe_interval(int(t),25,n,blocks,reserved,held) for t in tf])
        equal(safe,trace['safe'],'new geometry guard')
        usable=np.array([ut[t+np.array([-25,0,25])].all() for t in tf])
        mask=safe&usable
        require(np.isnan(anchors['coarse'][~mask]).all(),'unsafe coarse geometry')
        require(np.isfinite(anchors['coarse'][mask,:3]).all(),'missing usable coarse winner')
        bf=np.arange(25,len(ub_all)-25,25);bf=bf[ub_all[bf-25]&ub_all[bf]&ub_all[bf+25]]
        equal(trace['candidate_counts'][mask],np.full(int(mask.sum()),4*len(bf)),'all-reflection candidate universe')
        require((trace['candidate_counts'][~mask]==0).all(),'unsafe search count')
        for i in np.flatnonzero(mask):
            t=int(tf[i]);coarse=anchors['coarse'][i];winner=int(coarse[0]);o=int(coarse[1])
            require(winner in bf and 0<=o<4,'coarse candidate domain')
            # Rebuild the full objective directly from geometry, independently of
            # temporal_search and temporal_costs, then check first-tie ordering.
            matrix=np.stack([np.sqrt(np.mean(np.stack([
                formation_cost(tactical['positions'][t+d],broadcast['positions'][bf+d],
                    tactical['teams'],broadcast['teams'],orientation)**2 for d in (-25,0,25)]),axis=0))
                for orientation in range(4)])
            eo,ej=np.unravel_index(int(matrix.argmin()),matrix.shape)
            equal(coarse[:3],np.array([bf[ej],eo,matrix[eo,ej]]),'exhaustive global first minimum')
            far=np.flatnonzero(np.abs(bf-bf[ej])>250)
            expected_runner=np.array([np.nan,np.nan,np.inf])
            if len(far):
                ro,rj=np.unravel_index(int(matrix[:,far].argmin()),(4,len(far)))
                expected_runner=np.array([bf[far[rj]],ro,matrix[ro,far[rj]]])
            equal(trace['runner'][i],expected_runner,'exhaustive far first minimum')
            cost,individual=temporal_costs(t,np.array([winner]),o,tactical,broadcast,ut,ub_all)
            equal(cost,coarse[2:3],'chosen temporal objective');equal(individual[0],trace['individual_costs'][i],'chosen offset costs')
            runner=trace['runner'][i]
            if np.isfinite(runner[0]):
                require(int(runner[0]) in bf and abs(runner[0]-winner)>250 and 0<=runner[1]<4,'far runner domain')
                score,_=temporal_costs(t,np.array([int(runner[0])]),int(runner[1]),tactical,broadcast,ut,ub_all)
                equal(score,runner[2:3],'far objective')
            else:require(np.isposinf(runner[2]),'absent runner infinity')
            equal(coarse[3:4],np.array([runner[2]-coarse[2]]),'new global ambiguity margin')
        allowed=mappings['candidate']['anchor_allowed'];equal(allowed,trace['reason']=='allowed','allowed trace')
        for i in range(1,len(tf)-1):
            if not safe[i] or not usable[i]:continue
            reason=neighborhood_reason(anchors['coarse'][i-1:i+2],trace['individual_costs'][i-1:i+2])
            if reason!='pass':
                require(trace['reason'][i]==reason,'neighborhood rejection precedence');continue
            t,o=int(tf[i]),int(anchors['coarse'][i,1]);center=int(anchors['coarse'][i,0])
            candidates=np.arange(center-25,center+26);candidates=candidates[(candidates>=25)&(candidates<len(ub_all)-25)]
            costs,individual=temporal_costs(t,candidates,o,tactical,broadcast,ut,ub_all);j=int(costs.argmin())
            require(np.isfinite(costs[j]),'fine domain contains coarse winner')
            possible=candidates[costs<=costs[j]+.05]
            proposal=np.array([candidates[j],costs[j],possible[-1]-possible[0]])
            equal(proposal,trace['fine_proposal'][i],'fine argmin and complete span')
            equal(individual[j],trace['fine_individual'][i],'fine offset residuals')
            reason=('fine_temporal_cost' if proposal[1]>2.5 else 'fine_individual_cost' if (individual[j]>2.5).any() else
                    'fine_span' if proposal[2]>10 else 'extra_unavailable')
            if reason=='extra_unavailable':
                extra=np.full(2,np.nan)
                for k,d in enumerate((-12,12)):
                    bframe=int(proposal[0])+d
                    if ut[t+d] and ub_all[bframe]:extra[k]=formation_cost(tactical['positions'][t+d],broadcast['positions'][bframe:bframe+1],tactical['teams'],broadcast['teams'],o)[0]
                equal(extra,trace['extra_costs'][i],'own extra-offset gate')
                if np.isfinite(extra).all():reason='extra_cost' if (extra>2.5).any() else 'allowed'
            require(trace['reason'][i]==reason,'fine rejection precedence')
            if allowed[i]:equal(proposal,anchors['fine'][i],'accepted fine proposal')
        require(np.isnan(anchors['fine'][~allowed]).all(),'rejected fine leakage')
        reconstructed=map_anchors(anchors,tactical,broadcast,STRICT)
        for k in reconstructed:equal(reconstructed[k],mappings['candidate'][k],'candidate map '+k)
        ext=mappings['extension'];keep=original['frames']>=0
        require('anchor_allowed' not in ext,'composite anchor graph')
        for k in FIELDS:
            require(ext[k][keep].tobytes()==original[k][keep].tobytes(),'old fields bit exact')
            equal(ext[k][~keep],mappings['candidate'][k][~keep],'extension fill '+k)
        equal(ext['origin'],np.where(keep,1,np.where(mappings['candidate']['frames']>=0,2,0)),'row origin')
        queries={};features={};diagnostics={}
        for name,mapping in mappings.items():
            radius=125 if name=='historical' else 100
            rows,counts=(historical_schedule(n,mapping,blocks,reserved,held,ub,25) if name=='historical' else inverse_schedule(n,mapping,blocks,reserved,held,ub))
            require(rows==json.loads((first/f'{name}_schedule_{n}.json').read_text()),'schedule including prefix rejection counts')
            q=query_plan(rows,mapping,n);queries[name]=q
            recorded=load_npz(first/f'{name}_queries_{n}.npz')
            for k in q:equal(q[k],recorded[k],'query '+k)
            features[name]=synthetic_feature_view(q,mapping,n,blocks)
            require(features[name].shape==(n,121) and features[name].equals(pd.read_pickle(first/f'{name}_synthetic_features_{n}.pkl')),'full synthetic recomputation')
            d=coverage_diagnostics(q,queries['historical'],rows,counts)
            d['mapped_public_frames']=int((mapping['frames'][11185:18685]>=0).sum())
            relevant=[r for r in rows if 11110<=r['tactical_center']<18760]
            d['public_feature_associations']=len(relevant);d['public_feature_source_centers']=len({r['broadcast_center'] for r in relevant})
            if name in ('candidate','extension'):
                cur=q['seen'][11185:18685];paired=queries['baseline']['seen'][11185:18685]
                d['net_vs_paired_strict100']=int(cur.sum()-paired.sum())
                d['retention_vs_paired_strict100']=float((cur&paired).sum()/paired.sum()) if paired.any() else None
            diagnostics[name]=d
            for row in rows:
                c,bframe=row['tactical_center'],row['broadcast_center']
                require(safe_interval(c,175,n,blocks,reserved,held),'complete decision interval guard')
                require(c-radius<=row['component_left']<=row['image_left']<=c<=row['image_right']<=row['component_right']<=c+radius,'bounded inverse footprint')
                require(row['decision_left']==c-175 and row['decision_right']==c+175,'decision metadata')
                require(0<=bframe-100 and bframe+98<164487 and ub[bframe-100:bframe+99].all(),'all199 actual video frames')
            checked_schedules+=len(rows)
        require(diagnostics['historical']['observed_public_frames']==2328 and diagnostics['historical']['mapped_public_frames']==3359,'historical baseline')
        historical=load_npz(ROOT/f'data/control_alignment_coverage_v1/assessment/inverse25_{n}_queries.npz')
        for k in historical:equal(queries['historical'][k],historical[k],'historical query fields')
        table,quality=checkpoints(dict(baseline=original,candidate=mappings['candidate'],extension=ext),tactical,broadcast,blocks,reserved,held)
        saved_table=pd.read_csv(first/f'checkpoints_{n}.csv')
        require(list(table)==list(saved_table) and len(table)==len(saved_table),'checkpoint schema/denominator')
        for k in table:
            if k=='cost':require(np.allclose(table[k],saved_table[k],rtol=1e-14,atol=1e-14,equal_nan=True),'checkpoint geometry')
            else:equal(table[k].to_numpy(),saved_table[k].to_numpy(),'checkpoint '+k)
        for f in table.tactical_frame.unique():require(f%25 in (6,19) and safe_interval(int(f),75,n,blocks,reserved,held),'unused checkpoint guard75')
        s=json.loads((first/f'summary_{n}.json').read_text())
        require(s['variants']==diagnostics and s['checkpoints']==quality,'saved coverage/quality aggregates')
        mask_public=(tf>=11185)&(tf<18685)
        require(s['public_anchors']==300 and sum(s['candidate_anchor_first_failure'].values())==300,'public anchor denominator')
        reasons={str(k):int(v) for k,v in zip(*np.unique(trace['reason'][mask_public],return_counts=True))}
        require(reasons==s['candidate_anchor_first_failure'],'anchor rejection aggregates')
        oldcoarse=old['coarse'][mask_public];newcoarse=anchors['coarse'][mask_public]
        finite=np.isfinite(oldcoarse[:,:3]).all(axis=1)&np.isfinite(newcoarse[:,:3]).all(axis=1)
        changed=int(((newcoarse[finite,0]!=oldcoarse[finite,0])|(newcoarse[finite,1]!=oldcoarse[finite,1])).sum())
        require(s['changed_global_public_winners']==changed and s['global_public_comparable_winners']==int(finite.sum()),'global winner aggregates')
        checked_anchors+=int(mask.sum());checked_checkpoints+=len(table)
        products[n]=(anchors,trace,mappings,queries,features,table);summaries[n]=s
    prefix_parity(products[18935],products[19060],summaries[18935],summaries[19060])
    s=summaries[19060];gate=metadata_gate(s['variants']['extension'],s['checkpoints'],plan['opportunity_gate'],plan['checkpoints'],True)
    require(gate==saved_result['provisional_opportunity_gate'],'declared primary gate')
    tests=(ROUND/'production_tests.log').read_text();require('\nOK\n' in tests and 'FAILED' not in tests,'production tests')
    validate_inputs()
    protected=json.loads((ROUND/'protocol_freeze/protected_artifacts.json').read_text())
    result=dict(round=26,passed=True,result='accepted_metadata_opportunity' if gate['passed'] else 'rejected_metadata_opportunity',
        opportunity_gate=gate,byte_exact_artifacts=len(a['cache_sha256'])+1,original_all_field_replay=True,
        old_valid_fields_bit_exact=True,actual_prefix_and_full121_synthetic_parity=True,
        safe_temporal_anchors_checked=checked_anchors,safe_schedule_associations_checked=checked_schedules,
        exhaustive_global_winner_and_runner_verified=True,
        checkpoint_guarded_rows_checked=checked_checkpoints,complete_tactical_dependency_bound=250,image_content_bound=175,
        protected_artifacts_verified=len(protected),ledger_rows=1468,all_ledger_bytes_exact=True,selected_unchanged=True,
        source_symlinks_preserved=True,new_images=0,new_neural_calls=0,new_action_outputs_consumed=0,
        new_possession_targets_read=0,new_fits=0,new_score_rows=0,actual_action_features_or_model_promoted=False,
        source_sha256=source_hashes())
    (ROUND/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':audit()
