"""Faithful diagnostic trace of frozen formation matching; no possession targets."""
import numpy as np
import pandas as pd
from broadcast_alignment import formation_cost, frame_usable, map_anchors, transform

ANCHOR_REASONS = [
    'prefix_grid_edge','tactical_source_unusable','coarse_neighborhood_nonfinite',
    'prefilter_coarse_geometry','prefilter_far_ambiguity','prefilter_orientation_change',
    'prefilter_coarse_increment','fine_no_usable_triplet','fine_nonfinite_geometry',
    'strict_fine_geometry','strict_fine_uncertainty_span','strict_neighbor_coarse_geometry',
    'strict_neighbor_far_ambiguity','allowed']
INTERVAL_REASONS = ['both_endpoints_rejected','left_endpoint_rejected','right_endpoint_rejected','orientation_change','fine_rate','accepted']
FRAME_REASONS = ['no_incident_anchor_interval','no_allowed_endpoint_pair',
    'no_common_orientation_after_allowed_pair','fine_rate_after_allowed_orientation',
    'broadcast_source_unusable','mapped']


def array_equal(a, b):
    a,b=np.asarray(a),np.asarray(b)
    return np.array_equal(a,b,equal_nan=True) if a.dtype.kind not in 'OUS' else np.array_equal(a,b)


def trace_anchors(tactical, broadcast, profiles):
    """Copy original arithmetic/order; additional arrays explain each gate only."""
    txy,bxy=tactical['positions'],broadcast['positions']
    usable_t,usable_b=frame_usable(tactical),frame_usable(broadcast)
    tf=np.arange(25,len(txy)-25,25)
    bf=np.arange(25,len(bxy)-25,25);bf=bf[usable_b[bf]]
    if not len(bf):raise ValueError('No eligible broadcast formation anchors')
    table=np.full((len(tf),4),np.nan)
    for index,frame in enumerate(tf):
        if not usable_t[frame]:continue
        costs=np.stack([formation_cost(txy[frame],bxy[bf],tactical['teams'],broadcast['teams'],orientation)
                        for orientation in range(4)])
        flattened=int(np.argmin(costs))
        orientation,candidate=np.unravel_index(flattened,costs.shape)
        best=costs[orientation,candidate]
        far=np.abs(bf-bf[candidate])>250
        runner=np.min(costs[:,far]) if far.any() else np.inf
        table[index]=[bf[candidate],orientation,best,runner-best]
    maximum_cost=max(config['maximum_cost'] for config in profiles.values())
    minimum_margin=min(config['minimum_margin'] for config in profiles.values())
    fine=np.full((len(tf),3),np.nan)
    flag_names=['tactical_usable','tactical_homography','coarse_neighbors_finite',
        'prefilter_cost','prefilter_margin','prefilter_orientation','prefilter_increment',
        'fine_finite','fine_cost_strict','fine_span_strict','neighbor_cost_strict',
        'neighbor_margin_strict','own_cost_strict','own_margin_strict']
    flags={name:np.zeros(len(tf),bool) for name in flag_names}
    flags['tactical_usable']=usable_t[tf]
    flags['tactical_homography']=np.asarray(tactical['valid'])[tf].astype(bool)
    reason=np.full(len(tf),'prefix_grid_edge',dtype='<U48')
    candidate_count=np.zeros(len(tf),int);triplet_count=np.zeros(len(tf),int)
    strict=profiles['strict']
    for index in range(1,len(tf)-1):
        neighborhood=table[index-1:index+2]
        flags['coarse_neighbors_finite'][index]=np.isfinite(neighborhood[:,:3]).all()
        flags['prefilter_cost'][index]=not (neighborhood[:,2]>maximum_cost).any()
        flags['prefilter_margin'][index]=not (neighborhood[:,3]<minimum_margin).any()
        flags['prefilter_orientation'][index]=np.all(neighborhood[:,1]==table[index,1])
        flags['prefilter_increment'][index]=not (np.abs(np.diff(neighborhood[:,0])-25)>25).any()
        flags['neighbor_cost_strict'][index]=(neighborhood[:,2]<=strict['maximum_cost']).all()
        flags['neighbor_margin_strict'][index]=(neighborhood[:,3]>=strict['minimum_margin']).all()
        flags['own_cost_strict'][index]=table[index,2]<=strict['maximum_cost']
        flags['own_margin_strict'][index]=table[index,3]>=strict['minimum_margin']
        checks=[('tactical_source_unusable',flags['tactical_usable'][index]),
                ('coarse_neighborhood_nonfinite',flags['coarse_neighbors_finite'][index]),
                ('prefilter_coarse_geometry',flags['prefilter_cost'][index]),
                ('prefilter_far_ambiguity',flags['prefilter_margin'][index]),
                ('prefilter_orientation_change',flags['prefilter_orientation'][index]),
                ('prefilter_coarse_increment',flags['prefilter_increment'][index])]
        failed=next((name for name,passed in checks if not passed),None)
        if failed is not None:
            reason[index]=failed
            continue
        frame,orientation=tf[index],int(table[index,1])
        candidates=np.arange(int(table[index,0])-25,int(table[index,0])+26)
        candidates=candidates[(candidates>=25)&(candidates<len(bxy)-25)]
        candidate_count[index]=len(candidates)
        valid=np.ones(len(candidates),bool)
        costs=[]
        for offset in (-25,0,25):
            valid &= usable_b[candidates+offset]&usable_t[frame+offset]
            value=formation_cost(txy[frame+offset],bxy[candidates+offset],tactical['teams'],broadcast['teams'],orientation)
            value[~usable_b[candidates+offset]|~usable_t[frame+offset]]=np.inf
            costs.append(value**2)
        triplet_count[index]=int(valid.sum())
        score=np.sqrt(np.mean(costs,axis=0))
        best=int(score.argmin())
        if not np.isfinite(score[best]):
            reason[index]='fine_no_usable_triplet' if not valid.any() else 'fine_nonfinite_geometry'
            continue
        possible=candidates[score<=score[best]+.05]
        fine[index]=[candidates[best],score[best],int(possible[-1]-possible[0])]
        flags['fine_finite'][index]=True
        flags['fine_cost_strict'][index]=fine[index,1]<=strict['maximum_cost']
        flags['fine_span_strict'][index]=fine[index,2]<=strict['maximum_fine_span']
        checks=[('strict_fine_geometry',flags['fine_cost_strict'][index]),
                ('strict_fine_uncertainty_span',flags['fine_span_strict'][index]),
                ('strict_neighbor_coarse_geometry',flags['neighbor_cost_strict'][index]),
                ('strict_neighbor_far_ambiguity',flags['neighbor_margin_strict'][index])]
        reason[index]=next((name for name,passed in checks if not passed),'allowed')
    anchors=dict(tactical_frames=tf,coarse=table,fine=fine)
    trace=dict(reason=reason,**flags,fine_candidate_count=candidate_count,fine_usable_triplets=triplet_count)
    return anchors,trace


def trace_mapping(anchors, trace, tactical, broadcast, profile):
    """Classify existence by OR over all incident inclusive intervals."""
    mapping=map_anchors(anchors,tactical,broadcast,profile)
    allowed=mapping['anchor_allowed']
    if not array_equal(allowed,trace['reason']=='allowed'):
        raise ValueError('Anchor diagnostic allowed flags differ from canonical mapper')
    tf,coarse,fine=[anchors[name] for name in ('tactical_frames','coarse','fine')]
    n=len(tactical['positions'])
    incident=np.zeros(n,bool);pair=np.zeros(n,bool);orientation=np.zeros(n,bool);rate_ok=np.zeros(n,bool)
    endpoint_flags=np.zeros((n,len(ANCHOR_REASONS)-1),bool)
    interval_reason=[];interval_rate=[]
    for i in range(len(tf)-1):
        left,right=int(tf[i]),int(tf[i+1])+1
        incident[left:right]=True
        both=bool(allowed[i:i+2].all());same=bool(coarse[i,1]==coarse[i+1,1])
        rate=(fine[i+1,0]-fine[i,0])/(tf[i+1]-tf[i])
        valid_rate=bool(.8<=rate<=1.2)
        pair[left:right] |= both
        orientation[left:right] |= both and same
        rate_ok[left:right] |= both and same and valid_rate
        if not both:
            label='both_endpoints_rejected' if not allowed[i:i+2].any() else 'left_endpoint_rejected' if not allowed[i] else 'right_endpoint_rejected'
        elif not same:label='orientation_change'
        elif not valid_rate:label='fine_rate'
        else:label='accepted'
        interval_reason.append(label);interval_rate.append(rate)
        for j in (i,i+1):
            if not allowed[j]:
                endpoint_flags[left:right,ANCHOR_REASONS.index(trace['reason'][j])]=True
    reason=np.full(n,'no_incident_anchor_interval',dtype='<U48')
    reason[incident]='no_allowed_endpoint_pair'
    reason[pair]='no_common_orientation_after_allowed_pair'
    reason[orientation]='fine_rate_after_allowed_orientation'
    reason[rate_ok]='broadcast_source_unusable'
    reason[mapping['frames']>=0]='mapped'
    if not array_equal(rate_ok & (mapping['frames']>=0),mapping['frames']>=0):
        raise ValueError('Mapped frame has no accepted source interval')
    return mapping,dict(reason=reason,incident=incident,endpoint_pair=pair,
        common_orientation=orientation,rate_ok=rate_ok,endpoint_reason_flags=endpoint_flags,
        interval_reason=np.array(interval_reason,dtype='<U48'),interval_rate=np.array(interval_rate),
        tactical_query_usable=frame_usable(tactical))


def source_availability(source, frames):
    finite=np.isfinite(source['positions'][frames]).all(axis=2)
    counts=[finite[:,source['teams']==team].sum(axis=1) for team in (0,1)]
    homography=np.asarray(source['valid'])[frames].astype(bool)
    return dict(frames=len(frames),homography_unusable=int((~homography).sum()),
        too_few_team0=int((counts[0]<4).sum()),too_few_team1=int((counts[1]<4).sum()),
        usable=int((homography & (counts[0]>=4)&(counts[1]>=4)).sum()))


def numeric_summary(values, quantiles, minimum=5):
    finite=np.asarray(values,dtype=float);finite=finite[np.isfinite(finite)]
    result=dict(finite_count=len(finite))
    if len(finite)>=minimum:
        result['quantiles']={str(q):float(np.quantile(finite,q)) for q in quantiles}
    return result


def attribution_summary(anchors, trace, mapping, frames, tactical, broadcast, rule):
    tf=anchors['tactical_frames'];start,stop=rule['public_start'],rule['public_stop']
    use=(tf>=start)&(tf<stop);public=np.arange(start,stop)
    anchor_counts={r:int((trace['reason'][use]==r).sum()) for r in ANCHOR_REASONS}
    frame_counts={r:int((frames['reason'][public]==r).sum()) for r in FRAME_REASONS}
    if sum(anchor_counts.values())!=int(use.sum()) or sum(frame_counts.values())!=len(public):
        raise ValueError('Attribution categories do not partition diagnostics')
    missing=mapping['frames'][public]<0
    overlap={name:int(frames['endpoint_reason_flags'][public][missing,i].sum()) for i,name in enumerate(ANCHOR_REASONS[:-1])}
    coarse=anchors['coarse'];fine=anchors['fine']
    interior=np.ones(len(tf),bool);interior[[0,-1]]=False
    neighbor_finite=trace['coarse_neighbors_finite']
    prefilter=interior & trace['tactical_usable'] & neighbor_finite
    for key in ('prefilter_cost','prefilter_margin','prefilter_orientation','prefilter_increment'):
        prefilter &= trace[key]
    fine_finite=np.isfinite(fine).all(axis=1)
    flags={}
    for key,value in trace.items():
        if value.dtype!=bool:continue
        evaluated=np.ones(len(tf),bool)
        if key=='coarse_neighbors_finite':evaluated=interior
        elif key.startswith('prefilter_') or key.startswith('neighbor_'):evaluated=interior & neighbor_finite
        elif key=='fine_finite':evaluated=prefilter
        elif key in ('fine_cost_strict','fine_span_strict'):evaluated=fine_finite
        elif key.startswith('own_'):evaluated=np.isfinite(coarse[:,:3]).all(axis=1)
        flags[key]=dict(evaluated=int((evaluated & use).sum()),failed=int((evaluated & use & ~value).sum()))
    strict_fine=np.isfinite(fine).all(axis=1)&trace['fine_cost_strict']&trace['fine_span_strict']
    neighbor_only_cost=use&strict_fine&trace['own_cost_strict']&~trace['neighbor_cost_strict']
    neighbor_only_margin=use&strict_fine&trace['neighbor_cost_strict']&trace['own_margin_strict']&~trace['neighbor_margin_strict']
    public_intervals=(tf[:-1]<stop)&(tf[1:]>=start)
    return dict(public_anchor_count=int(use.sum()),anchor_first_failure=anchor_counts,
        anchor_failed_predicates_overlapping=flags,
        fine_eligible_strict_geometry_but_neighbor_only_cost_veto=int(neighbor_only_cost.sum()),
        fine_eligible_other_strict_checks_but_neighbor_only_margin_veto=int(neighbor_only_margin.sum()),
        public_frame_count=len(public),frame_first_failure=frame_counts,
        unmapped_endpoint_reason_overlap=overlap,
        interval_first_failure={name:int((frames['interval_reason'][public_intervals]==name).sum()) for name in INTERVAL_REASONS},
        mapped_tactical_interior_unusable=int(((mapping['frames'][public]>=0)&~frames['tactical_query_usable'][public]).sum()),
        tactical_public_anchor_availability=source_availability(tactical,tf[use]),
        tactical_public_frame_availability=source_availability(tactical,public),
        broadcast_coarse_grid_availability=source_availability(broadcast,np.arange(25,len(broadcast['positions'])-25,25)))


def additional_consistency(anchors, trace, tactical, broadcast, blocks, reserved, held, plan):
    rule=plan['additional_consistency'];scope=plan['reporting']
    usable_t,usable_b=frame_usable(tactical),frame_usable(broadcast)
    rows=[];skipped=0
    for i,frame in enumerate(anchors['tactical_frames']):
        if not scope['public_start']<=frame<scope['public_stop']:continue
        left,right=frame-50,frame+50
        if left<0 or right>=len(tactical['positions']) or not any(a<=left and right<=b for _,a,b in blocks) or any(left<=r['end']+250 and right>=r['start']-250 for r in reserved) or np.any((np.asarray(held)+250>=left)&(np.asarray(held)-250<=right)):
            skipped+=1;continue
        for kind,index in (('coarse',0),('fine',0)):
            candidate=anchors[kind][i,index]
            if not np.isfinite(candidate):continue
            b=int(candidate);orientation=int(anchors['coarse'][i,1])
            record=dict(anchor_index=i,tactical_frame=int(frame),candidate_kind=kind,
                        strict_allowed=bool(trace['reason'][i]=='allowed'),anchor_reason=str(trace['reason'][i]),
                        current_usable=False,extra_offsets_usable=False,forward=np.nan,reverse=np.nan,balanced_rms=np.nan,extra_offset_worst=np.nan)
            if 0<=b<len(usable_b) and usable_t[frame] and usable_b[b]:
                t=tactical['positions'][frame];other=broadcast['positions'][b:b+1]
                forward=float(formation_cost(t,other,tactical['teams'],broadcast['teams'],orientation)[0])
                reverse=float(formation_cost(transform(other[0],orientation),t[None],broadcast['teams'],tactical['teams'],0)[0])
                record.update(current_usable=True,forward=forward,reverse=reverse,
                              balanced_rms=float(np.sqrt((forward**2+reverse**2)/2)))
            values=[]
            for offset in rule['unused_offsets']:
                bt=b+offset;tt=int(frame)+offset
                if bt<0 or bt>=len(usable_b) or tt<0 or tt>=len(usable_t) or not (usable_b[bt] and usable_t[tt]):break
                values.append(float(formation_cost(tactical['positions'][tt],broadcast['positions'][bt:bt+1],tactical['teams'],broadcast['teams'],orientation)[0]))
            if len(values)==len(rule['unused_offsets']):
                record.update(extra_offsets_usable=True,extra_offset_worst=max(values))
            rows.append(record)
    table=pd.DataFrame(rows)
    aggregates=[]
    if len(table):
        for (kind,allowed),group in table.groupby(['candidate_kind','strict_allowed'],sort=True):
            item=dict(candidate_kind=kind,strict_allowed=bool(allowed),candidates=len(group),
                      current_usable=int(group.current_usable.sum()),extra_offsets_usable=int(group.extra_offsets_usable.sum()))
            for field in ('forward','reverse','balanced_rms','extra_offset_worst'):
                item[field]=numeric_summary(group[field],rule['quantiles'],rule['minimum_numeric_group'])
                item[field+'_above_existing_cutoff']=int((group[field]>rule['diagnostic_cutoff']).sum())
            aggregates.append(item)
    return table,dict(public_anchors_skipped_context_guard=skipped,groups=aggregates,
        interpretation='Correlated internal checks at original winners; not correspondence truth, independent validation, or new acceptance gates.')
