"""Arrival timing from camera disagreement or incomplete-source likelihood."""
import numpy as np
from possession_boundary_precision import spans

VARIANTS={
    'camera_contact5':dict(camera=True,posterior_radius=0),
    'sparse_owner13':dict(camera=False,posterior_radius=13),
    'sparse_owner25':dict(camera=False,posterior_radius=25),
    'camera_owner13':dict(camera=True,posterior_radius=13),
    'camera_owner25':dict(camera=True,posterior_radius=25),
}


def refine(reference,owner,raw,broadcast,blocks,camera=True,posterior_radius=13):
    """Move only H/A arrival boundaries inward; preserve D/X, labels and order.

    Complete broadcast spans may use a5-frame contact-arrival envelope only
    when broadcast contacts favor the current team and tactical contacts favor
    its opponent, with at least3votes per source. Incomplete spans may instead
    minimize saved binary-owner log loss within13or25frames; the replacement
    must also have arithmetic mean probability>.5. No fitting or truth input.
    """
    reference=np.asarray(reference);owner=np.asarray(owner)
    n=len(reference)
    if (reference.ndim!=1 or owner.shape!=(n,2) or len(raw)!=n or len(broadcast)!=n
        or posterior_radius not in (0,13,25) or not isinstance(camera,bool)
        or not np.isin(reference,['H','A','D','X']).all()):raise ValueError('Invalid arrival-refinement inputs')
    td=raw[['h__distance_1','a__distance_1']].to_numpy()
    bd=broadcast[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    observed=(broadcast.bc_matched.to_numpy()==1)&(broadcast.bc_same_ball_seen.to_numpy()==1)&np.isfinite(bd).all(1)
    tc=(td<2)&(td[:,::-1]-td>.5)
    bc=(bd<2)&(bd[:,::-1]-bd>.5)&observed[:,None]
    result=reference.copy();events=[];covered=np.zeros(n,bool)
    for left,right in blocks:
        if not 0<=left<right<=n or covered[left:right].any():raise ValueError('Invalid or overlapping blocks')
        covered[left:right]=True;p=owner[left:right]
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Invalid ownership probabilities')
        original=spans(reference[left:right])
        for index,(a,b,team) in enumerate(original):
            if index==0 or index==len(original)-1 or team not in ('H','A') or not 8<=b-a<=250:continue
            other='H' if team=='A' else 'A'
            if original[index-1][2]!=other:continue
            start,end=left+a,left+b;k=int(team=='A');shift=0;reason=None;info={}
            complete=bool(observed[start:end].all())
            if camera and complete:
                cv=bc[start:end].sum(0);tv=tc[start:end].sum(0)
                if cv[k]>=3 and tv[1-k]>=3 and cv[k]>cv[1-k] and tv[1-k]>tv[k]:
                    first=int(np.flatnonzero(bc[start:end,k])[0])
                    shift=min(25,end-start-8,max(0,first-5));reason='camera_disagreement'
                    info=dict(broadcast_contacts=cv.tolist(),tactical_contacts=tv.tolist())
            elif posterior_radius and not complete:
                capacity=min(posterior_radius,end-start-8)
                if capacity>0:
                    local=owner[start:start+capacity]
                    difference=np.log(np.maximum(local[:,k],1e-12)/np.maximum(local[:,1-k],1e-12))
                    cost=np.r_[0.,np.cumsum(difference)];choice=int(np.argmin(cost))
                    if choice and cost[choice]<-1e-12 and owner[start:start+choice,1-k].mean()>.5:
                        shift=choice;reason='incomplete_source_posterior'
                        info=dict(log_loss_gain=float(-cost[choice]),mean_replacement_probability=float(owner[start:start+choice,1-k].mean()))
            if shift:
                result[start:start+shift]=other
                events.append(dict(start=start,end_exclusive=end,new_start=start+shift,new_end_exclusive=end,
                                   team=team,kind=reason,**info))
        after=spans(result[left:right])
        if [x[2] for x in after]!=[x[2] for x in original] or any(e-s<min(b-a,8) for (s,e,_),(a,b,_) in zip(after,original)):
            raise AssertionError('Arrival refinement changed topology')
    for state in ('D','X'):np.testing.assert_array_equal(result==state,reference==state)
    return result,events
