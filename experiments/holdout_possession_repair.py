"""Physical stoppage closure,continuous-control bridges and short-spell bounds."""
import numpy as np
from possession_boundary_precision import spans
from dual_anchor_stoppage import refine as unseen_proposals

VARIANTS={
    'physical':dict(stages=('pitch','edge')),
    'continuity':dict(stages=('unseen','bridge')),
    'control':dict(stages=('late','cross')),
    'release':dict(stages=('release',)),
    'combined':dict(stages=('pitch','edge','unseen','bridge','late','cross','release')),
}


def inward(probability,k,start,end,radius=25):
    capacity=min(radius,end-start-8)
    if capacity<=0:return start
    q=probability[start:start+capacity]
    cost=np.r_[0.,np.cumsum(np.log(np.maximum(q[:,k],1e-12)/np.maximum(q[:,1-k],1e-12)))]
    choice=int(np.argmin(cost))
    return start+choice if choice and cost[choice]<-1e-12 and q[:choice,1-k].mean()>.5 else start


def refine(reference,heads,owner,fast,raw,broadcast,blocks,stages=VARIANTS['combined']['stages']):
    """Uniform rules using original inputs;no targets,frame IDs or protocol.

    D may grow or shrink. All actions are local to original short runs and
    original decode blocks. The prior selected recipe supplies the reference;
    changes are proposed independently before the writes below.
    """
    reference=np.asarray(reference);heads=np.asarray(heads);owner=np.asarray(owner);fast=np.asarray(fast)
    n=len(reference);allowed={'pitch','edge','unseen','bridge','late','cross','release'}
    if (reference.ndim!=1 or heads.shape!=(n,2) or owner.shape!=(n,2) or fast.shape!=(n,2)
        or len(raw)!=n or len(broadcast)!=n or not np.isin(reference,['H','A','D','X']).all()
        or not set(stages)<=allowed):raise ValueError('Invalid holdout-repair inputs')
    td=raw[['h__distance_1','a__distance_1']].to_numpy()
    bd=broadcast[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    seen=(broadcast.bc_matched.to_numpy()==1)&(broadcast.bc_same_ball_seen.to_numpy()==1)&np.isfinite(bd).all(1)
    tc=(td<2)&(td[:,::-1]-td>.5);bc=(bd<2)&(bd[:,::-1]-bd>.5)&seen[:,None]
    visible=raw.ball_seen.to_numpy()>0;moving=raw.ball_displacement_1s.to_numpy()>4.5
    outside=(raw.ball_goal_line_distance.to_numpy()<0)|(raw.ball_touchline_distance.to_numpy()<0)
    proposals=[];covered=np.zeros(n,bool)
    for left,right in blocks:
        if not 0<=left<right<=n or covered[left:right].any():raise ValueError('Invalid decode blocks')
        covered[left:right]=True
        for p in (owner[left:right],fast[left:right]):
            if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Invalid owner probabilities')
        if not np.isfinite(heads[left:right]).all() or ((heads[left:right]<0)|(heads[left:right]>1)).any():raise ValueError('Invalid dead probabilities')
        runs=spans(reference[left:right])
        for i,(a,b,label) in enumerate(runs):
            start,end=left+a,left+b;length=end-start
            before=runs[i-1][2] if i else None;after=runs[i+1][2] if i+1<len(runs) else None
            if label=='D' and 'bridge' in stages and length<=125 and before==after and before in ('H','A'):
                k=int(before=='A');lo=max(left,start-13);hi=min(right,end+13)
                if (visible[start:end].mean()>=.5 and moving[start:end].mean()>=.5 and fast[start:end,k].mean()>=.65
                    and tc[lo:hi,k].sum()>=3 and bc[lo:hi,1-k].sum()<3 and not outside[start:end].any()):
                    proposals.append(('control_bridge',start,end,before,{}))
            if label not in ('H','A'):continue
            k=int(label=='A');other='H' if label=='A' else 'A'
            if 'pitch' in stages and length<=250 and 'D' in (before,after) and not seen[start:end].any():
                hits=np.flatnonzero(np.convolve(outside[start:end].astype(int),np.ones(3,dtype=int),'valid')==3) if length>=3 else []
                if len(hits):
                    if before==after=='D':proposals.append(('outside_dead_gap',start,end,'D',{}))
                    elif before==other and after=='D':
                        stop=start+int(hits[0])+2
                        if stop>start:proposals.append(('outside_continuation',start,stop,other,{}))
                        proposals.append(('outside_stoppage',stop,end,'D',{}))
            if ('edge' in stages and length<=125 and (i==0 or i==len(runs)-1) and not seen[start:end].any()
                and visible[start:end].mean()>=.5 and moving[start:end].mean()<=.5 and heads[start:end].mean(0).max()>.5):
                proposals.append(('edge_stoppage',start,end,'D',{}))
            if 'late' in stages and 8<=length<=250 and before==after==other and seen[start:end].all():
                contacts=np.flatnonzero(bc[start:end,k]);groups=np.split(contacts,np.flatnonzero(np.diff(contacts)>5)+1)
                groups=[g for g in groups if len(g)>=3]
                if len(groups)>=2 and groups[-1][0]-groups[-2][-1]>=25:
                    g=groups[-1];lo=start+int(g[0]);hi=start+int(g[-1])+1
                    votes=tc[lo:hi].sum(0);new_start=max(start,lo-5)
                    if (votes[1-k]>=3 and votes[1-k]>votes[k] and end-new_start>=8
                        and owner[start:new_start,1-k].mean()>.5):
                        proposals.append(('terminal_contact',start,new_start,other,{}))
            if 'cross' in stages and 8<=length<=250 and before==after==other:
                tv=tc[start:end].sum(0);bv=bc[start:end].sum(0)
                if tv[k]>=3 and tv[k]>tv[1-k] and bv[1-k]>=3 and bv[1-k]>bv[k]:
                    new_start=inward(owner,k,start,end)
                    reversed_fast=fast[start:end][::-1]
                    cut=inward(reversed_fast,k,0,length)
                    new_end=end-cut
                    if new_end-new_start>=8:
                        if new_start>start:proposals.append(('cross_source_arrival',start,new_start,other,{}))
                        if new_end<end:proposals.append(('cross_source_release',new_end,end,other,{}))
            if 'release' not in stages or not 8<=length<=125 or bc[start:end,k].sum()<3:continue
            if after=='D':
                hi=min(right,end+25,left+runs[i+1][1]);points=np.flatnonzero(bc[max(start,end-13):hi,k])
                if len(points):
                    last=max(start,end-13)+int(points[-1])+1;finish=min(hi,last+13)
                    if finish>end:proposals.append(('contact_dead_release',end,finish,label,{}))
                if length<50 and before==other:
                    lo=max(left,start-13);hi=min(right,start+13);points=np.flatnonzero(tc[lo:hi,k])
                    recent=np.flatnonzero(tc[lo:start,k])
                    if len(points)>=3 and len(recent) and lo+int(recent[-1])>=start-5:
                        begin=max(left,start-25,lo+int(points[0])-5)
                        if begin<start and np.all(reference[begin:start]==other):
                            proposals.append(('short_contact_arrival',begin,start,label,{}))
            if i+3<len(runs):
                s0,e0,next_label=runs[i+1];s1,e1,trace_label=runs[i+2];s2,e2,dead_label=runs[i+3]
                ns,ne=left+s0,left+e0;trace_end=left+e1
                if (next_label==other and trace_label==label and dead_label=='D' and 8<=ne-ns<=125 and e1-s1<8
                    and bc[ns:ne,1-k].sum()<3 and owner[ns:ne,1-k].mean()<.65):
                    last=start+int(np.flatnonzero(bc[start:end,k])[-1])+1
                    finish=min(ns+25,last+30,trace_end)
                    if finish>=ns and trace_end-finish>=8:
                        if finish>ns:proposals.append(('unsupported_release_tail',ns,finish,label,{}))
                        proposals.append(('unsupported_release_stoppage',finish,trace_end,'D',{}))
    if 'unseen' in stages:
        _,events=unseen_proposals(reference,heads,raw,broadcast,blocks)
        for event in events:
            if event['kind']!='dual_anchor_closure':continue
            start,end=event['start'],event['end_exclusive'];a,b=event['dead_start'],event['dead_end_exclusive']
            if b==end:a=max(start,a-5)
            elif a==start:b=min(end,b+5)
            else:raise AssertionError('Unseen closure is not D-facing')
            proposals.append(('unseen_continuation',start,end,event['replacement'],{}))
            proposals.append(('unseen_dead_allowance',a,b,'D',{}))
    result=reference.copy();events=[]
    for kind,start,end,label,info in proposals:
        if end<=start:continue
        result[start:end]=label;events.append(dict(kind=kind,start=start,end_exclusive=end,label=label,**info))
    np.testing.assert_array_equal(result=='X',reference=='X')
    return result,events
