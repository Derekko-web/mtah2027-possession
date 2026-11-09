"""Observation-based arrival/release boundaries with fixed possession topology."""
import numpy as np

VARIANTS={
    'release_envelope':dict(leading=13,trailing=5,approach=0),
    'arrival_envelope':dict(leading=5,trailing=13,approach=0),
    'balanced_envelope':dict(leading=5,trailing=5,approach=0),
    'approach13':dict(leading=None,trailing=None,approach=13),
    'approach25':dict(leading=None,trailing=None,approach=25),
    'release_approach25':dict(leading=13,trailing=5,approach=25),
    'balanced_approach25':dict(leading=5,trailing=5,approach=25),
}


def spans(values):
    edges=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1,len(values)]
    return [(int(a),int(b),str(values[a])) for a,b in zip(edges[:-1],edges[1:])]


def refine(reference,fast,raw,broadcast,old_mapped,blocks,recoveries,leading=None,trailing=None,approach=0):
    """Refine existing intervals; no truth, frame-specific rule or protocol input.

    Arrival/release contact envelopes tighten short islands by <=25frames per
    edge. Proven accepted camera recoveries can retrospectively start at three
    consecutive bilateral3m approach observations within13or25preceding frames.
    Both proposals read unchanged inputs. Their union must preserve every run
    label and minimum length; otherwise that decode block retains its reference.
    """
    reference=np.asarray(reference);fast=np.asarray(fast);old_mapped=np.asarray(old_mapped,bool)
    n=len(reference)
    if (reference.ndim!=1 or fast.shape!=(n,2) or old_mapped.shape!=(n,)
        or len(raw)!=n or len(broadcast)!=n or not np.isin(reference,['H','A','D','X']).all()
        or (leading is None)!=(trailing is None) or leading not in (None,5,13)
        or trailing not in (None,5,13) or approach not in (0,13,25)):
        raise ValueError('Invalid aligned boundary-refinement inputs')
    td=raw[['h__distance_1','a__distance_1']].to_numpy()
    bd=broadcast[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    observed=(broadcast.bc_matched.to_numpy()==1)&(broadcast.bc_same_ball_seen.to_numpy()==1)&np.isfinite(bd).all(1)
    contacts=(bd<2)&(bd[:,::-1]-bd>.5)&observed[:,None]
    nearby=(td<3)&(td[:,::-1]-td>.5)&(bd<3)&(bd[:,::-1]-bd>.5)&observed[:,None]&~old_mapped[:,None]
    envelope=reference.copy();onsets=reference.copy();covered=np.zeros(n,bool);events=[]
    checked=[]
    for left,right in blocks:
        if not 0<=left<right<=n or covered[left:right].any():raise ValueError('Invalid or overlapping decode blocks')
        covered[left:right]=True;checked.append((left,right))
        if not np.isfinite(fast[left:right]).all() or (fast[left:right]<0).any() or not np.allclose(fast[left:right].sum(1),1):
            raise ValueError('Invalid fast probability')
        original=spans(reference[left:right])
        if leading is not None:
            for i,(a,b,team) in enumerate(original):
                if team not in ('H','A') or not 8<=b-a<=250 or i==0 or i==len(original)-1:continue
                other='A' if team=='H' else 'H'
                if original[i-1][2]!=other or original[i+1][2]!=other:continue
                start,end=left+a,left+b;k=int(team=='A')
                if not observed[start:end].all():continue
                contact=np.where(contacts[start:end,k])[0]
                if len(contact)<3:continue
                first,last=start+int(contact[0]),start+int(contact[-1])+1
                begin=max(start,min(first-leading,start+25));finish=min(end,max(last+trailing,end-25))
                if finish-begin<8:continue
                envelope[start:begin]=other;envelope[finish:end]=other
                if begin!=start or finish!=end:events.append(dict(kind='envelope',start=start,end_exclusive=end,
                    new_start=begin,new_end_exclusive=finish,team=team))
        if approach:
            for event in recoveries:
                start,end,team=event['start'],event['end_exclusive'],event['team']
                if not left<start<end<right:continue
                k=int(team=='A');other='H' if team=='A' else 'A'
                if (not np.all(reference[start:end]==team) or reference[start-1]!=other
                    or reference[end]!=other or reference[start-1]==team):continue
                previous=start
                while previous>left and reference[previous-1]==other:previous-=1
                low=max(previous+8,start-approach)
                if low>=start or old_mapped[low:end].any() or not observed[low:end].all():continue
                evidence=nearby[low:start,k]&(fast[low:start,k]>.5)
                candidates=np.where(np.convolve(evidence.astype(int),np.ones(3,dtype=int),mode='valid')==3)[0] if len(evidence)>=3 else []
                if not len(candidates):continue
                begin=low+int(candidates[0]);onsets[begin:start]=team
                events.append(dict(kind='approach',start=start,end_exclusive=end,new_start=begin,
                                   new_end_exclusive=end,team=team))
    result=reference.copy();result[envelope!=reference]=envelope[envelope!=reference];result[onsets!=reference]=onsets[onsets!=reference]
    for left,right in checked:
        old,new=spans(reference[left:right]),spans(result[left:right])
        if ([s[2] for s in old]!=[s[2] for s in new]
            or any(b-a<min(y-x,8) for (a,b,_),(x,y,_) in zip(new,old))):
            result[left:right]=reference[left:right]
            events.append(dict(kind='topology_refusal',start=left,end_exclusive=right))
    for state in ('D','X'):np.testing.assert_array_equal(result==state,reference==state)
    return result,events
