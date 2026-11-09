"""Constrain short reference islands to observed broadcast contact envelopes."""
import numpy as np


def refine(reference, broadcast, blocks, padding=13):
    """Trim at most25frames per edge, never add islands or alter D/X.

    Only fully observed, bracketed8..250-frame live islands qualify. At least
    three exclusive same-camera contacts are required. Geometry thresholds and
    temporal scales reuse the existing contact/control inference vocabulary.
    """
    reference=np.asarray(reference)
    if (reference.ndim!=1 or len(broadcast)!=len(reference)
            or not np.isin(reference,['H','A','D','X']).all() or padding not in (5,13)):
        raise ValueError('Invalid aligned broadcast refinement inputs')
    n=len(reference);covered=np.zeros(n,bool);result=reference.copy();events=[]
    observed=(broadcast.bc_matched.to_numpy()==1)&(broadcast.bc_same_ball_seen.to_numpy()==1)
    distances=broadcast[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    for left,right in blocks:
        if not 0<=left<right<=n or covered[left:right].any():
            raise ValueError('Invalid or overlapping decode blocks')
        covered[left:right]=True
        local=reference[left:right]
        edges=np.r_[0,np.flatnonzero(local[1:]!=local[:-1])+1,len(local)]
        for a,b in zip(edges[:-1],edges[1:]):
            team=local[a]
            if team not in ('H','A') or not 8<=b-a<=250 or a==0 or b==len(local):continue
            other='A' if team=='H' else 'H'
            if local[a-1]!=other or local[b]!=other:continue
            start,end=left+int(a),left+int(b);k=0 if team=='H' else 1
            if not observed[start:end].all() or not np.isfinite(distances[start:end]).all():continue
            contact=(distances[start:end,k]<2)&(distances[start:end,1-k]-distances[start:end,k]>.5)
            where=np.flatnonzero(contact)
            if len(where)<3:continue
            first,last=start+int(where[0]),start+int(where[-1])+1
            new_start=max(start,min(first-padding,start+25))
            new_end=min(end,max(last+padding,end-25))
            if new_end-new_start<8:continue
            result[start:new_start]=other;result[new_end:end]=other
            if new_start!=start or new_end!=end:
                events.append(dict(start=start,end_exclusive=end,team=str(team),
                    new_start=new_start,new_end_exclusive=new_end,contacts=len(where)))
    for state in ('D','X'):np.testing.assert_array_equal(result==state,reference==state)
    return result,events
