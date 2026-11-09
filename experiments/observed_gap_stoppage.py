"""Anchor an unseen flip's dead boundary to the actual visibility gap."""
import numpy as np
from spurious_stoppage_fragments import refine as propose

VARIANTS={'source_gap13':dict(allowance=13),'source_gap25':dict(allowance=25)}


def refine(reference,heads,raw,broadcast,blocks,allowance=25):
    """Use visibility-gap timing instead of the already delayed D prediction.

    Original unseen owner flips are proposed exactly as before. At their
    D-facing side,retain13/25frames after ball visibility disappears before
    extending D (mirror at restart). Search at most the same allowance outside
    the flip,and stay in the original decode block. All observed suppressions
    retain their unchanged weak-head/stationarity conditions.
    """
    if allowance not in (13,25):raise ValueError('Invalid visibility allowance')
    reference=np.asarray(reference)
    result,events=propose(reference,heads,raw,broadcast,blocks,invisible=True,stoppage=True)
    distance=broadcast[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    visible=(raw.ball_seen.to_numpy()>0)|((broadcast.bc_matched.to_numpy()==1)&
        (broadcast.bc_same_ball_seen.to_numpy()==1)&np.isfinite(distance).all(1))
    accepted=[]
    for event in events:
        event=dict(event)
        if event['kind']=='invisible_continuation':
            start,end=event['start'],event['end_exclusive']
            left,right=next((a,b) for a,b in blocks if a<=start<end<=b)
            if start>left and reference[start-1]=='D':
                cursor=end
                while cursor<min(right,end+allowance) and not visible[cursor]:cursor+=1
                a,b=start,max(start,min(end,cursor-allowance))
            elif end<right and reference[end]=='D':
                cursor=start
                while cursor>max(left,start-allowance) and not visible[cursor-1]:cursor-=1
                a,b=min(end,max(start,cursor+allowance)),end
            else:raise AssertionError('Visibility closure has no original D neighbor')
            result[a:b]='D'
            event.update(kind='visibility_gap_closure',dead_start=a,dead_end_exclusive=b,
                visibility_anchor=cursor,allowance=allowance)
        accepted.append(event)
    np.testing.assert_array_equal(result=='X',reference=='X')
    np.testing.assert_array_equal(result[reference=='D'],reference[reference=='D'])
    return result,accepted
