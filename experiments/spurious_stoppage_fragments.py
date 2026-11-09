"""Remove unsupported short ownership flips at existing stoppage boundaries."""
import numpy as np
from possession_boundary_precision import spans

VARIANTS={
    'invisible_continuation':dict(invisible=True,stoppage=False),
    'observed_stoppage':dict(invisible=False,stoppage=True),
    'combined_fragments':dict(invisible=True,stoppage=True),
}


def refine(reference,heads,raw,broadcast,blocks,invisible=True,stoppage=True):
    """Use unchanged run evidence, never labels or target-specific intervals.

    Only original short H/A runs (1..125frames) touching D are eligible. An
    entirely unobserved flip between opposite live control and D inherits the
    live neighbor. Otherwise a broadcast-unobserved,mostly stationary visible
    ball plus mean weak-D-head probability>.5 may extend the existing D span.
    Existing D/X are preserved. No rule chains through its own output.
    """
    reference=np.asarray(reference);heads=np.asarray(heads)
    n=len(reference)
    if (reference.ndim!=1 or heads.shape!=(n,2) or len(raw)!=n or len(broadcast)!=n
        or not np.isin(reference,['H','A','D','X']).all()
        or not isinstance(invisible,bool) or not isinstance(stoppage,bool)):
        raise ValueError('Invalid stoppage-fragment inputs')
    distance=broadcast[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    bc_seen=(broadcast.bc_matched.to_numpy()==1)&(broadcast.bc_same_ball_seen.to_numpy()==1)&np.isfinite(distance).all(1)
    tc_seen=raw.ball_seen.to_numpy()>0
    moving=raw.ball_displacement_1s.to_numpy()>4.5
    result=reference.copy();events=[];covered=np.zeros(n,bool)
    for left,right in blocks:
        if not 0<=left<right<=n or covered[left:right].any():raise ValueError('Invalid or overlapping blocks')
        covered[left:right]=True
        if not np.isfinite(heads[left:right]).all() or ((heads[left:right]<0)|(heads[left:right]>1)).any():
            raise ValueError('Invalid dead-head probabilities')
        runs=spans(reference[left:right])
        for i,(a,b,team) in enumerate(runs):
            if team not in ('H','A') or not 1<=b-a<=125 or i==0 or i==len(runs)-1:continue
            before,after=runs[i-1][2],runs[i+1][2]
            if 'D' not in (before,after):continue
            start,end=left+a,left+b
            if bc_seen[start:end].any():continue
            other='A' if team=='H' else 'H';replacement=None;kind=None
            if invisible and not tc_seen[start:end].any() and other in (before,after):
                replacement=other;kind='invisible_continuation'
            elif (stoppage and tc_seen[start:end].mean()>=.5
                  and moving[start:end].mean()<=.5 and heads[start:end].mean(0).max()>.5):
                replacement='D';kind='observed_stoppage'
            if replacement is not None:
                result[start:end]=replacement
                events.append(dict(start=start,end_exclusive=end,team=team,replacement=replacement,kind=kind,
                    mean_dead_heads=heads[start:end].mean(0).tolist(),visible_fraction=float(tc_seen[start:end].mean()),
                    moving_fraction=float(moving[start:end].mean())))
    np.testing.assert_array_equal(result=='X',reference=='X')
    np.testing.assert_array_equal(result[reference=='D'],reference[reference=='D'])
    return result,events
