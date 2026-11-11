"""Remove only short D islands supported as continuing same-team live play."""
import numpy as np


def bridge_dead(reference,fast,ball_seen,moving_contact,restart,blocks):
    reference=np.asarray(reference);fast=np.asarray(fast,float);ball_seen=np.asarray(ball_seen,float)
    moving_contact=np.asarray(moving_contact,bool);restart=np.asarray(restart,bool)
    n=len(reference)
    if fast.shape!=(n,2) or moving_contact.shape!=(n,2) or ball_seen.shape!=(n,) or restart.shape!=(n,):
        raise ValueError('Unaligned gap evidence')
    output=reference.copy();trace=[]
    for left,right in blocks:
        dead=reference[left:right]=='D'
        edges=np.flatnonzero(np.diff(np.r_[False,dead,False]))
        for a,b in zip(edges[::2],edges[1::2]):
            start,end=left+a,left+b;reason='accepted';team=None
            if end-start>63:reason='duration'
            elif start-13<left or end+13>right:reason='boundary'
            else:
                owner=reference[start-1]
                if owner not in ('H','A') or not np.all(reference[start-13:start]==owner) or not np.all(reference[end:end+13]==owner):
                    reason='flanks'
                else:
                    team=0 if owner=='H' else 1
                    if restart[start:end].any():reason='restart'
                    elif np.mean(ball_seen[start:end])<.5:reason='visibility'
                    elif moving_contact[start:end,team].sum()<3:reason='moving_contact'
                    elif moving_contact[start:end,1-team].any():reason='opposing_contact'
                    elif not np.isfinite(fast[start:end]).all() or fast[start:end,team].mean()<.65:reason='fast_agreement'
            if reason=='accepted':output[start:end]='H' if team==0 else 'A'
            trace.append(dict(start=start,end_exclusive=end,frames=end-start,accepted=reason=='accepted',reason=reason))
    if not np.array_equal(output[reference!='D'],reference[reference!='D']):raise AssertionError('Bridge changed an existing live/unknown label')
    if ((output=='D')&(reference!='D')).any():raise AssertionError('Bridge added a stoppage')
    return output,trace
