"""Require saved ownership likelihood to corroborate proposed contact trims."""
import numpy as np
from possession_boundary_precision import refine,spans

VARIANTS={'guarded_release':dict(approach=0),'guarded_release_approach25':dict(approach=25)}


def guarded_refine(reference,fast,owner,raw,broadcast,old_mapped,blocks,recoveries,approach=0):
    reference=np.asarray(reference);owner=np.asarray(owner)
    if owner.shape!=(len(reference),2) or approach not in (0,25):raise ValueError('Invalid saved ownership probabilities')
    proposed,events=refine(reference,fast,raw,broadcast,old_mapped,blocks,recoveries,leading=13,trailing=5)
    guarded=reference.copy();accepted=[]
    for left,right in blocks:
        p=owner[left:right]
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Incomplete ownership evidence')
        old,new=spans(reference[left:right]),spans(proposed[left:right])
        if [s[2] for s in old]!=[s[2] for s in new]:raise ValueError('Envelope changed topology')
        for i in range(len(old)-1):
            a,b=old[i][1],new[i][1]
            if a==b:continue
            start,end=left+min(a,b),left+max(a,b)
            label=old[i][2] if b>a else old[i+1][2]
            if label not in ('H','A') or not np.isin(reference[start:end],['H','A']).all():raise ValueError('Invalid proposed boundary')
            confidence=float(owner[start:end,int(label=='A')].mean())
            if confidence>.5:
                guarded[start:end]=label
                accepted.append(dict(kind='guarded_envelope',start=start,end_exclusive=end,team=label,mean_owner_probability=confidence))
    if approach:
        onset,onset_events=refine(reference,fast,raw,broadcast,old_mapped,blocks,recoveries,approach=approach)
        guarded[onset!=reference]=onset[onset!=reference];accepted.extend(onset_events)
    for left,right in blocks:
        a,b=spans(reference[left:right]),spans(guarded[left:right])
        if [s[2] for s in a]!=[s[2] for s in b] or any(y-x<min(v-u,8) for (x,y,_),(u,v,_) in zip(b,a)):
            guarded[left:right]=reference[left:right]
            accepted.append(dict(kind='topology_refusal',start=left,end_exclusive=right))
    for state in ('D','X'):np.testing.assert_array_equal(guarded==state,reference==state)
    return guarded,accepted
