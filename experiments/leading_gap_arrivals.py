"""Restrict sparse-source arrival proposals to short leading camera gaps."""
import numpy as np
from source_aware_arrivals import refine as propose
from possession_boundary_precision import spans

VARIANTS={
    'leading_owner13':dict(camera=False,posterior_radius=13),
    'leading_camera13':dict(camera=True,posterior_radius=13),
    'leading_camera25':dict(camera=True,posterior_radius=25),
}


def refine(reference,owner,raw,broadcast,blocks,camera=True,posterior_radius=13):
    """Apply camera disagreement, or a physically supported leading-gap repair.

    A sparse-source proposal needs missing observations at its original onset,
    first camera visibility within the same13/25-frame optimization radius,
    and >=3observed contacts for its current owner later in that reference run.
    Tail gaps and completely unsupported intervals cannot trigger the repair.
    """
    reference=np.asarray(reference)
    _,proposals=propose(reference,owner,raw,broadcast,blocks,camera=camera,posterior_radius=posterior_radius)
    distance=broadcast[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    observed=(broadcast.bc_matched.to_numpy()==1)&(broadcast.bc_same_ball_seen.to_numpy()==1)&np.isfinite(distance).all(1)
    contacts=(distance<2)&(distance[:,::-1]-distance>.5)&observed[:,None]
    result=reference.copy();accepted=[]
    for event in proposals:
        start,end,team=event['start'],event['end_exclusive'],event['team'];k=int(team=='A')
        if event['kind']=='incomplete_source_posterior':
            valid=np.flatnonzero(observed[start:end])
            if not len(valid):continue
            gap=int(valid[0]);count=int(contacts[start:end,k].sum())
            if not 0<gap<=posterior_radius or count<3:continue
            event=dict(event,leading_gap_frames=gap,observed_owner_contacts=count)
        result[start:event['new_start']]='H' if team=='A' else 'A';accepted.append(event)
    for left,right in blocks:
        before,after=spans(reference[left:right]),spans(result[left:right])
        if [x[2] for x in before]!=[x[2] for x in after]:raise AssertionError('Filtered arrival changed topology')
    for state in ('D','X'):np.testing.assert_array_equal(result==state,reference==state)
    return result,accepted
