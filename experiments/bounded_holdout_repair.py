"""Retain live prefixes before pitch exit and refine independently at block edges."""
import numpy as np
from holdout_possession_repair import refine as propose,inward,VARIANTS as ORIGINAL
from possession_boundary_precision import spans

VARIANTS={
    'bounded_physical':dict(stages=ORIGINAL['physical']['stages']),
    'edge_cross':dict(stages=('cross',)),
    'bounded_combined':dict(stages=ORIGINAL['combined']['stages']),
}


def refine(reference,heads,owner,fast,raw,broadcast,blocks,stages=VARIANTS['bounded_combined']['stages']):
    """Keep the original live prefix if outside evidence starts later in a gap.

    For a D/live/D run,whole-run suppression needs three outside observations
    already at its start. Otherwise only the suffix after outside confirmation
    becomes D. A first/last short run may receive the unchanged cross-source
    inward trim when its available live neighbor is the opposing team.
    """
    reference=np.asarray(reference);owner=np.asarray(owner);fast=np.asarray(fast)
    _,proposals=propose(reference,heads,owner,fast,raw,broadcast,blocks,stages=stages)
    outside=(raw.ball_goal_line_distance.to_numpy()<0)|(raw.ball_touchline_distance.to_numpy()<0)
    result=reference.copy();events=[]
    for event in proposals:
        event=dict(event);start,end=event['start'],event['end_exclusive']
        if event['kind']=='outside_dead_gap':
            hits=np.flatnonzero(np.convolve(outside[start:end].astype(int),np.ones(3,dtype=int),'valid')==3)
            if hits[0]>0:event.update(kind='bounded_outside_dead_gap',start=start+int(hits[0])+2)
        result[event['start']:event['end_exclusive']]=event['label'];events.append(event)
    if 'cross' in stages:
        td=raw[['h__distance_1','a__distance_1']].to_numpy()
        bd=broadcast[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
        seen=(broadcast.bc_matched.to_numpy()==1)&(broadcast.bc_same_ball_seen.to_numpy()==1)&np.isfinite(bd).all(1)
        tc=(td<2)&(td[:,::-1]-td>.5);bc=(bd<2)&(bd[:,::-1]-bd>.5)&seen[:,None]
        for left,right in blocks:
            runs=spans(reference[left:right])
            for i,(a,b,label) in enumerate(runs):
                if label not in ('H','A') or not 8<=b-a<=250 or i not in (0,len(runs)-1) or len(runs)<2:continue
                other='A' if label=='H' else 'H';neighbor=runs[1][2] if i==0 else runs[-2][2]
                if neighbor!=other:continue
                start,end=left+a,left+b;k=int(label=='A');tv=tc[start:end].sum(0);bv=bc[start:end].sum(0)
                if tv[k]>=3 and tv[k]>tv[1-k] and bv[1-k]>=3 and bv[1-k]>bv[k]:
                    begin=inward(owner,k,start,end);finish=end-inward(fast[start:end][::-1],k,0,end-start)
                    if finish-begin<8:continue
                    for kind,s,e in (('edge_cross_arrival',start,begin),('edge_cross_release',finish,end)):
                        if e>s:
                            result[s:e]=other;events.append(dict(kind=kind,start=s,end_exclusive=e,label=other))
    np.testing.assert_array_equal(result=='X',reference=='X')
    return result,events
