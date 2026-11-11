"""Independent replay across shifted approaches, contact gaps and model votes."""
import unittest
import numpy as np
import pandas as pd
from audit_guarded_boundaries import independent_refine
from possession_boundary_precision import refine,VARIANTS
from probability_guarded_boundaries import guarded_refine


class IndependentBoundaryTests(unittest.TestCase):
    def test_shifted_contact_and_approach_replays(self):
        rng=np.random.default_rng(9341)
        for case in range(100):
            n=450;start=int(rng.integers(70,140));length=int(rng.integers(35,140));end=start+length
            team=case%2;other=1-team;label='A' if team else 'H';opposite='H' if team else 'A'
            ref=np.array([opposite]*start+[label]*length+[opposite]*(n-end))
            probability=rng.uniform(.1,.9,size=n);owner=np.column_stack((probability,1-probability))
            fast=np.full((n,2),.5);d=np.full((n,2),4.);d[:,other]=1.
            bd=d.copy();first=start+int(rng.integers(5,20));last=end-int(rng.integers(5,20))
            bd[first:last,team]=1.;bd[first:last,other]=4.
            approach=max(10,start-int(rng.integers(3,28)));d[approach:approach+5,team]=1.;d[approach:approach+5,other]=4.
            bd[approach:approach+5,team]=1.;bd[approach:approach+5,other]=4.;fast[approach:approach+5,team]=.8;fast[approach:approach+5,other]=.2
            raw=pd.DataFrame(d,columns=['h__distance_1','a__distance_1'])
            bc=pd.DataFrame(bd,columns=['h__bc_same_distance_1','a__bc_same_distance_1']);bc['bc_matched']=1.;bc['bc_same_ball_seen']=1.
            old=np.zeros(n,bool)
            if case%9==0:bc.loc[first+2,'bc_same_ball_seen']=0.
            if case%13==0:old[start-3]=True
            events=[dict(start=start,end_exclusive=end,team=label)];blocks=[(0,n)]
            for config in VARIANTS.values():
                actual,_=refine(ref,fast,raw,bc,old,blocks,events,**config)
                expected=independent_refine(ref,fast,owner,raw,bc,old,blocks,events,**config)
                np.testing.assert_array_equal(actual,expected)
            for back in (0,25):
                actual,_=guarded_refine(ref,fast,owner,raw,bc,old,blocks,events,approach=back)
                expected=independent_refine(ref,fast,owner,raw,bc,old,blocks,events,13,5,back,True)
                np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':unittest.main()
