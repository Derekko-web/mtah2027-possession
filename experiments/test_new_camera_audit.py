"""Differential checks of independent control reconstruction under jitter/gaps."""
import unittest
import numpy as np
import pandas as pd
from new_camera_control import recover
from audit_new_camera_control import independently_recover


class IndependentReplayTests(unittest.TestCase):
    def test_time_shifted_camera_jitter_and_missingness(self):
        rng=np.random.default_rng(2187)
        for case in range(100):
            n=600;start=int(rng.integers(50,200));duration=int(rng.integers(20,120))
            team=case%2;other=1-team;ref=np.full(n,'H' if team==0 else 'A')
            probability=np.tile([.9,.1] if team==0 else [.1,.9],(n,1))
            probability[start:start+6]=[.4,.6] if team==0 else [.6,.4]
            distance=np.full((n,2),4.);distance[:,team]=1.
            distance[start:start+6,team]=4.;distance[start:start+6,other]=1.
            bd=distance.copy();jitter=int(rng.integers(-2,3));bd[start:start+6]=[1.,4.] if team==0 else [4.,1.]
            bd[start+jitter:start+jitter+6,team]=4.;bd[start+jitter:start+jitter+6,other]=1.
            kernel=np.full((n,2),.1);kernel[:,team]=.9;kernel[start-5:start+duration,team]=.1
            raw=pd.DataFrame(distance,columns=['h__distance_1','a__distance_1'])
            motion=pd.DataFrame(kernel,columns=['h__tc_kernel','a__tc_kernel'])
            bc=pd.DataFrame(bd,columns=['h__bc_same_distance_1','a__bc_same_distance_1']);bc['bc_matched']=1.;bc['bc_same_ball_seen']=1.
            old=np.zeros(n,bool)
            if case%7==0:old[start+10]=True
            if case%11==0:bc.loc[start+12,'bc_same_ball_seen']=0.
            for confirmation in (13,25):
                actual,_=recover(ref,probability,raw,motion,bc,old,[(0,n)],confirmation)
                expected=independently_recover(ref,probability,raw,motion,bc,old,[(0,n)],confirmation)
                np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':unittest.main()
