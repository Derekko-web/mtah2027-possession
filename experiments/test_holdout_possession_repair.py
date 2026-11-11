"""Physical and continuity repairs must keep observed live evidence intact."""
import unittest
import numpy as np
import pandas as pd
from holdout_possession_repair import refine


class HoldoutRepairTests(unittest.TestCase):
    def fixture(self,pattern):
        ref=np.array(list(pattern));n=len(ref)
        heads=np.full((n,2),.1);owner=np.tile([.8,.2],(n,1));fast=owner.copy()
        raw=pd.DataFrame(dict(ball_seen=np.ones(n),ball_displacement_1s=np.full(n,6.),
            ball_goal_line_distance=np.full(n,10.),ball_touchline_distance=np.full(n,10.),
            h__distance_1=np.ones(n),a__distance_1=np.full(n,4.)))
        bc=pd.DataFrame(dict(bc_matched=np.zeros(n),bc_same_ball_seen=np.zeros(n),
            h__bc_same_distance_1=np.full(n,np.nan),a__bc_same_distance_1=np.full(n,np.nan)))
        return [ref,heads,owner,fast,raw,bc,[(0,n)]]

    def test_three_outside_observations_close_stoppage(self):
        data=self.fixture('A'*30+'H'*40+'D'*50);data[4].loc[40:,'ball_goal_line_distance']=-1
        out,_=refine(*data,stages=('pitch',))
        self.assertTrue(np.all(out[:42]=='A'));self.assertTrue(np.all(out[42:]=='D'))

    def test_false_dead_gap_bridges_continuous_control(self):
        data=self.fixture('H'*30+'D'*40+'H'*50)
        out,_=refine(*data,stages=('bridge',));self.assertTrue(np.all(out=='H'))
        data[4]['ball_displacement_1s']=1
        out,_=refine(*data,stages=('bridge',));np.testing.assert_array_equal(out,data[0])

    def test_visible_broadcast_protects_outside_candidate(self):
        data=self.fixture('A'*30+'H'*40+'D'*50);data[4]['ball_goal_line_distance']=-1
        data[5].loc[40,['bc_matched','bc_same_ball_seen','h__bc_same_distance_1','a__bc_same_distance_1']]=[1,1,1,4]
        out,_=refine(*data,stages=('pitch',));np.testing.assert_array_equal(out,data[0])

    def test_edge_stoppage_requires_model_and_motion_support(self):
        data=self.fixture('H'*40+'A'*100);data[1][:40]=.8;data[4].loc[:39,'ball_displacement_1s']=1
        out,_=refine(*data,stages=('edge',));self.assertTrue(np.all(out[:40]=='D'))
        data[1][:]=.5;out,_=refine(*data,stages=('edge',));np.testing.assert_array_equal(out,data[0])

    def test_short_dead_release_uses_future_owner_contacts(self):
        data=self.fixture('A'*30+'H'*50+'D'*80)
        data[5].loc[55:85,['bc_matched','bc_same_ball_seen','h__bc_same_distance_1','a__bc_same_distance_1']]=[1,1,1,4]
        out,_=refine(*data,stages=('release',));self.assertTrue(np.all(out[80:99]=='H'))
        self.assertTrue(np.all(out[99:]=='D'))

    def test_invalid_blocks_and_probabilities_rejected(self):
        for condition in ('block','probability'):
            data=self.fixture('H'*30+'D'*40+'H'*50)
            if condition=='block':data[-1]=[(0,80),(70,120)]
            else:data[2][3]=np.nan
            with self.assertRaises(ValueError):refine(*data)


if __name__=='__main__':unittest.main()
