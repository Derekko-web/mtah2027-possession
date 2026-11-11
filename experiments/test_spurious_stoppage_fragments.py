"""Regressions for unseen flips and observed stoppage fragments."""
import unittest
import numpy as np
import pandas as pd
from spurious_stoppage_fragments import refine


class StoppageFragmentsTests(unittest.TestCase):
    def fixture(self,pattern=None):
        ref=np.array(list(pattern or 'A'*50+'H'*30+'D'*70));n=len(ref)
        raw=pd.DataFrame(dict(ball_seen=np.zeros(n),ball_displacement_1s=np.zeros(n)))
        bc=pd.DataFrame(dict(bc_matched=np.zeros(n),bc_same_ball_seen=np.zeros(n),
            h__bc_same_distance_1=np.full(n,np.nan),a__bc_same_distance_1=np.full(n,np.nan)))
        return ref,np.full((n,2),.8),raw,bc,[(0,n)]

    def test_invisible_flip_continues_opposite_control(self):
        data=self.fixture();out,event=refine(*data)
        self.assertTrue(np.all(out[:80]=='A'));self.assertEqual(event[0]['kind'],'invisible_continuation')

    def test_stationary_observed_fragment_extends_dead(self):
        data=self.fixture();data[2]['ball_seen']=1
        out,event=refine(*data)
        self.assertTrue(np.all(out[50:]=='D'));self.assertEqual(event[0]['kind'],'observed_stoppage')

    def test_real_contact_movement_or_low_head_protects_run(self):
        for condition in ('broadcast','moving','head','long','no_dead'):
            data=list(self.fixture('A'*50+'H'*126+'D'*70 if condition=='long' else
                'A'*50+'H'*30+'A'*70 if condition=='no_dead' else None));data[2]['ball_seen']=1
            if condition=='broadcast':
                data[3].loc[60,['bc_matched','bc_same_ball_seen','h__bc_same_distance_1','a__bc_same_distance_1']]=[1,1,1,4]
            if condition=='moving':data[2]['ball_displacement_1s']=6
            if condition=='head':data[1][:]=.5
            out,event=refine(*data);np.testing.assert_array_equal(out,data[0]);self.assertEqual(event,[])

    def test_dead_flanks_without_evidence_do_not_create_owner(self):
        data=self.fixture('D'*50+'H'*30+'D'*70)
        out,event=refine(*data);np.testing.assert_array_equal(out,data[0]);self.assertEqual(event,[])

    def test_team_swap_immutable_and_block_edge(self):
        data=self.fixture();original=data[0].copy();out,_=refine(*data)
        swap=lambda x:np.where(x=='A','H',np.where(x=='H','A',x))
        swapped=list(data);swapped[0]=swap(data[0]);other,_=refine(*swapped)
        np.testing.assert_array_equal(other,swap(out));np.testing.assert_array_equal(original,data[0])
        data=list(data);data[-1]=[(50,len(original))]
        result,_=refine(*data);np.testing.assert_array_equal(result,original)

    def test_bad_blocks_probabilities_and_alignment_rejected(self):
        for condition in ('blocks','heads','shape'):
            data=list(self.fixture())
            if condition=='blocks':data[-1]=[(0,100),(90,150)]
            if condition=='heads':data[1][60,0]=np.nan
            if condition=='shape':data[1]=data[1][:-1]
            with self.assertRaises(ValueError):refine(*data)


if __name__=='__main__':unittest.main()
