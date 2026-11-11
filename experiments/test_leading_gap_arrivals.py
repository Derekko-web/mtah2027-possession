"""Posterior movement requires a leading, bounded and observed-control gap."""
import unittest
import numpy as np
import pandas as pd
from leading_gap_arrivals import refine


class LeadingGapTests(unittest.TestCase):
    def fixture(self):
        n=220;ref=np.array(list('A'*50+'H'*70+'D'*100));owner=np.tile([.9,.1],(n,1));owner[50:60]=[.2,.8]
        raw=pd.DataFrame({'h__distance_1':np.full(n,1.),'a__distance_1':np.full(n,4.)})
        bc=pd.DataFrame({'h__bc_same_distance_1':np.full(n,1.),'a__bc_same_distance_1':np.full(n,4.),'bc_matched':np.ones(n),'bc_same_ball_seen':np.ones(n)})
        bc.loc[50:58,'bc_matched']=0
        return ref,owner,raw,bc,[(0,n)]

    def test_short_leading_gap_and_owner_contacts_allow_change(self):
        data=self.fixture();actual,events=refine(*data,camera=False)
        self.assertTrue(np.all(actual[50:60]=='A'));self.assertEqual(events[0]['leading_gap_frames'],9)

    def test_tail_only_gap_cannot_trigger_posterior_repair(self):
        data=list(self.fixture());data[3]['bc_matched']=1.;data[3].loc[100:105,'bc_matched']=0
        actual,events=refine(*data,camera=False);np.testing.assert_array_equal(actual,data[0]);self.assertEqual(events,[])

    def test_long_gap_or_missing_control_refuses_movement(self):
        for condition in ('long','contact'):
            data=list(self.fixture())
            if condition=='long':data[3].loc[50:75,'bc_matched']=0
            else:data[3]['h__bc_same_distance_1']=4.
            actual,_=refine(*data,camera=False);np.testing.assert_array_equal(actual,data[0])

    def test_later_internal_gap_does_not_remove_valid_leading_evidence(self):
        data=list(self.fixture());data[3].loc[80:90,'bc_matched']=0
        actual,_=refine(*data,camera=False);self.assertTrue(np.all(actual[50:60]=='A'))

    def test_swap_and_immutability(self):
        ref,p,raw,bc,blocks=self.fixture();copy=ref.copy();actual,_=refine(ref,p,raw,bc,blocks,camera=False)
        swap=lambda x:np.where(x=='H','A',np.where(x=='A','H',x))
        reverse=lambda d:d.rename(columns={c:('a__'+c[3:] if c.startswith('h__') else 'h__'+c[3:] if c.startswith('a__') else c) for c in d})
        other,_=refine(swap(ref),p[:,::-1],reverse(raw),reverse(bc),blocks,camera=False)
        np.testing.assert_array_equal(other,swap(actual));np.testing.assert_array_equal(ref,copy)


if __name__=='__main__':unittest.main()
