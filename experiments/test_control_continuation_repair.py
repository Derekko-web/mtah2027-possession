"""Continuation must require contact and stable observed control, not a timer."""
import unittest
import numpy as np
import pandas as pd
from control_continuation_repair import repair


class ContinuationTests(unittest.TestCase):
    def fixture(self):
        n=400;ref=np.array(list('H'*n));p=np.tile([.9,.1],(n,1));p[100:108]=[.2,.8]
        raw=pd.DataFrame({'h__distance_1':np.full(n,1.),'a__distance_1':np.full(n,4.)})
        raw.loc[100:105,['h__distance_1','a__distance_1']]=[4.,1.]
        m=pd.DataFrame({'h__tc_kernel':np.full(n,.9),'a__tc_kernel':np.full(n,.1)})
        m.loc[95:150,'h__tc_kernel']=.1
        return ref,p,raw,m,[(0,n)]

    def test_brief_foreign_contact_carries_until_stable_control(self):
        ref,p,raw,m,blocks=self.fixture();actual,events=repair(ref,p,raw,m,blocks)
        self.assertTrue(np.all(actual[100:164]=='A'))
        self.assertEqual(events[0]['end_exclusive'],164)

    def test_no_stable_control_within_limit_rejects_event(self):
        ref,p,raw,m,blocks=self.fixture();m['h__tc_kernel']=.1
        actual,events=repair(ref,p,raw,m,blocks);np.testing.assert_array_equal(actual,ref)
        self.assertEqual(events,[])

    def test_existing_short_spells_cannot_be_overwritten(self):
        ref,p,raw,m,_=self.fixture();ref[180:]='A'
        actual,_=repair(ref,p,raw,m,[(0,400)]);np.testing.assert_array_equal(actual,ref)

    def test_three_contacts_and_fast_support_are_both_required(self):
        for missing in ('contacts','model'):
            ref,p,raw,m,blocks=self.fixture()
            if missing=='contacts':raw['a__distance_1']=4.
            else:p[:]=[.9,.1]
            actual,_=repair(ref,p,raw,m,blocks);np.testing.assert_array_equal(actual,ref)

    def test_team_swap_and_immutability(self):
        ref,p,raw,m,blocks=self.fixture();before=ref.copy()
        actual,_=repair(ref,p,raw,m,blocks)
        swap=lambda x:np.where(x=='H','A',np.where(x=='A','H',x))
        b,_=repair(swap(ref),p[:,::-1],raw.rename(columns={'h__distance_1':'a__distance_1','a__distance_1':'h__distance_1'}),
                   m.rename(columns={'h__tc_kernel':'a__tc_kernel','a__tc_kernel':'h__tc_kernel'}),blocks)
        np.testing.assert_array_equal(b,swap(actual));np.testing.assert_array_equal(ref,before)


if __name__=='__main__':unittest.main()
