"""Source disagreement, sparse-source change points and preservation contracts."""
import unittest
import numpy as np
import pandas as pd
from source_aware_arrivals import refine


class SourceAwareArrivalTests(unittest.TestCase):
    def fixture(self):
        n=220;ref=np.array(list('A'*50+'H'*70+'D'*100));owner=np.tile([.9,.1],(n,1))
        raw=pd.DataFrame({'h__distance_1':np.full(n,4.),'a__distance_1':np.full(n,1.)})
        bc=pd.DataFrame({'h__bc_same_distance_1':np.full(n,4.),'a__bc_same_distance_1':np.full(n,1.),'bc_matched':np.ones(n),'bc_same_ball_seen':np.ones(n)})
        bc.loc[70:110,['h__bc_same_distance_1','a__bc_same_distance_1']]=[1.,4.]
        return ref,owner,raw,bc,[(0,n)]

    def test_cross_camera_disagreement_can_override_wrong_owner_head(self):
        data=self.fixture();actual,events=refine(*data,camera=True,posterior_radius=0)
        self.assertTrue(np.all(actual[50:65]=='A'));self.assertTrue(np.all(actual[65:120]=='H'))
        self.assertEqual(events[0]['kind'],'camera_disagreement')

    def test_camera_agreement_does_not_enable_override(self):
        data=list(self.fixture());data[2].loc[50:119,['h__distance_1','a__distance_1']]=[1.,4.]
        actual,events=refine(*data,camera=True,posterior_radius=0);np.testing.assert_array_equal(actual,data[0]);self.assertEqual(events,[])

    def test_incomplete_camera_uses_probability_change_point(self):
        data=list(self.fixture());data[3].loc[50:69,'bc_matched']=0;data[1][50:60]=[.2,.8]
        actual,events=refine(*data,camera=True,posterior_radius=13)
        self.assertTrue(np.all(actual[50:60]=='A'));self.assertEqual(events[0]['new_start'],60)
        self.assertEqual(events[0]['kind'],'incomplete_source_posterior')

    def test_neutral_or_opposing_probabilities_preserve_reference(self):
        for p in ([.5,.5],[.8,.2]):
            data=list(self.fixture());data[3].loc[50:69,'bc_matched']=0;data[1][:]=p
            actual,_=refine(*data);np.testing.assert_array_equal(actual,data[0])

    def test_dead_flank_and_decode_edge_are_protected(self):
        data=list(self.fixture());data[0][:50]='D'
        actual,_=refine(*data);np.testing.assert_array_equal(actual,data[0])
        data=list(self.fixture());data[-1]=[(50,120)]
        actual,_=refine(*data);np.testing.assert_array_equal(actual,data[0])

    def test_symmetry_and_immutability(self):
        ref,p,raw,bc,blocks=self.fixture();before=[ref.copy(),p.copy(),raw.copy(),bc.copy()]
        a,_=refine(ref,p,raw,bc,blocks)
        swap=lambda v:np.where(v=='H','A',np.where(v=='A','H',v))
        reverse=lambda d:d.rename(columns={c:('a__'+c[3:] if c.startswith('h__') else 'h__'+c[3:] if c.startswith('a__') else c) for c in d})
        b,_=refine(swap(ref),p[:,::-1],reverse(raw),reverse(bc),blocks)
        np.testing.assert_array_equal(b,swap(a))
        for x,y in zip((ref,p,raw,bc),before):
            if isinstance(x,pd.DataFrame):pd.testing.assert_frame_equal(x,y)
            else:np.testing.assert_array_equal(x,y)

    def test_invalid_blocks_and_probability_rejected(self):
        data=list(self.fixture());data[-1]=[(0,150),(140,220)]
        with self.assertRaises(ValueError):refine(*data)
        data=list(self.fixture());data[1][50]=np.nan
        with self.assertRaises(ValueError):refine(*data)


if __name__=='__main__':unittest.main()
