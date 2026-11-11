"""A contact proposal alone cannot override an opposing ownership majority."""
import unittest
import numpy as np
import pandas as pd
from probability_guarded_boundaries import guarded_refine


class ProbabilityGuardTests(unittest.TestCase):
    def fixture(self):
        n=200;ref=np.array(list('A'*50+'H'*60+'A'*90));p=np.tile([.5,.5],(n,1))
        raw=pd.DataFrame({'h__distance_1':np.full(n,4.),'a__distance_1':np.full(n,1.)})
        bc=pd.DataFrame({'h__bc_same_distance_1':np.full(n,4.),'a__bc_same_distance_1':np.full(n,1.),'bc_matched':np.ones(n),'bc_same_ball_seen':np.ones(n)})
        bc.loc[63:85,['h__bc_same_distance_1','a__bc_same_distance_1']]=[1.,4.]
        owner=np.tile([.9,.1],(n,1));return ref,p,owner,raw,bc,np.ones(n,bool),[(0,n)],[]

    def test_model_disagreement_preserves_the_boundary(self):
        data=self.fixture();actual,events=guarded_refine(*data)
        np.testing.assert_array_equal(actual,data[0]);self.assertEqual(events,[])

    def test_model_corroboration_accepts_contact_release(self):
        data=list(self.fixture());data[2][91:110]=[.1,.9]
        actual,events=guarded_refine(*data);self.assertTrue(np.all(actual[91:110]=='A'));self.assertEqual(len(events),1)

    def test_exact_tie_does_not_move_boundary(self):
        data=list(self.fixture());data[2][:]=.5
        actual,_=guarded_refine(*data);np.testing.assert_array_equal(actual,data[0])

    def test_team_symmetry_and_immutability(self):
        ref,p,o,raw,bc,old,blocks,events=self.fixture();o[91:110]=[.1,.9];before=[ref.copy(),o.copy()]
        actual,_=guarded_refine(ref,p,o,raw,bc,old,blocks,events)
        swap=lambda x:np.where(x=='H','A',np.where(x=='A','H',x))
        reverse=lambda d:d.rename(columns={c:('a__'+c[3:] if c.startswith('h__') else 'h__'+c[3:] if c.startswith('a__') else c) for c in d})
        b,_=guarded_refine(swap(ref),p[:,::-1],o[:,::-1],reverse(raw),reverse(bc),old,blocks,events)
        np.testing.assert_array_equal(b,swap(actual));np.testing.assert_array_equal(ref,before[0]);np.testing.assert_array_equal(o,before[1])

    def test_invalid_model_probability_rejected(self):
        data=list(self.fixture());data[2][80]=np.nan
        with self.assertRaises(ValueError):guarded_refine(*data)


if __name__=='__main__':unittest.main()
