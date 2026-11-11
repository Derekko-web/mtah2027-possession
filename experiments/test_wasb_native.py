import unittest
import numpy as np
from wasb_native import contact_arrays
from games import GAME,GAMES


class OpticalContactTests(unittest.TestCase):
    def test_projection_and_team_swap_preserve_physical_contact_meaning(self):
        pitch=np.asarray(GAMES[GAME]['pitch']);positions=np.array([[[20.,20.],[80.,40.]]]);teams=np.array([0,1]);matrices=np.array([np.diag([1920.,1080.,1.])]);point=np.array([[20.,20.]])/pitch*[1920.,1080.]
        p,s,q=contact_arrays(point,positions,teams,matrices,np.array([True]));self.assertTrue(s[0]);self.assertTrue(q[0]);self.assertGreater(p[0],.99)
        other,ss,qq=contact_arrays(point,positions,1-teams,matrices,np.array([True]));np.testing.assert_allclose(p,1-other);np.testing.assert_array_equal(s,ss);np.testing.assert_array_equal(q,qq)

    def test_missing_outside_or_excluded_points_never_propose_a_team(self):
        positions=np.zeros((3,2,2));teams=np.array([0,1]);matrices=np.repeat(np.diag([1920.,1080.,1.])[None],3,axis=0);points=np.array([[np.nan,np.nan],[-1000.,0.],[0.,0.]])
        p,s,q=contact_arrays(points,positions,teams,matrices,np.array([True,True,False]));self.assertTrue(np.isnan(p).all());self.assertFalse(s.any());self.assertFalse(q.any())


if __name__=='__main__':unittest.main()
