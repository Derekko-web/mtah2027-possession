import unittest
import numpy as np
from ball_context_data import ball_views,assemble_ball


class BallDataTests(unittest.TestCase):
    def test_views_are_deterministic_target_free_and_nan_safe(self):
        n=100;xy=np.zeros((4,n,4,2),np.float32);seen=np.ones((4,n,4),bool);ball=np.zeros((n,2),np.float32);ball[7]=np.nan
        a,m=ball_views(ball,[100,60],xy,seen,39);b,k=ball_views(ball,[100,60],xy,seen,39)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(m,k);self.assertFalse(m[:,7].any());self.assertTrue(np.isfinite(a).all())
        seen[:]=False;_,m=ball_views(ball,[100,60],xy,seen,39);self.assertFalse(m.any())

    def test_control_drops_ball_and_reflections_follow_players(self):
        x=np.zeros((4,100,4,2),np.float32);mask=np.ones(x.shape[:-1],bool);teams=np.tile([0,0,1,1],(100,1));half=np.zeros(100,int);ball=np.full((4,100,2),.2,np.float32);seen=np.ones(ball.shape[:-1],bool)
        values=assemble_ball(x,mask,teams,half,ball,seen,[50],[0],[True],[False]);np.testing.assert_allclose(values[3][0,0],[-.2,.2])
        control=assemble_ball(x,mask,teams,half,ball,seen,[50],[0],control=True);self.assertFalse(control[4].any());self.assertTrue((control[3]==0).all())
        with self.assertRaises(ValueError):assemble_ball(x,mask,teams,half,ball,seen,[0],[0])


if __name__=='__main__':unittest.main()
