"""Missing-source, team symmetry and bounded-prefix checks for color geometry."""

import unittest

import numpy as np
import pandas as pd

from jersey_ownership import decode_jersey_ownership,expand_team_probability,jersey_geometry


class JerseyOwnershipTests(unittest.TestCase):
    def test_interpolation_requires_two_valid_adjacent_endpoints(self):
        p = np.array([[[.9,.1]],[[.8,.2]],[[np.nan,np.nan]],[[.1,.9]]])
        result = expand_team_probability([5,10,15,25],p,30)
        self.assertTrue(np.isnan(result[:5]).all())
        np.testing.assert_allclose(result[7,0],[.86,.14])
        self.assertTrue(np.isnan(result[11:15]).all())
        self.assertTrue(np.isnan(result[16:25]).all())
        self.assertTrue(np.isnan(result[26:]).all())

    def test_geometry_team_swap_and_real_padding_scope(self):
        n = 220
        xy = np.tile(np.array([[10.,10.],[20.,10.],[30.,10.],[40.,10.]])[None],(n,1,1))
        speed = np.ones((n,4));ball = np.tile([11.,10.],(n,1));teams = np.array([0,0,1,1])
        frames = np.arange(0,n,5)
        p = np.tile(np.array([[.99,.01],[.99,.01],[.01,.99],[.01,.99]])[None],(len(frames),1,1))
        a = jersey_geometry(xy,speed,ball,teams,frames,p)
        b = jersey_geometry(xy,speed,ball,1-teams,frames,p[:,:,::-1])
        for name in a:
            other = "a__"+name[3:] if name.startswith("h__") else "h__"+name[3:] if name.startswith("a__") else name
            np.testing.assert_allclose(a[name],b[other],equal_nan=True)
        shorter = jersey_geometry(xy[:200],speed[:200],ball[:200],teams,frames[frames < 200],p[frames < 200])
        pd.testing.assert_frame_equal(a.iloc[:65],shorter.iloc[:65])

    def test_decoders_keep_dead_and_missing_source_reference(self):
        reference = np.r_[np.zeros(25,int),np.full(10,2),np.ones(25,int)]
        features = pd.DataFrame(dict(jc_ball_seen=np.ones(60),h__jc_distance_1=np.ones(60),a__jc_distance_1=np.full(60,5.),
            h__jc_control_share=np.full(60,.9),a__jc_control_share=np.full(60,.1)))
        features.loc[40:59,"jc_ball_seen"] = 0
        p = np.tile([.99,.01],(60,1))
        for mode in ("replace_live","short_proposals","source_closure"):
            result = decode_jersey_ownership(reference,p,features,mode)
            np.testing.assert_array_equal(result == 2,reference == 2)
            np.testing.assert_array_equal(result[40:],reference[40:])


if __name__ == "__main__": unittest.main()
