"""Synthetic identity, gap, and source-footprint checks without match data."""
import unittest
import numpy as np
from pathcrf_observations import bounded_fill,window


class PathInputsTests(unittest.TestCase):
    def setUp(self):
        self.teams=np.repeat([0,1],11);self.xy=np.zeros((701,22,2))
        for team in (0,1):
            for player in range(11):
                self.xy[:,team*11+player]=[5+player*3 if team==0 else 100-player*3,5+player*5]
        self.xy[:,:,0]+=np.arange(701)[:,None]*.01

    def test_complete_geometry_and_slot_permutation(self):
        value,left,reason=window(self.xy,self.teams,350)
        self.assertEqual(reason,'supported');self.assertTrue(left);self.assertEqual(value.shape,(51,26,8))
        permutation=np.arange(22)[::-1]
        other,other_left,_=window(self.xy[:,permutation],self.teams[permutation],350)
        np.testing.assert_array_equal(value,other);self.assertEqual(left,other_left)

    def test_footprint_and_track_identity_switch(self):
        original,_,_=window(self.xy,self.teams,350)
        modified=self.xy.copy();modified[:200]=999;modified[501:]=999
        np.testing.assert_array_equal(original,window(modified,self.teams,350)[0])
        swapped=self.xy.copy();swapped[350:,[0,1]]=swapped[350:,[1,0]]
        np.testing.assert_array_equal(original,window(swapped,self.teams,350)[0])

    def test_missing_positions_require_bounded_observation_support(self):
        path=np.zeros((301,11,2));path[100:125,0]=np.nan
        self.assertIsNotNone(bounded_fill(path))
        path[100:160,0]=np.nan;self.assertIsNone(bounded_fill(path))
        path=np.zeros((301,11,2));path[:11,0]=np.nan;self.assertIsNone(bounded_fill(path))
        xy=self.xy.copy();xy[:,:,0]=np.nan
        self.assertEqual(window(xy,self.teams,350)[2],'seed_roster')


if __name__=='__main__':unittest.main()
