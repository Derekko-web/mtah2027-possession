"""Synthetic co-motion, correspondence ambiguity, source gaps and event timing."""

import unittest

import numpy as np
import pandas as pd

from pair_motion import enclosure_mask,native_pair_features
from pair_motion_inference import motion_events
from pair_motion_symmetric import symmetric_motion_events


class PairMotionTests(unittest.TestCase):
    def source(self,n=80):
        t = np.arange(n)/25
        points = np.array([[1.,0.],[11.,0.],[25.,0.],[35.,0.]])
        xy = np.tile(points[None],(n,1,1));xy[:,:,0] += t[:,None]
        ball = np.column_stack((t,np.zeros(n)))
        return dict(positions=xy,ball=ball,teams=np.array([0,0,1,1]),valid=np.ones(n,bool))

    def test_joint_translation_has_zero_relative_motion_and_is_slot_invariant(self):
        source = self.source();frames = np.arange(30,50)
        a = native_pair_features(source,frames)
        self.assertTrue((a.h__past5_nearest_relative < 1e-5).all())
        self.assertTrue((a.h__future25_nearest_relative < 1e-5).all())
        self.assertGreater(float(a.h__past5_control_kernel.mean()),.8)
        permutation = np.array([3,1,2,0])
        other = dict(source,positions=source["positions"][:,permutation],teams=source["teams"][permutation])
        pd.testing.assert_frame_equal(a,native_pair_features(other,frames))

    def test_ambiguous_player_matches_and_native_source_cuts_stay_missing(self):
        source = self.source();source["positions"][:,1] = source["positions"][:,0]
        result = native_pair_features(source,np.arange(30,40))
        self.assertTrue(result.h__past5_nearest_relative.isna().all())
        source = self.source();source["valid"][34] = False
        result = native_pair_features(source,np.array([35]))
        self.assertTrue(np.isnan(result.h__past5_control_kernel.iloc[0]))

    def test_enclosure_rejects_gaps_orientation_changes_and_edges(self):
        mapping = dict(frames=np.arange(100)+1000,orientation=np.zeros(100,int))
        mask = enclosure_mask(mapping,5,10)
        self.assertFalse(mask[:10].any());self.assertTrue(mask[20])
        mapping["frames"][50] = -1;mapping["orientation"][75:] = 1
        mask = enclosure_mask(mapping,5,10)
        self.assertFalse(mask[40:61].any());self.assertFalse(mask[65:85].any())

    def test_motion_core_can_shorten_existing_algorithmic_source_extent(self):
        n = 400;base = np.ones(n,int);base[300:350] = 0
        reference = base.copy();reference[225:300] = 0
        kernel = np.tile([.01,.01],(n,1));kernel[250:270] = [.9,.01]
        distance = np.tile([1.,5.],(n,1));probability = np.tile([.99,.01],(n,1))
        result = motion_events(reference,base,probability,kernel,distance,np.ones(n,bool))
        self.assertTrue((result[225:245] == 1).all())
        self.assertTrue((result[245:275] == 0).all())
        self.assertTrue((result[275:300] == 1).all())
        self.assertTrue((result[300:350] == 0).all())

    def test_overlapping_events_have_team_symmetric_resolution(self):
        n = 800;base = np.ones(n,int);base[300:650] = 0
        reference = base.copy();reference[225:300] = 0
        kernel = np.tile([.01,.01],(n,1));kernel[250:270] = [.9,.01];kernel[270:290] = [.01,.9]
        distance = np.tile([1.,5.],(n,1));distance[270:] = [5.,1.]
        p = np.tile([.99,.01],(n,1));p[265:275] = [.5,.5];p[275:] = [.01,.99]
        observed = np.ones(n,bool)
        # The historical version's counterexample stays explicit; it cannot be promoted.
        old = motion_events(reference,base,p,kernel,distance,observed)
        old_swap = motion_events(1-reference,1-base,p[:,::-1],kernel[:,::-1],distance[:,::-1],observed)
        self.assertFalse(np.array_equal(old_swap,1-old))
        a = symmetric_motion_events(reference,base,p,kernel,distance,observed)
        b = symmetric_motion_events(1-reference,1-base,p[:,::-1],kernel[:,::-1],distance[:,::-1],observed)
        np.testing.assert_array_equal(b,1-a)


if __name__ == "__main__": unittest.main()
