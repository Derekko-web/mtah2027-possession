"""Known synthetic camera correspondences, cuts, ambiguity and set symmetry."""

import unittest

import numpy as np

from broadcast_alignment import (PITCH,formation_anchors,formation_cost,map_anchors,
                                  reconcile_teams,transform)

PROFILES = {"strict":dict(maximum_cost=2.5,minimum_margin=.5,maximum_fine_span=10)}


def synthetic_sources():
    rng = np.random.default_rng(42)
    n,slots,offset = 350,12,125
    teams = np.repeat([0,1],6)
    initial = rng.uniform([20,15],[75,50],(slots,2))
    xy = initial+np.cumsum(rng.normal(0,.4,(n,slots,2)),axis=0)
    tactical = dict(positions=xy,teams=teams,valid=np.ones(n,bool))
    other = rng.uniform([0,0],PITCH,(650,slots,2))
    other[offset:offset+n] = transform(xy,3)
    permutation = rng.permutation(slots)
    broadcast = dict(positions=other[:,permutation],teams=teams[permutation],valid=np.ones(650,bool))
    return tactical,broadcast,offset


class BroadcastAlignmentTests(unittest.TestCase):
    def test_known_offset_reflection_and_slot_permutation(self):
        tactical,broadcast,offset = synthetic_sources()
        anchors = formation_anchors(tactical,broadcast,PROFILES)
        mapping = map_anchors(anchors,tactical,broadcast,PROFILES["strict"])
        valid = mapping["frames"] >= 0
        self.assertGreater(valid.sum(),150)
        self.assertLessEqual(np.max(np.abs(mapping["frames"][valid]-np.flatnonzero(valid)-offset)),1)
        self.assertTrue(np.all(mapping["orientation"][valid] == 3))

    def test_cuts_and_missing_players_cannot_be_interpolated_through(self):
        tactical,broadcast,offset = synthetic_sources()
        broadcast["valid"][offset+125:offset+175] = False
        anchors = formation_anchors(tactical,broadcast,PROFILES)
        mapping = map_anchors(anchors,tactical,broadcast,PROFILES["strict"])
        self.assertTrue(np.all(mapping["frames"][125:175] == -1))
        self.assertTrue(np.all(mapping["frames"][:25] == -1))

    def test_set_cost_is_team_symmetric_and_ignores_slot_order(self):
        tactical,broadcast,_ = synthetic_sources()
        a,b = tactical["positions"][100],broadcast["positions"][200:210]
        ta,tb = tactical["teams"],broadcast["teams"]
        first = formation_cost(a,b,ta,tb,3)
        second = formation_cost(a[::-1],b[:,::-1],ta[::-1],tb[::-1],3)
        np.testing.assert_allclose(first,second)
        np.testing.assert_allclose(first,formation_cost(a,b,1-ta,1-tb,3))
        np.testing.assert_array_equal(reconcile_teams(("Home FC","Away Club"),("Away Club","Home FC"),tb),1-tb)

    def test_identical_far_replay_is_rejected_by_margin(self):
        tactical,broadcast,offset = synthetic_sources()
        replay = np.concatenate([broadcast["positions"],broadcast["positions"][offset:offset+350]],axis=0)
        broadcast["positions"] = replay
        broadcast["valid"] = np.ones(len(replay),bool)
        anchors = formation_anchors(tactical,broadcast,PROFILES)
        mapping = map_anchors(anchors,tactical,broadcast,PROFILES["strict"])
        self.assertFalse((mapping["frames"] >= 0).any())


if __name__ == "__main__":
    unittest.main()
