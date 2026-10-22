"""Learned/source intersection can close at D while preserving brief references."""

import unittest

import numpy as np

from broadcast_intersection import intersect_broadcast
from test_broadcast_source_intervals import features


class BroadcastIntersectionTests(unittest.TestCase):
    def test_intersection_removes_unsupported_tail_and_accepts_D_closure(self):
        reference = np.ones(800,int); reference[500:510] = 2
        source = np.ones(800,int); source[440:500] = 0
        learned = np.ones(800,int); learned[480:500] = 0
        x = features(source)
        np.testing.assert_array_equal(intersect_broadcast(reference,learned,x,False),reference)
        result = intersect_broadcast(reference,learned,x,True)
        self.assertTrue(np.all(result[480:500] == 0))
        self.assertTrue(np.all(result[440:480] == 1))
        np.testing.assert_array_equal(result == 2,reference == 2)

    def test_protected_island_and_team_symmetry(self):
        reference = np.ones(600,int); reference[100:242] = 0
        source = np.zeros(600,int); source[140:180] = 1
        learned = reference.copy(); learned[140:180] = 1
        np.testing.assert_array_equal(intersect_broadcast(reference,learned,features(source),True),reference)
        reference = np.ones(600,int); learned[220:240] = 0
        source = np.ones(600,int); source[200:240] = 0
        result = intersect_broadcast(reference,learned,features(source),True)
        np.testing.assert_array_equal(intersect_broadcast(1-reference,1-learned,features(1-source),True),1-result)


if __name__ == "__main__": unittest.main()
