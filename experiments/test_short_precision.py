"""Structural checks for constrained endpoint research on synthetic sequences."""

import itertools
import unittest

import numpy as np

from boundary_refinement import runs
from short_precision import ownership_evidence, refine_short_endpoints


class ShortPrecisionTests(unittest.TestCase):
    def test_joint_solution_matches_exhaustive_likelihood(self):
        states = np.r_[np.zeros(9, int), np.ones(10, int), np.zeros(9, int)]
        rng = np.random.default_rng(42)
        p = rng.uniform(.1, .9, (len(states), 2))
        p /= p.sum(axis=1, keepdims=True)
        actual = refine_short_endpoints(states, p, 3)
        costs = []
        for left, right in itertools.product(range(6, 13), range(16, 23)):
            if right-left < 8:
                continue
            trial = np.r_[np.zeros(left, int), np.ones(right-left, int), np.zeros(len(states)-right, int)]
            costs.append(-np.log(p[np.arange(len(states)), trial]).sum())
        self.assertAlmostEqual(-np.log(p[np.arange(len(states)), actual]).sum(), min(costs))

    def test_adjacent_islands_preserve_topology_dead_and_minimum(self):
        states = np.r_[np.full(7, 2), np.zeros(20, int), np.ones(10, int),
                       np.zeros(12, int), np.ones(20, int), np.full(7, 2)]
        rng = np.random.default_rng(43)
        p = rng.uniform(.01, 1, (len(states), 2))
        result = refine_short_endpoints(states, p, 25)
        self.assertEqual([r[2] for r in runs(result)], [r[2] for r in runs(states)])
        np.testing.assert_array_equal(result == 2, states == 2)
        self.assertTrue(all(end-start >= 8 for start, end, _ in runs(result)[2:4]))
        np.testing.assert_array_equal(refine_short_endpoints(states, p, 0), states)

    def test_symmetry_identity_and_no_eligible_island(self):
        states = np.r_[np.zeros(20, int), np.ones(10, int), np.zeros(20, int)]
        p = np.tile([.8, .2], (len(states), 1))
        p[22:30] = [.2, .8]
        result = refine_short_endpoints(states, p, 13)
        np.testing.assert_array_equal(refine_short_endpoints(1-states, p[:, ::-1], 13), 1-result)
        np.testing.assert_array_equal(refine_short_endpoints(states, np.full_like(p, .5), 13), states)
        edge = np.r_[np.zeros(10, int), np.ones(20, int)]
        np.testing.assert_array_equal(refine_short_endpoints(edge, p[:30], 13), edge)

    def test_evidence_symmetry_and_validation(self):
        old = np.array([[.2, .5, .3], [0, 0, 1], [.7, .1, .2]])
        fast = np.array([[.1, .9], [.4, .6], [.9, .1]])
        for log_pool in (False, True):
            p = ownership_evidence(old, fast, "joint", 1, log_pool)
            q = ownership_evidence(old[:, [1, 0, 2]], fast[:, ::-1], "joint", 1, log_pool)
            np.testing.assert_allclose(p[:, ::-1], q)
            np.testing.assert_allclose(p.sum(axis=1), 1)
        with self.assertRaises(ValueError):
            refine_short_endpoints(np.array([0]), np.array([[np.nan, .5]]), 1)
        with self.assertRaises(ValueError):
            ownership_evidence(old, fast, "unknown", 1)


if __name__ == "__main__":
    unittest.main()
