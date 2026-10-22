"""Verify decoding optimality and meaningful possession-sequence behaviours."""

from itertools import product
import unittest

import numpy as np

from transition_penalty import decode


class TransitionPenaltyTests(unittest.TestCase):
    def test_agrees_with_exhaustive_search(self):
        rng = np.random.default_rng(42)
        for penalty in [0, .7, 3]:
            p = rng.dirichlet(np.ones(3), size=5)
            def objective(path):
                return -sum(np.log(p[t, s]) for t, s in enumerate(path)) + penalty * sum(
                    a != b for a, b in zip(path, path[1:]))
            optimum = min(objective(path) for path in product(range(3), repeat=5))
            self.assertAlmostEqual(objective(decode(p, penalty)), optimum)

    def test_zero_penalty_is_independent_frame_prediction(self):
        p = np.array([[.8, .1, .1], [.1, .7, .2], [.1, .1, .8]])
        np.testing.assert_array_equal(decode(p, 0), p.argmax(axis=1))

    def test_weak_one_frame_flip_is_removed(self):
        p = np.tile([.9, .05, .05], (21, 1))
        p[10] = [.1, .85, .05]
        np.testing.assert_array_equal(decode(p), np.zeros(21, dtype=int))

    def test_sustained_strong_evidence_keeps_team_change_and_dead_ball(self):
        p = np.repeat([[.98, .01, .01], [.01, .98, .01], [.01, .01, .98]], 10, axis=0)
        np.testing.assert_array_equal(decode(p), np.repeat([0, 1, 2], 10))

    def test_invalid_probability_rows_are_rejected(self):
        for p in [[[0, 0, 0]], [[np.nan, .5, .5]], [[-.1, .5, .6]]]:
            with self.assertRaises(ValueError):
                decode(p)


if __name__ == "__main__":
    unittest.main()
