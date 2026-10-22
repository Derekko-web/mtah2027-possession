"""Synthetic trajectory, symmetry and context checks for touch dynamics."""

import unittest

import numpy as np

from contact_dynamics import bounded_fill, geometry_features, impulse_features, repair_ball


class ContactDynamicsTests(unittest.TestCase):
    def test_repairs_single_outlier_and_preserves_linear_motion(self):
        ball = np.column_stack((np.arange(100)*.2, np.arange(100)*.05))
        noisy = ball.copy()
        noisy[50] += [30, -20]
        for method in ("median", "robust"):
            result = repair_ball(noisy, method)
            self.assertLess(np.linalg.norm(result[50]-ball[50]), 1.)
            np.testing.assert_allclose(result[20:30], ball[20:30], atol=1e-10)

    def test_only_complete_short_gaps_are_filled(self):
        ball = np.column_stack((np.arange(120)*.1, np.zeros(120)))
        ball[:3] = np.nan
        ball[15:20] = np.nan
        ball[40:54] = np.nan
        for method in ("median", "robust"):
            result = repair_ball(ball, method)
            self.assertTrue(np.isfinite(result[15:20]).all())
            self.assertTrue(np.isnan(result[40:54]).all())
            self.assertTrue(np.isnan(result[:3]).all())
        filled = bounded_fill(ball, 5)
        self.assertTrue(np.isnan(filled[40:54]).all())

    def test_geometric_invariance_and_team_swap(self):
        rng = np.random.default_rng(42)
        ball = rng.normal(size=(100, 2))
        xy = rng.normal(size=(100, 6, 2))
        speeds = rng.uniform(0, 4, (100, 6))
        teams = np.array([0, 0, 0, 1, 1, 1])
        original = ball+.1
        first = geometry_features(xy, speeds, ball, original, teams)
        rotation = np.array([[0., -1], [1, 0.]])
        second = geometry_features(xy@rotation+13, speeds, ball@rotation+13, original@rotation+13, teams)
        np.testing.assert_allclose(first, second, atol=1e-6)
        permutation = [4, 0, 2, 5, 1, 3]
        reordered = geometry_features(xy[:, permutation], speeds[:, permutation], ball, original, teams[permutation])
        np.testing.assert_array_equal(first, reordered)
        swapped = geometry_features(xy, speeds, ball, original, 1-teams)
        for name in first:
            opposite = ("a__" if name.startswith("h__") else "h__")+name[3:] if name.startswith(("h__", "a__")) else name
            np.testing.assert_array_equal(first[name], swapped[opposite])

    def test_touch_impulse_and_padding(self):
        ball = np.column_stack((np.arange(500)*.1, np.zeros(500)))
        ball[250:, 1] = np.arange(250)*.1
        xy = np.repeat(ball[:, None, :], 6, axis=1)
        xy[:, 3:] += 4
        speeds, teams = np.ones((500, 6)), np.array([0, 0, 0, 1, 1, 1])
        for method in ("median", "robust"):
            full = repair_ball(ball, method)
            prefix = repair_ball(ball[:400], method)
            np.testing.assert_allclose(full[125:275], prefix[125:275])
            geometry = geometry_features(xy, speeds, full, ball, teams)
            geometry_prefix = geometry_features(xy[:400], speeds[:400], prefix, ball[:400], teams)
            impulse = impulse_features(full, geometry)
            shorter = impulse_features(prefix, geometry_prefix)
            np.testing.assert_allclose(impulse.iloc[125:275], shorter.iloc[125:275])
            self.assertGreater(impulse.clean_impulse.iloc[250], .5)
            self.assertGreater(impulse.h__impulse_strength.iloc[250], impulse.a__impulse_strength.iloc[250])


if __name__ == "__main__":
    unittest.main()
