"""Synthetic lag and context matching controls without competition data."""

import unittest

import numpy as np

from ball_location_diagnostics import context_diagnostics, lag_diagnostics
from broadcast_alignment import PITCH


class BallLocationDiagnosticsTests(unittest.TestCase):
    def test_lag_uses_current_camera_and_common_observations(self):
        ball = np.c_[np.arange(60, dtype=float), np.zeros(60)]
        matrices = np.tile(np.diag([PITCH[0], PITCH[1], 1.]), (60, 1, 1))
        matrices[:, 0, 2] = np.arange(60)*3.
        frames = np.array([20, 30])
        detected = np.c_[frames+2+frames*3., np.zeros(2)]
        result, distance, complete = lag_diagnostics(frames, detected, matrices, ball, [-2, -1, 0, 1, 2], np.ones(2))
        self.assertEqual(result['best_diagnostic_lags'], [2])
        self.assertEqual(result['median_pixel_residual'][-1], 0.)
        np.testing.assert_array_equal(complete, [True, True])
        ball[18] = np.nan
        result, _, complete = lag_diagnostics(frames, detected, matrices, ball, [-2, -1, 0, 1, 2], np.ones(2))
        self.assertEqual(result['complete_cases'], 1)
        np.testing.assert_array_equal(complete, [False, True])
        self.assertFalse(result['sufficient_for_lag_interpretation'])

    def test_context_matches_sets_and_requires_nearby_observations(self):
        first = [np.array([[10., 10.], [100., 100.]]), np.empty((0, 2)), np.array([[0., 0.]])]
        second = [np.array([[102., 100.], [11., 10.]]), np.array([[2., 2.]]), np.array([[20., 20.]])]
        result = context_diagnostics(first, second)
        self.assertEqual(result['both_detected_centers'], 2)
        self.assertEqual(result['consistent_centers'], 1)
        self.assertEqual(result['consistent_fraction'], .5)
        self.assertEqual(result['median_matched_center_distance'], 1.5)


if __name__ == '__main__':
    unittest.main()
