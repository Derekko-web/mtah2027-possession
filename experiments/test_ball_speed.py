"""Checks the physical units, missing observations and causal speed window."""

import unittest

import numpy as np

from xgboost_baseline import recent_ball_speed


class BallSpeedTests(unittest.TestCase):
    def test_known_constant_motion_in_metres_per_second(self):
        # A ball moving 0.4 metres per frame at 25 fps is travelling at 10 m/s.
        xy = np.column_stack((np.arange(80) * .4, np.zeros(80)))
        speed = recent_ball_speed(xy, 25)
        self.assertTrue(np.isnan(speed[:13]).all())
        np.testing.assert_allclose(speed[13:], 10)

    def test_missing_gap_is_not_treated_as_motion_or_zero_speed(self):
        xy = np.zeros((100, 2))
        xy[30:65] = np.nan
        xy[65:, 0] = 100  # Reappearance is not a 100-metre instantaneous jump.
        speed = recent_ball_speed(xy, 25)
        self.assertTrue(np.isnan(speed[50:78]).all())
        np.testing.assert_allclose(speed[78:], 0)

    def test_future_positions_cannot_change_past_features(self):
        xy = np.column_stack((np.arange(100) * .4, np.zeros(100)))
        before = recent_ball_speed(xy, 25)
        xy[60:] = 10000
        after = recent_ball_speed(xy, 25)
        np.testing.assert_allclose(before[:60], after[:60], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
