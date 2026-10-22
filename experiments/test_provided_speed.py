"""Sparse-scalar, symmetry and bounded future-observation controls."""

import unittest

import numpy as np
import pandas as pd

from provided_speed import bounded_statistics, finish_speed, native_speed_core
from rich_features import swap_teams


class ProvidedSpeedTests(unittest.TestCase):
    def source(self, n):
        return dict(ball=np.tile([2., 0.], (n, 1)), positions=np.tile([[[1., 0.], [4., 0.]]], (n, 1, 1)),
            speeds=np.tile([3.6, 7.2], (n, 1)), teams=np.array([0, 1]), valid=np.ones(n, bool))

    def test_scalar_only_survives_missing_xy_and_converts_declared_units(self):
        source = self.source(40)
        source['ball'][20] = np.nan
        scalar = np.full(40, 3.6)
        core = native_speed_core(source, scalar, 1/3.6, np.arange(40))
        self.assertEqual(core.given.iloc[20], 1.)
        self.assertEqual(core.without_xy.iloc[20], 1.)
        self.assertTrue(np.isnan(core.h__nearest_signed_gap.iloc[20]))
        self.assertEqual(core.h__nearest_abs_gap.iloc[15], 0.)
        self.assertEqual(core.a__nearest_abs_gap.iloc[15], 1.)

    def test_team_swap_and_half_boundary_isolation(self):
        source = self.source(40)
        scalar = np.r_[np.full(20, np.nan), np.full(20, 3.6)]
        core = native_speed_core(source, scalar, 1/3.6, np.arange(40))
        source['teams'] = 1-source['teams']
        swapped = native_speed_core(source, scalar, 1/3.6, np.arange(40))
        pd.testing.assert_frame_equal(swapped, swap_teams(core), check_exact=True)
        finished = finish_speed(core, 'test_', [(0, 20), (20, 40)])
        self.assertTrue(finished.test_given_future125_mean.iloc[:20].isna().all())
        self.assertEqual(finished.test_given_future125_fraction.iloc[19], 0.)
        self.assertEqual(finished.test_given_past125_mean.iloc[20], 1.)

    def test_future_statistic_has_exact_bounded_prefix_parity(self):
        values = np.random.default_rng(42).normal(size=1000)
        values[::3] = np.nan
        first = bounded_statistics(values, 125, 'future')
        shorter = bounded_statistics(values[:800], 125, 'future')
        for a, b in zip(first, shorter):
            np.testing.assert_array_equal(a[:675], b[:675])


if __name__ == '__main__':
    unittest.main()
