"""Synthetic detection matching, visibility and bounded-context controls."""
import unittest
import numpy as np
import pandas as pd
from detector_features import consensus_pairs, sample_features, interpolate_samples, feature_views, PRESENCE, GEOMETRY
from detector_roi import crop_centers
from rich_features import swap_teams


class DetectorFeatureTests(unittest.TestCase):
    def test_center_brackets_are_bounded_and_do_not_cross_halves(self):
        ball = np.full((100, 2), np.nan)
        ball[10], ball[40], ball[70] = [10, 10], [40, 10], [70, 10]
        before = ball.copy()
        centers, flags = crop_centers(ball, [(0, 50), (50, 100)])
        self.assertTrue(np.isnan(centers[0]).all())
        self.assertTrue(np.isnan(centers[11]).all())
        np.testing.assert_array_equal(centers[25], [25, 10])
        self.assertTrue(flags[25])
        self.assertTrue(np.isnan(centers[55]).all())
        np.testing.assert_array_equal(ball, before)

    def test_consensus_requires_mutual_nearest_and_fixed_radius(self):
        first = [(.8, np.array([0., 0., 4., 4.])), (.7, np.array([3., 0., 4., 4.]))]
        self.assertEqual(len(consensus_pairs(first, [(.9, first[0][1])])), 1)
        self.assertEqual(len(consensus_pairs(first[:1], [(.9, np.array([8., 0., 4., 4.]))])), 1)
        self.assertEqual(consensus_pairs(first[:1], [(.9, np.array([9., 0., 4., 4.]))]), [])

    def test_missing_crops_and_geometry_are_not_filled(self):
        samples = pd.DataFrame(1., index=[0, 5, 15], columns=PRESENCE+GEOMETRY)
        samples.loc[5, 'vd_offset_norm'] = np.nan
        dense = interpolate_samples(samples, 20)
        self.assertTrue(np.isnan(dense.vd_offset_norm.iloc[2]))
        self.assertEqual(dense.vd_crop_observed.iloc[10], 0.)
        self.assertEqual(dense.vd_crop_observed.iloc[16], 0.)
        self.assertEqual(dense.vd_crop_observed.iloc[2], 1.)

    def test_geometry_swaps_teams_without_changing_global_observations(self):
        record = dict(x=50., y=50., diameter=2., interpolated=False)
        prediction = dict(objects=[dict(box=[.4921875, .4953125, .015625, .009375],
            labels=[dict(label='sports ball', confidence=.9)])])
        xy = np.array([[50., 50.], [58., 50.]])
        matrix = np.diag([105.3, 68., 1.])
        first = pd.DataFrame([sample_features(record, prediction, prediction, xy, np.array([0, 1]), matrix, .2)])
        second = pd.DataFrame([sample_features(record, prediction, prediction, xy, np.array([1, 0]), matrix, .2)])
        pd.testing.assert_frame_equal(second, swap_teams(first), check_exact=True)
        self.assertAlmostEqual(first.h__vd_distance1.iloc[0], 0.)
        self.assertAlmostEqual(first.a__vd_distance1.iloc[0], 8.)

    def test_temporal_features_have_exact_prefix_parity(self):
        samples = pd.DataFrame(np.random.default_rng(42).normal(size=(120, len(PRESENCE+GEOMETRY))),
                               index=np.arange(0, 600, 5), columns=PRESENCE+GEOMETRY)
        long, _ = feature_views(samples, 600, [(0, 600)])
        short, _ = feature_views(samples, 450, [(0, 600)])
        for name in long:
            pd.testing.assert_frame_equal(long[name].iloc[:300], short[name].iloc[:300], check_exact=True)


if __name__ == '__main__':
    unittest.main()
