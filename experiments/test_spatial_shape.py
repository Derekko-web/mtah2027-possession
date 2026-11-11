import unittest
import numpy as np
import pandas as pd
from spatial_shape import shape_features, spatial_core
from rich_features import swap_teams


class SpatialShapeTests(unittest.TestCase):
    def fixture(self, n=10):
        rng = np.random.default_rng(42)
        p = rng.normal(size=(n, 12, 2))*5
        p[:, :6, 0] -= 5
        p[:, 6:, 0] += 5
        return p, np.zeros((n, 2)), np.repeat([0, 1], 6), np.ones(n, bool)

    def test_team_and_slot_permutations(self):
        p, b, t, v = self.fixture()
        original = spatial_core(p, b, t, v, [(0, len(b))])
        swapped = spatial_core(p, b, 1-t, v, [(0, len(b))])
        pd.testing.assert_frame_equal(swap_teams(original), swapped, check_exact=True)
        order = np.arange(12)[::-1]
        permuted = spatial_core(p[:, order], b, t[order], v, [(0, len(b))])
        np.testing.assert_allclose(original, permuted, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_translation_reflection_and_ring_complements(self):
        p, b, t, v = self.fixture()
        original = spatial_core(p, b, t, v, [(0, len(b))])
        changed = spatial_core(p*[-1, 1]+[23, 7], b*[-1, 1]+[23, 7], t, v, [(0, len(b))])
        np.testing.assert_allclose(original, changed, rtol=1e-6, atol=1e-6, equal_nan=True)
        for radius in (5, 10, 20):
            np.testing.assert_allclose(original[f'h__shape_ring{radius}']+original[f'a__shape_ring{radius}'], 1, atol=1e-6)

    def test_missing_and_half_boundaries(self):
        p, b, t, v = self.fixture()
        p[0, :3] = np.nan
        b[1] = np.nan
        result = spatial_core(p, b, t, v, [(0, 6), (6, 10)])
        self.assertTrue(np.isnan(result.h__shape_scale.iloc[0]))
        self.assertTrue(np.isnan(result.h__shape_ring10.iloc[1]))
        self.assertTrue(np.isnan(result.shape_ball_speed5.iloc[6:]).all())

    def test_actual_prefix_and_summary_horizon(self):
        p, b, t, v = self.fixture(220)
        full = shape_features(p, b, t, v, [(0, 220)])
        prefix = shape_features(p[:200], b[:200], t, v[:200], [(0, 220)])
        pd.testing.assert_frame_equal(full.iloc[:120], prefix.iloc[:120], check_exact=True)


if __name__ == '__main__':
    unittest.main()
