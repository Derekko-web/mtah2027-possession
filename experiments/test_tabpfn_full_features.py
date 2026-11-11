"""Boundary and public-parity contracts for the separate full-match extension."""
import unittest
import numpy as np

from tabpfn_full_features import interpolate_halves, full_context_frames


class FullFeatureTests(unittest.TestCase):
    def hidden(self, frames):
        rng = np.random.default_rng(17)
        return rng.normal(size=(len(frames), 64)).astype(np.float32)

    def test_exact_public_arithmetic_matches_frozen_implementation(self):
        from player_representation import interpolate
        frames = np.arange(11185, 18690, 5)
        hidden = self.hidden(frames)
        supported = np.ones(len(frames), bool)
        supported[73:76] = False
        expected = interpolate(frames, hidden, supported, 18935)
        actual = interpolate_halves(frames, hidden, supported, 18935, [(11185, 18935)])
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a[11185:18685], b[11185:18685])

    def test_no_interpolation_across_halftime(self):
        frames = np.r_[np.arange(5, 21, 5), np.arange(40, 56, 5)]
        features, eligible = interpolate_halves(frames, self.hidden(frames), np.ones(len(frames), bool), 60, [(3, 23), (38, 58)])
        np.testing.assert_array_equal(np.flatnonzero(eligible), np.r_[np.arange(5, 21), np.arange(40, 56)])
        self.assertTrue(np.isnan(features[23:38]).all())

    def test_unsupported_endpoint_never_bridged(self):
        frames = np.arange(0, 21, 5)
        _, eligible = interpolate_halves(frames, self.hidden(frames), [True, True, False, True, True], 21, [(0, 21)])
        np.testing.assert_array_equal(np.flatnonzero(eligible), np.r_[np.arange(0, 6), np.arange(15, 21)])

    def test_missing_inside_half_center_rejected(self):
        frames = np.array([0, 5, 15])
        with self.assertRaises(ValueError):
            interpolate_halves(frames, self.hidden(frames), np.ones(3, bool), 20, [(0, 20)])

    def test_full_context_uses_only_live_centers_with_complete_public_footprint(self):
        labels = np.full(18935, "X")
        labels[11185:18685] = "H"
        labels[14000:16000] = "D"
        centers, observed = full_context_frames(labels)
        self.assertTrue(np.all(centers % 5 == 0))
        self.assertTrue(np.all(labels[centers] == "H"))
        self.assertEqual(observed[0], 11265)
        self.assertEqual(observed[-1], 18605)
        self.assertTrue(np.isin(centers, observed).all())

    def test_extra_nonpublic_target_rejected(self):
        labels = np.full(18935, "X")
        labels[11185:18685] = "H"
        labels[190] = "A"
        with self.assertRaises(AssertionError):
            full_context_frames(labels)


if __name__ == "__main__":
    unittest.main()
