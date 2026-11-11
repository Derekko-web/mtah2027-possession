import unittest
import numpy as np
import pandas as pd
from contact_heads import contact_view
from rich_features import swap_teams


class ContactHistoryTests(unittest.TestCase):
    def test_interpolation_requires_adjacent_samples_and_same_half(self):
        scores = np.full((80, 2), np.nan)
        scores[[10, 15, 25, 30, 35]] = [.8, .2]
        view = contact_view(scores, 'test', 80, [(0, 33), (33, 80)])
        self.assertTrue((view.cc_test_seen.iloc[10:16] == 1).all())
        self.assertTrue((view.cc_test_seen.iloc[16:25] == 0).all())
        self.assertTrue((view.cc_test_seen.iloc[31:35] == 0).all())

    def test_exact_prefix_and_team_symmetry(self):
        rng = np.random.default_rng(42)
        scores = np.full((700, 2), np.nan)
        scores[::5] = rng.random((140, 2))
        full = contact_view(scores, 'test', 700, [(0, 700)])
        prefix = contact_view(scores, 'test', 600, [(0, 700)])
        pd.testing.assert_frame_equal(full.iloc[:450], prefix.iloc[:450], check_exact=True)
        swapped = contact_view(scores[:, ::-1], 'test', 700, [(0, 700)])
        pd.testing.assert_frame_equal(swap_teams(full), swapped, check_exact=True)

    def test_future_and_past_have_bounded_support(self):
        scores = np.full((400, 2), np.nan)
        scores[200] = [.8, .2]
        view = contact_view(scores, 'test', 400, [(0, 400)])
        self.assertTrue(np.isnan(view.h__cc_test_future125_mean.iloc[74]))
        self.assertEqual(view.h__cc_test_future125_mean.iloc[75], np.float32(.8))
        self.assertEqual(view.h__cc_test_past125_mean.iloc[325], np.float32(.8))
        self.assertTrue(np.isnan(view.h__cc_test_past125_mean.iloc[326]))


if __name__ == '__main__':
    unittest.main()
