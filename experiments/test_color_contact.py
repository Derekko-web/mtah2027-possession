"""Synthetic contact evidence, episode reset and ownership protection checks."""
import unittest
import numpy as np
from color_contact import contact_scores, contact_posterior, predict_contact_block


class ColorContactTests(unittest.TestCase):
    def test_contact_scores_reject_uncertain_colors_and_swap_teams(self):
        xy = np.array([[0., 0.], [5., 0.]])
        q = np.array([[.1, .9], [.95, .05]])
        first = contact_scores(np.array([0., 0.]), xy, q)
        np.testing.assert_array_equal(contact_scores(np.array([0., 0.]), xy, q[:, ::-1]), first[::-1])
        self.assertAlmostEqual(first[1], .9)
        self.assertTrue(np.isnan(contact_scores(np.array([0., 0.]), xy, np.full((2, 2), .5))).all())

    def test_memory_expires_and_resets_at_dead_ball(self):
        reference = np.zeros(200, int)
        score = np.full((200, 2), np.nan)
        score[10] = [.01, .99]
        posterior, support = contact_posterior(reference, score)
        np.testing.assert_array_equal(posterior[85], [.01, .99])
        np.testing.assert_array_equal(posterior[86], [.9, .1])
        self.assertEqual(int(support.sum()), 1)
        reference[30:40] = 2
        posterior, _ = contact_posterior(reference, score)
        np.testing.assert_array_equal(posterior[40], [.9, .1])

    def test_new_short_island_is_possible_and_team_symmetric(self):
        reference = np.zeros(220, int)
        score = np.full((220, 2), np.nan)
        score[[40, 50, 60]] = [.01, .99]
        score[85] = [.99, .01]
        first = predict_contact_block(reference, score)
        self.assertGreater(int((first == 1).sum()), 20)
        swapped = predict_contact_block(1-reference, score[:, ::-1])
        np.testing.assert_array_equal(swapped, 1-first)
        reference[100:120] = 2
        result = predict_contact_block(reference, score)
        np.testing.assert_array_equal(result == 2, reference == 2)

    def test_opposite_existing_short_island_is_preserved(self):
        reference = np.zeros(240, int)
        reference[30:50] = 1
        reference[90:110] = 1
        score = np.full((240, 2), np.nan)
        score[[30, 45, 60, 80]] = [.01, .99]
        score[110] = [.99, .01]
        result = predict_contact_block(reference, score)
        np.testing.assert_array_equal(result[50:90], reference[50:90])


if __name__ == '__main__':
    unittest.main()
