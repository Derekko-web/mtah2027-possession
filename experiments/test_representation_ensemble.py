import unittest

import numpy as np

from representation_ensemble import equal_probability_pool


class ProbabilityPoolTests(unittest.TestCase):
    def test_fixed_average_preserves_distribution_and_identity(self):
        first = np.array([[.8, .1, .1], [.1, .5, .4]])
        second = np.array([[.2, .5, .3], [.3, .3, .4]])
        np.testing.assert_array_equal(equal_probability_pool(first, second), (first + second) / 2)
        np.testing.assert_array_equal(equal_probability_pool(first, first), first)
        np.testing.assert_array_equal(equal_probability_pool(second, first), equal_probability_pool(first, second))
        np.testing.assert_allclose(equal_probability_pool(first, second).sum(axis=1), 1.)

    def test_team_swap_commutes_with_pooling(self):
        first = np.array([[.8, .1, .1], [.1, .5, .4]])
        second = np.array([[.2, .5, .3], [.3, .3, .4]])
        swap = [1, 0, 2]
        np.testing.assert_array_equal(
            equal_probability_pool(first[:, swap], second[:, swap]),
            equal_probability_pool(first, second)[:, swap])

    def test_rejects_invalid_probability_inputs(self):
        valid = np.array([[.2, .5, .3]])
        for invalid in (np.array([[np.nan, .5, .5]]), np.array([[-.1, .5, .6]]),
                        np.array([[.1, .1, .1]]), np.array([[.5, .5]]),
                        np.array([.2, .5, .3])):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    equal_probability_pool(valid, invalid)


if __name__ == "__main__":
    unittest.main()
