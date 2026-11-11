import unittest

import numpy as np

from latent_seed_ensemble import member_seeds, pool_probabilities


class LatentSeedPoolTests(unittest.TestCase):
    def test_fixed_disjoint_seed_sets(self):
        sets = [set(member_seeds(seed)) for seed in (42, 43, 44)]
        self.assertEqual(member_seeds(42), (42, 142, 242))
        self.assertEqual(len(set.union(*sets)), 9)
        with self.assertRaises(ValueError):
            member_seeds(142)

    def test_probability_pool_and_team_symmetry(self):
        members = [np.array([[.8, .1, .1], [.1, .7, .2]], np.float32),
                   np.array([[.2, .5, .3], [.3, .3, .4]], np.float32),
                   np.array([[.3, .3, .4], [.6, .3, .1]], np.float32)]
        result = pool_probabilities(members)
        self.assertEqual(result.dtype, np.float64)
        expected = (members[0].astype(float) + members[1] + members[2]) / 3
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_allclose(result.sum(1), 1., rtol=1e-7)
        np.testing.assert_array_equal(pool_probabilities([p[:, [1, 0, 2]] for p in members]),
                                      result[:, [1, 0, 2]])
        np.testing.assert_allclose(pool_probabilities([members[0]] * 3), members[0], rtol=1e-7)

    def test_rejects_invalid_class_order_count_shape_and_values(self):
        valid = np.array([[.2, .5, .3]])
        for members in ([valid] * 2, [valid] * 4,
                        [valid, valid, np.array([[.5, .5]])],
                        [valid, valid, np.array([[np.nan, .5, .5]])],
                        [valid, valid, np.array([[-.1, .5, .6]])],
                        [valid, valid, np.array([[.1, .1, .1]])],
                        [valid, valid, np.vstack((valid, valid))]):
            with self.subTest(members=members):
                with self.assertRaises(ValueError):
                    pool_probabilities(members)
        with self.assertRaises(ValueError):
            pool_probabilities([valid] * 3, class_order=("A", "H", "D"))


if __name__ == "__main__":
    unittest.main()
