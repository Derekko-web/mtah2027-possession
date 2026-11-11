import unittest

import numpy as np

from factorized_latent import compose_owner, training_record


class FactorizedOwnershipTests(unittest.TestCase):
    def test_composition_changes_only_conditional_owner(self):
        original = np.array([[.2, .3, .5], [.7, .2, .1], [0., 0., 1.]])
        owner = np.array([[.8, .2], [.1, .9], [.4, .6]])
        actual = compose_owner(original, owner)
        np.testing.assert_array_equal(actual[:, 2], original[:, 2])
        np.testing.assert_allclose(actual[:, :2], [[.4, .1], [.09, .81], [0., 0.]])
        np.testing.assert_allclose(actual.sum(1), 1.)
        np.testing.assert_array_equal(compose_owner(original[:, [1, 0, 2]], owner[:, [1, 0]]), actual[:, [1, 0, 2]])

    def test_training_omits_dead_and_validation_labels(self):
        labels = np.array(list("HADHAD"))
        train, validation = np.array([0, 1, 2]), np.array([3, 4, 5])
        record = training_record(labels, train, validation)
        changed = labels.copy()
        changed[validation] = "X"
        masked = training_record(changed, train, validation)
        for key, value in record.items():
            np.testing.assert_array_equal(value, masked[key])
        np.testing.assert_array_equal(record["train"], [0, 1])
        np.testing.assert_array_equal(record["target"], [0, 1, 1, 0])
        self.assertEqual(record["weight"].sum(), 4.)

    def test_rejects_bad_probabilities_and_overlapping_split(self):
        for owner in (np.array([[.8, .4]]), np.array([[np.nan, .2]]), np.array([[-.1, 1.1]])):
            with self.assertRaises(ValueError):
                compose_owner([[.2, .3, .5]], owner)
        with self.assertRaises(ValueError):
            training_record(np.array(list("HAD")), [0, 1], [1, 2])


if __name__ == "__main__":
    unittest.main()
