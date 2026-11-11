import unittest

import numpy as np

from nested_owner_stack import PUBLIC, CHUNKS, inner_splits, stack_inputs, meta_record


class NestedOwnerStackTests(unittest.TestCase):
    def test_inner_cover_and_two_exclusion_layers(self):
        validation = np.arange(12685, 14185)
        train = PUBLIC[(PUBLIC < validation[0]-250) | (PUBLIC >= validation[-1]+251)]
        splits = inner_splits(train, validation)
        np.testing.assert_array_equal(np.sort(np.concatenate([v for _, _, v in splits])), train)
        for (_, inner_train, inner_val), (start, end) in zip(splits, CHUNKS):
            self.assertTrue(np.isin(inner_train, train).all())
            self.assertTrue(np.isin(inner_val, train).all())
            self.assertFalse(np.intersect1d(inner_train, validation).size)
            self.assertFalse(np.intersect1d(inner_train, inner_val).size)
            self.assertTrue(((inner_train < start-250) | (inner_train >= end+250)).all())

    def test_log_odds_swap_and_dead_mass_invariance(self):
        original = np.array([[.4, .1, .5], [.1, .6, .3], [0., 0., 1.]])
        representation = np.array([[.1, .2, .7], [.5, .25, .25], [.2, .3, .5]])
        value = stack_inputs(original, representation)
        np.testing.assert_array_equal(stack_inputs(original[:, [1, 0, 2]], representation[:, [1, 0, 2]]), -value)
        np.testing.assert_allclose(value[:2, 0], np.log([4., 1/6]))
        self.assertTrue(np.isfinite(value).all())

    def test_meta_uses_only_live_inner_oof_rows_and_mirrors(self):
        labels = np.full(199319, "X", dtype="<U1")
        frames = PUBLIC[:3]
        labels[frames] = list("HAD")
        probability = np.tile([.4, .3, .3], (3, 1))
        record = meta_record(labels, frames, probability, probability)
        np.testing.assert_array_equal(record["train"], frames[:2])
        np.testing.assert_array_equal(record["inputs"][2:], -record["inputs"][:2])
        np.testing.assert_array_equal(record["target"], [1, 0, 0, 1])
        np.testing.assert_array_equal(record["weight"], np.ones(4))
        changed = labels.copy()
        changed[PUBLIC[3:]] = "D"
        other = meta_record(changed, frames, probability, probability)
        for key in record:
            np.testing.assert_array_equal(record[key], other[key])

    def test_invalid_expert_probabilities_and_outer_overlap_rejected(self):
        valid = np.array([[.4, .3, .3]])
        for bad in ([[.4, .4, .4]], [[np.nan, .5, .5]], [[-.1, .5, .6]], [[.4, .6]]):
            with self.assertRaises(ValueError):
                stack_inputs(valid, bad)
        with self.assertRaises(ValueError):
            inner_splits(PUBLIC[:10], PUBLIC[5:15])
        with self.assertRaises(ValueError):
            inner_splits(np.array([11185, 11185]), np.array([11186]))


if __name__ == "__main__":
    unittest.main()
