"""Protect isolation of validation labels and rolling features in both split modes."""

import unittest

import numpy as np

from compare_time_blocks import buffered_blocks, time_blocks


class ValidationBlocksTests(unittest.TestCase):
    def test_every_public_frame_is_held_out_once(self):
        folds = buffered_blocks(11185, 18684, 25)
        np.testing.assert_array_equal(np.concatenate([v for _, _, v in folds]),
                                      np.arange(11185, 18685))

    def test_both_buffers_and_future_training_feature_windows_exclude_validation(self):
        for _, train, validation in buffered_blocks(11185, 18684, 25):
            self.assertEqual(len(np.intersect1d(train, validation)), 0)
            distance = np.minimum(abs(train - validation[0]), abs(train - validation[-1]))
            self.assertTrue((distance >= 251).all())
            # A future training row's past-second observations cannot touch validation.
            for shift in range(25):
                self.assertEqual(len(np.intersect1d(train - shift, validation)), 0)

    def test_last_buffered_fold_matches_last_chronological_fold(self):
        blocked = buffered_blocks(11185, 18684, 25)[-1]
        chronological = time_blocks(11185, 18684, 25)[-1]
        np.testing.assert_array_equal(blocked[1], chronological[1])
        np.testing.assert_array_equal(blocked[2], chronological[2])


if __name__ == "__main__":
    unittest.main()
