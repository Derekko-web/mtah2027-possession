import unittest

import numpy as np
import pandas as pd

from tabicl_native import context_record, query_indices


class NativeContextTests(unittest.TestCase):
    def test_only_live_5hz_training_rows_and_train_medians(self):
        inputs = pd.DataFrame({'h__x': np.full(19000, np.nan), 'a__x': np.full(19000, np.nan),
                               'empty': np.full(19000, np.nan)})
        inputs.loc[11185, ['h__x', 'a__x']] = [1., 7.]
        inputs.loc[11190, ['h__x', 'a__x']] = [3., 5.]
        inputs.loc[11200] = [999., 999., 999.]
        labels = np.full(19000, 'X')
        labels[[11185, 11186, 11190, 11195]] = list('HHAD')
        context, target, imputer, record = context_record(inputs, labels,
            np.array([11185, 11186, 11190, 11195]), np.array([11200]))
        np.testing.assert_array_equal(record['context_frames'], [11185, 11190])
        np.testing.assert_array_equal(target, [0, 1, 1, 0])
        np.testing.assert_array_equal(imputer.statistics_, [4., 4., 0.])
        self.assertEqual(context.shape, (4, 3))
        np.testing.assert_array_equal(context[:, 2], np.zeros(4))
        labels[11200] = 'A'
        second = context_record(inputs, labels, np.array([11185, 11186, 11190, 11195]), np.array([11200]))
        np.testing.assert_array_equal(second[0], context)
        np.testing.assert_array_equal(second[1], target)

    def test_fixed_query_shape_and_last_row_padding(self):
        batches = list(query_indices(513))
        np.testing.assert_array_equal(batches[0][2], np.arange(512))
        np.testing.assert_array_equal(batches[1][2], np.full(512, 512))
        self.assertEqual([(a, b) for a, b, _ in batches], [(0, 512), (512, 513)])

    def test_overlap_and_empty_context_rejected(self):
        inputs = pd.DataFrame({'h__x': np.zeros(19000), 'a__x': np.zeros(19000)})
        labels = np.full(19000, 'D')
        with self.assertRaises(ValueError):
            context_record(inputs, labels, np.array([11185]), np.array([11185]))
        with self.assertRaises(ValueError):
            context_record(inputs, labels, np.array([11185]), np.array([11200]))


if __name__ == '__main__':
    unittest.main()
