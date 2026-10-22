import unittest

import numpy as np

from ball_pooled import blob_map, pooled_features


class BallPooledTests(unittest.TestCase):
    def test_blob_curvature_rejects_an_elongated_line(self):
        y, x = np.indices((33, 33))
        spot = np.exp(-((x-16)**2+(y-16)**2)/8)
        line = np.exp(-(x-16)**2/8)
        a, _ = blob_map(spot, 1.)
        b, _ = blob_map(line, 1.)
        self.assertGreater(a[16, 16], 10*b[16, 16])

    def test_constant_and_spot_summaries_are_finite_and_distinct(self):
        rows = np.zeros((2, 778), np.float32)
        rgb = rows[1, :768].reshape(16, 16, 3)
        rgb[7:10, 6:9] = 2
        result = pooled_features(rows)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(np.linalg.norm(result[0]-result[1]), 1)
        np.testing.assert_array_equal(pooled_features(rows), result)


if __name__ == '__main__':
    unittest.main()
