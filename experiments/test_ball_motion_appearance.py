import unittest

import numpy as np

from ball_appearance import HEIGHT, WIDTH
from ball_motion_appearance import MOTION_FEATURES, compensated_features


class BallMotionAppearanceTests(unittest.TestCase):
    def image(self):
        return np.full((HEIGHT, WIDTH, 3), [20, 120, 30], np.uint8)

    def test_static_ground_marking_has_no_compensated_motion(self):
        image = self.image()
        image[:, 599:602] = 230
        value = compensated_features([image]*3, [np.eye(3)]*3, 600, 350, 3, 2)
        self.assertEqual(value.shape, (MOTION_FEATURES,))
        np.testing.assert_allclose(value[103:], 0, atol=1e-6)

    def test_translating_camera_cancels_background(self):
        current = self.image()
        current[340:360, 595:605] = 230
        moved = np.roll(current, [3, 4], axis=[0, 1])
        transform = np.array([[1., 0, 4], [0, 1, 3], [0, 0, 1]])
        value = compensated_features([moved, current, moved], [transform, np.eye(3), transform], 600, 350, 3, 2)
        np.testing.assert_allclose(value[103:], 0, atol=1e-6)

    def test_moving_small_spot_produces_residual_and_rejects_border(self):
        empty = self.image()
        current = empty.copy()
        current[348:353, 598:603] = 230
        value = compensated_features([empty, current, empty], [np.eye(3)]*3, 600, 350, 3, 2)
        self.assertGreater(np.linalg.norm(value[103:]), 1)
        self.assertIsNone(compensated_features([empty]*3, [np.eye(3)]*3, 1, 1, 3, 2))


if __name__ == '__main__':
    unittest.main()
