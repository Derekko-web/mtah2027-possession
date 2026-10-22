import unittest

import numpy as np
from PIL import Image

from ball_appearance import FEATURES, HEIGHT, PITCH, WIDTH, frame_features, image_geometry, negative_positions, patch_features, weak_split


class BallAppearanceTests(unittest.TestCase):
    def test_projection_scale_and_border_rejection(self):
        matrix = np.array([[1000., 0, 100], [0, 600, 50], [0, 0, 1]])
        x, y, diameter = image_geometry(matrix, PITCH/2)
        self.assertAlmostEqual(x, 600)
        self.assertAlmostEqual(y, 350)
        self.assertTrue(2 <= diameter <= 12)
        image = Image.new('RGB', (WIDTH, HEIGHT), (20, 120, 30))
        self.assertIsNone(patch_features(image, 1, 1, diameter, 2))
        self.assertEqual(patch_features(image, x, y, diameter, 1).shape, (FEATURES,))

    def test_local_center_signal_and_pixel_translation(self):
        image = np.full((HEIGHT, WIDTH, 3), [20, 120, 30], np.uint8)
        yy, xx = np.ogrid[:HEIGHT, :WIDTH]
        image[(xx-600)**2+(yy-350)**2 <= 4] = 230
        a = patch_features(Image.fromarray(image), 600, 350, 2, 2)
        moved = np.roll(image, [20, 30], axis=[0, 1])
        b = patch_features(Image.fromarray(moved), 630, 370, 2, 2)
        np.testing.assert_array_equal(a, b)
        empty = patch_features(Image.new('RGB', (WIDTH, HEIGHT), (20, 120, 30)), 600, 350, 2, 2)
        self.assertGreater(np.linalg.norm(a-empty), 1)

    def test_frame_targets_jitter_and_hard_negative_exclusion(self):
        ball = PITCH/2
        positions = np.array([ball+[1, 0], ball+[5, 0], ball+[-5, 0], ball+[9, 2]])
        negatives = negative_positions(ball, positions)
        self.assertTrue(all(np.linalg.norm(point-ball) >= 4 for point in negatives))
        image = np.full((HEIGHT, WIDTH, 3), [20, 120, 30], np.uint8)
        matrix = np.array([[1000., 0, 100], [0, 600, 50], [0, 0, 1]])
        values, labels = frame_features(image, matrix, ball, positions, [1., 2.], True)
        self.assertEqual(values.shape, (2, 11, FEATURES))
        self.assertEqual((labels == 1).sum(), 5)
        _, labels = frame_features(image, matrix, ball, positions, [1.], False)
        self.assertEqual((labels == 1).sum(), 1)

    def test_public_reserved_and_validation_exclusions(self):
        eligible = np.ones(50000, bool)
        reserved = [dict(start=30000, end=30200)]
        train, validation, _, _ = weak_split(eligible, [(1, 0, 49999)], reserved)
        self.assertEqual(len(train), 160)
        self.assertEqual(len(validation), 60)
        self.assertTrue((np.abs(train[:, None]-validation[None]) > 250).all())
        all_frames = np.r_[train, validation]
        self.assertFalse(((all_frames >= 10935) & (all_frames <= 18934)).any())
        self.assertFalse(((all_frames >= 29750) & (all_frames <= 30450)).any())
        self.assertTrue((all_frames % 25 == 0).all())


if __name__ == '__main__':
    unittest.main()
