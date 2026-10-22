"""Rectangular-input geometry controls without competition images."""

import unittest

import numpy as np
from PIL import Image

from modern_ball import letterbox_crop, modern_scores, unletterbox_objects


class ModernBallTests(unittest.TestCase):
    def test_letterbox_preserves_geometry_and_rejects_clipping(self):
        image = Image.new('RGB', (100, 100), (7, 17, 27))
        crop = letterbox_crop(image, 50, 50, 32)
        self.assertEqual(crop.size, (384, 640))
        self.assertEqual(crop.getpixel((192, 127)), (114, 114, 114))
        self.assertEqual(crop.getpixel((192, 128)), (7, 17, 27))
        self.assertEqual(crop.getpixel((192, 511)), (7, 17, 27))
        self.assertEqual(crop.getpixel((192, 512)), (114, 114, 114))
        self.assertIsNone(letterbox_crop(image, 5, 5, 32))
        self.assertIsNone(letterbox_crop(image, np.nan, 50, 32))

    def test_box_inverse_keeps_center_and_length(self):
        objects = [dict(box=[.45, 300/640, .1, 40/640], labels=[dict(label='sports ball', confidence=.9)])]
        result = unletterbox_objects(objects)
        np.testing.assert_allclose(result[0]['box'], [.45, 172/384, .1, 40/384])
        self.assertEqual(objects[0]['box'][1], 300/640)
        near, anywhere = modern_scores(objects, 64, 2)
        self.assertEqual((near, anywhere), (.9, .9))
        objects[0]['box'][1] = .8
        self.assertEqual(modern_scores(objects, 64, 2), (0., .9))


if __name__ == '__main__':
    unittest.main()
