import unittest
import numpy as np
from PIL import Image
from ownership_montage import make_montage, SIZE


class OwnershipMontageTests(unittest.TestCase):
    def test_reference_and_ordered_scene_centers(self):
        reference = Image.new('RGB', (768, 360), (80, 80, 80))
        scenes = [Image.new('RGB', (100, 100), color) for color in ('red', 'green', 'blue')]
        image = make_montage(reference, scenes)
        self.assertEqual(image.size, SIZE)
        self.assertEqual(image.getpixel((768, 100)), (80, 80, 80))
        for x, color in ((256, (255, 0, 0)), (768, (0, 128, 0)), (1280, (0, 0, 255))):
            self.assertEqual(image.getpixel((x, 676)), color)

    def test_wide_scene_preserves_aspect_with_white_padding(self):
        reference = Image.new('RGB', (768, 360), 'white')
        scenes = [Image.new('RGB', (200, 50), 'red')]*3
        image = np.asarray(make_montage(reference, scenes))
        self.assertTrue((image[430, 20] == 255).all())
        self.assertTrue((image[676, 256] == [255, 0, 0]).all())
        with self.assertRaises(ValueError):
            make_montage(reference, scenes[:2])


if __name__ == '__main__':
    unittest.main()
