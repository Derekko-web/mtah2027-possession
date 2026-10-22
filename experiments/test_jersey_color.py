"""Synthetic crop/color and exclusion checks without competition images."""

import unittest

import numpy as np

from jersey_color import color_statistics,torso_bounds,weak_frame_split


class JerseyColorTests(unittest.TestCase):
    def test_color_statistics_distinguishes_uniforms_and_rejects_grass(self):
        red = np.full((8,8,3),[200,20,30],dtype=np.uint8)
        blue = np.full((8,8,3),[20,30,200],dtype=np.uint8)
        a,b = color_statistics(red),color_statistics(blue)
        self.assertEqual(a.shape,(124,))
        self.assertAlmostEqual(float(a[:108].sum()),1.)
        self.assertGreater(float(np.linalg.norm(a-b)),1.)
        self.assertIsNone(color_statistics(np.full((8,8,3),[20,100,20],dtype=np.uint8)))
        self.assertIsNone(color_statistics(red[:1,:1]))

    def test_projection_has_sensible_torso_and_rejects_missing(self):
        matrix = np.array([[1920,0,0],[0,1080,0],[0,0,1.]])
        box = torso_bounds(matrix,[52.65,34.])
        self.assertIsNotNone(box)
        self.assertLess(box[3],270)
        self.assertLess(box[0],480)
        self.assertGreater(box[2],480)
        self.assertIsNone(torso_bounds(matrix,[np.nan,34]))
        self.assertIsNone(torso_bounds(matrix,[0,0]))

    def test_weak_split_excludes_public_reserved_and_validation_neighbors(self):
        reserved = [dict(start=40000,end=41000)]
        a,b = weak_frame_split(199319,[(1,11185,90651),(2,110808,190197)],reserved)
        self.assertEqual((len(a),len(b)),(240,80))
        self.assertTrue((np.abs(a[:,None]-b[None]) > 250).all())
        for values in (a,b):
            self.assertFalse(((values >= 10935)&(values <= 18934)).any())
            self.assertFalse(((values >= 39750)&(values <= 41250)).any())


if __name__ == "__main__": unittest.main()
