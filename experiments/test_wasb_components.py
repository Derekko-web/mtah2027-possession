import unittest
import numpy as np
from wasb_adapter import blobs,normalize,warp
from wasb_components import components,normalize_warped


class ComponentTests(unittest.TestCase):
    def test_components_match_scipy_under_diagonal_branches_and_edges(self):
        rng=np.random.default_rng(38002);hm=np.zeros((288,512),np.float32)
        hm[10:17,10:17]=rng.uniform(.3,.9,(7,7));hm[11:18,11:18]=.7;hm[0,0]=1;hm[-1,-1]=1
        np.testing.assert_array_equal(components(hm,1920,1080),blobs(hm,1920,1080))

    def test_prewarped_rgb_normalization_matches_frozen_adapter(self):
        rng=np.random.default_rng(38003);image=rng.integers(0,256,(576,1024,3),dtype=np.uint8)
        np.testing.assert_array_equal(normalize_warped(warp(image)),normalize(image))


if __name__=='__main__':unittest.main()
