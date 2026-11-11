import unittest
import numpy as np
from wasb_adapter import inverse_affine,warp,normalize,blobs


class WasbAdapterTests(unittest.TestCase):
    def test_published_unrotated_affine_identity_and_exact_integer_decimation(self):
        rng=np.random.default_rng(38);image=rng.integers(0,256,(288,512,3),dtype=np.uint8)
        np.testing.assert_array_equal(warp(image),image)
        large=np.repeat(np.repeat(image,2,0),2,1);np.testing.assert_array_equal(warp(large),image)
        m=inverse_affine(1920,1080);np.testing.assert_array_equal(m@np.array([256.,144.,1]),[960.,540.])
        self.assertEqual(normalize(image).shape,(3,288,512));self.assertTrue(np.isfinite(normalize(image)).all())

    def test_strict_half_threshold_diagonal_connectivity_mass_and_inverse(self):
        hm=np.zeros((288,512),np.float32);hm[20,40]=.75;hm[21,41]=1;hm[22,42]=.5
        results=blobs(hm,1024,576);self.assertEqual(len(results),1)
        np.testing.assert_allclose(results[0],[2*(40*.75+41)/1.75,2*(20*.75+21)/1.75,1.75])
        self.assertEqual(blobs(np.zeros_like(hm),1920,1080),[])


if __name__=='__main__':unittest.main()
