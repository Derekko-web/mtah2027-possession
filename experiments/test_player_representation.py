import unittest
import numpy as np
from player_representation import interpolate,live_overlay


class RepresentationTests(unittest.TestCase):
    def test_interpolation_requires_both_observed_endpoints(self):
        frames=np.arange(11185,18690,5);z=np.repeat(np.arange(len(frames),dtype=np.float32)[:,None],64,axis=1);support=np.ones(len(frames),bool);support[1]=False
        f,m=interpolate(frames,z,support,18935)
        self.assertTrue(m[11185]);self.assertFalse(m[11186:11195].any());self.assertTrue(m[11195])
        self.assertAlmostEqual(float(f[18684,0]),1499.8,places=3);self.assertFalse(m[:11185].any());self.assertFalse(m[18685:].any())

    def test_live_overlay_preserves_dead_unknown_and_alternate_abstentions(self):
        ref=np.array(list('HADXHA'));alt=np.array(list('AHHHXD'))
        np.testing.assert_array_equal(live_overlay(ref,alt),list('AHDXHA'))


if __name__=='__main__':unittest.main()
