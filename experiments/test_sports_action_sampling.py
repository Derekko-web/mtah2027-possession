import unittest
import numpy as np
from sports_action_sampling import action_centers, clip_frames, contiguous_ranges


class SportsActionSamplingTests(unittest.TestCase):
    def test_windows_respect_halves_and_sealed_context(self):
        reserved = [dict(start=13000, end=13020)]
        held = np.array([15000])
        centers = action_centers(19060, [('half',11185,20000)], reserved, held)
        self.assertTrue(len(centers) > 0)
        for center in centers:
            frames = clip_frames(center)
            self.assertEqual(len(frames),100)
            self.assertGreaterEqual(frames[0],11185)
            self.assertLess(frames[-1],19060)
            self.assertFalse(((frames >= 12750) & (frames <= 13270)).any())
            self.assertFalse((np.abs(frames-15000) <= 250).any())

    def test_prefix_keeps_every_public_summary_source_choice(self):
        blocks = [('half',11185,30000)]
        full = action_centers(19060, blocks, [], [])
        short = action_centers(18935, blocks, [], [])
        for frame in range(11185-50,18685+50):
            def selected(centers):
                order = np.lexsort((centers,np.abs(centers-frame)))
                c = centers[order[0]]
                return int(c) if abs(c-frame) <=25 else None
            self.assertEqual(selected(full),selected(short))

    def test_decoder_ranges_cover_exact_frame_set(self):
        frames = np.array([10,12,14,30,32,50])
        ranges = contiguous_ranges(frames)
        rebuilt = np.concatenate([np.arange(a,b+1,2) for a,b in ranges])
        np.testing.assert_array_equal(frames,rebuilt)


if __name__ == '__main__':
    unittest.main()
