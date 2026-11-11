"""Insufficient unseen continuation cannot borrow the neighboring long span."""
import unittest
import numpy as np
import test_spurious_stoppage_fragments as fixtures
from supported_gap_stoppage import refine


class SupportedGapTests(unittest.TestCase):
    def test_short_continuation_is_refused(self):
        data=list(fixtures.StoppageFragmentsTests().fixture('A'*70+'H'*30+'D'*70))
        data[2].loc[:59,'ball_seen']=1
        result,events=refine(*data)
        np.testing.assert_array_equal(result,data[0]);self.assertEqual(events,[])

    def test_eight_frame_continuation_is_allowed(self):
        data=list(fixtures.StoppageFragmentsTests().fixture('A'*70+'H'*33+'D'*70))
        data[2].loc[:59,'ball_seen']=1
        result,events=refine(*data)
        self.assertEqual(events[0]['live_continuation_frames'],8)
        self.assertTrue(np.all(result[:78]=='A'));self.assertTrue(np.all(result[78:]=='D'))

    def test_observed_stoppage_does_not_need_invisible_continuation(self):
        data=list(fixtures.StoppageFragmentsTests().fixture());data[2]['ball_seen']=1
        result,events=refine(*data)
        self.assertEqual(events[0]['kind'],'observed_stoppage');self.assertTrue(np.all(result[50:]=='D'))


if __name__=='__main__':unittest.main()
