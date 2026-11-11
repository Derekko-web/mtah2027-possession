"""Bounded dead growth on the correct side of an unobserved ownership flip."""
import unittest
import numpy as np
from test_spurious_stoppage_fragments import StoppageFragmentsTests
from stoppage_gap_closure import refine


class GapClosureTests(unittest.TestCase):
    def test_trailing_dead_allowance_retains_live_prefix(self):
        data=StoppageFragmentsTests().fixture()
        result,events=refine(*data,dead_extension=25)
        self.assertTrue(np.all(result[:55]=='A'));self.assertTrue(np.all(result[55:]=='D'))
        self.assertEqual(events[0]['dead_start'],55)

    def test_leading_dead_allowance_preserves_following_control(self):
        data=StoppageFragmentsTests().fixture('D'*70+'H'*30+'A'*50)
        result,_=refine(*data,dead_extension=13)
        self.assertTrue(np.all(result[:83]=='D'));self.assertTrue(np.all(result[83:]=='A'))

    def test_short_gap_is_clipped_and_swap_preserved(self):
        data=StoppageFragmentsTests().fixture('A'*50+'H'*10+'D'*70)
        result,_=refine(*data);self.assertTrue(np.all(result[50:]=='D'))
        swap=lambda x:np.where(x=='H','A',np.where(x=='A','H',x))
        changed=list(data);changed[0]=swap(data[0]);other,_=refine(*changed)
        np.testing.assert_array_equal(other,swap(result))

    def test_camera_evidence_and_block_edges_are_protected(self):
        data=list(StoppageFragmentsTests().fixture())
        data[-1]=[(0,80),(80,len(data[0]))]
        result,_=refine(*data);np.testing.assert_array_equal(result,data[0])
        with self.assertRaises(ValueError):refine(*data,dead_extension=26)


if __name__=='__main__':unittest.main()
