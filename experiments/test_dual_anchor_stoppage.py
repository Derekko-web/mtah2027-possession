"""The combined decision uses original inputs and only the two bounded edges."""
import unittest
import numpy as np
import test_spurious_stoppage_fragments as fixtures
from dual_anchor_stoppage import refine


class DualAnchorTests(unittest.TestCase):
    def test_earlier_supported_dead_edge_wins(self):
        data=list(fixtures.StoppageFragmentsTests().fixture('A'*70+'H'*40+'D'*70))
        data[2].loc[:59,'ball_seen']=1
        result,event=refine(*data)
        self.assertTrue(np.all(result[:85]=='A'));self.assertTrue(np.all(result[85:]=='D'))
        self.assertEqual(event[0]['dead_start'],85)

    def test_visibility_can_close_more_than_one_second_of_late_prediction(self):
        data=list(fixtures.StoppageFragmentsTests().fixture('A'*70+'H'*70+'D'*40))
        data[2].loc[:59,'ball_seen']=1
        result,_=refine(*data)
        self.assertTrue(np.all(result[85:]=='D'));self.assertTrue(np.all(result[:85]=='A'))


if __name__=='__main__':unittest.main()
