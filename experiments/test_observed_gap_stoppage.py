"""Observation-anchored closure protects continuation and decode boundaries."""
import unittest
import numpy as np
import test_spurious_stoppage_fragments as fixtures
from observed_gap_stoppage import refine


class VisibilityGapTests(unittest.TestCase):
    def fixture(self):
        data=list(fixtures.StoppageFragmentsTests().fixture('A'*70+'H'*40+'D'*70))
        data[2].loc[:59,'ball_seen']=1
        return data

    def test_dead_onset_uses_missing_observation_start(self):
        data=self.fixture();result,events=refine(*data)
        self.assertTrue(np.all(result[:85]=='A'));self.assertTrue(np.all(result[85:]=='D'))
        self.assertEqual(events[0]['visibility_anchor'],60)

    def test_original_dead_timing_does_not_move_visibility_anchor(self):
        first=self.fixture();second=self.fixture();second[0][110:120]='H'
        a,_=refine(*first);b,_=refine(*second);np.testing.assert_array_equal(a,b)

    def test_mirror_and_team_swap(self):
        data=self.fixture();result,_=refine(*data)
        mirror=[data[0][::-1].copy(),data[1][::-1].copy(),data[2].iloc[::-1].reset_index(drop=True),
            data[3].iloc[::-1].reset_index(drop=True),data[4]]
        other,_=refine(*mirror);np.testing.assert_array_equal(other,result[::-1])
        swap=lambda x:np.where(x=='H','A',np.where(x=='A','H',x))
        changed=list(data);changed[0]=swap(data[0]);other,_=refine(*changed)
        np.testing.assert_array_equal(other,swap(result))

    def test_lookup_cannot_cross_decode_boundary(self):
        data=self.fixture();data[2]['ball_seen']=0;data[-1]=[(0,50),(50,len(data[0]))]
        result,events=refine(*data)
        self.assertEqual(events[0]['visibility_anchor'],50)
        self.assertTrue(np.all(result[:75]=='A'));self.assertTrue(np.all(result[75:]=='D'))


if __name__=='__main__':unittest.main()
