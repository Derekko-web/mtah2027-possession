"""Observed-envelope clipping properties and meaningful refusal cases."""
import unittest
import numpy as np
import pandas as pd
from broadcast_short_envelope import refine


class EnvelopeTests(unittest.TestCase):
    def fixture(self):
        ref=np.array(list('A'*30+'H'*80+'A'*30))
        d=np.full((140,2),4.);d[65:80,0]=1.
        bc=pd.DataFrame(dict(bc_matched=np.ones(140),bc_same_ball_seen=np.ones(140),
                            h__bc_same_distance_1=d[:,0],a__bc_same_distance_1=d[:,1]))
        return ref,bc,[(0,140)]

    def test_contact_envelope_tightens_both_edges_with_bounded_motion(self):
        ref,bc,blocks=self.fixture();actual,events=refine(ref,bc,blocks)
        self.assertEqual(events[0]['new_start'],52)
        self.assertEqual(events[0]['new_end_exclusive'],93)
        self.assertTrue(np.all(actual[52:93]=='H'))
        self.assertTrue(np.all(actual[30:52]=='A'))

    def test_one_missing_observation_keeps_reference(self):
        ref,bc,blocks=self.fixture();bc.loc[70,'bc_same_ball_seen']=0
        actual,events=refine(ref,bc,blocks);np.testing.assert_array_equal(actual,ref)
        self.assertEqual(events,[])

    def test_dead_flank_and_block_edges_are_protected(self):
        ref,bc,blocks=self.fixture();ref[:30]='D'
        actual,_=refine(ref,bc,blocks);np.testing.assert_array_equal(actual,ref)
        ref,bc,_=self.fixture()
        actual,_=refine(ref,bc,[(30,110)]);np.testing.assert_array_equal(actual,ref)

    def test_fewer_than_three_contacts_do_not_trigger_a_trim(self):
        ref,bc,blocks=self.fixture();bc['h__bc_same_distance_1']=4.;bc.loc[70:71,'h__bc_same_distance_1']=1.
        actual,_=refine(ref,bc,blocks);np.testing.assert_array_equal(actual,ref)

    def test_symmetry_and_immutability(self):
        ref,bc,blocks=self.fixture();copy=ref.copy();original=bc.copy()
        a,_=refine(ref,bc,blocks)
        swap=lambda x:np.where(x=='H','A',np.where(x=='A','H',x))
        b,_=refine(swap(ref),bc.rename(columns={'h__bc_same_distance_1':'a__bc_same_distance_1',
                                              'a__bc_same_distance_1':'h__bc_same_distance_1'}),blocks)
        np.testing.assert_array_equal(b,swap(a));np.testing.assert_array_equal(ref,copy)
        pd.testing.assert_frame_equal(bc,original)

    def test_overlapping_blocks_are_rejected(self):
        ref,bc,_=self.fixture()
        with self.assertRaises(ValueError):refine(ref,bc,[(0,90),(80,140)])


if __name__=='__main__':unittest.main()
