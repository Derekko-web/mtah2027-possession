"""Leakage, symmetry and preservation tests for duration-balanced repairs."""
import unittest
import numpy as np
import pandas as pd
from short_possession_repair import training_record, repair


class RepairTests(unittest.TestCase):
    def fixture(self):
        n=160
        ref=np.array(list('H'*n))
        p=np.tile([.9,.1],(n,1));p[50:100]=[.1,.9]
        raw=pd.DataFrame({'h__distance_1':np.full(n,3.),'a__distance_1':np.full(n,1.)})
        return ref,p,p.copy(),raw,[(0,n)]

    def test_recovers_a_supported_missing_island(self):
        ref,p,o,raw,blocks=self.fixture()
        result,events=repair(ref,p,o,raw,blocks)
        self.assertTrue(np.all(result[55:95]=='A'))
        self.assertEqual(len(events),1)

    def test_no_physical_or_independent_support_rejects_proposal(self):
        ref,p,o,raw,blocks=self.fixture();o[:]=[.9,.1];raw['a__distance_1']=4.
        actual,events=repair(ref,p,o,raw,blocks)
        np.testing.assert_array_equal(actual,ref);self.assertEqual(events,[])

    def test_dead_and_unknown_frames_cannot_be_overwritten(self):
        for state in ('D','X'):
            ref,p,o,raw,blocks=self.fixture();ref[70]=state
            actual,_=repair(ref,p,o,raw,blocks)
            np.testing.assert_array_equal(actual,ref)

    def test_team_swap_symmetry_and_input_immutability(self):
        ref,p,o,raw,blocks=self.fixture();before=[x.copy() for x in (ref,p,o)]
        a,_=repair(ref,p,o,raw,blocks)
        swap=lambda x:np.where(x=='H','A',np.where(x=='A','H',x))
        swapped=raw.rename(columns={'h__distance_1':'a__distance_1','a__distance_1':'h__distance_1'})
        b,_=repair(swap(ref),p[:,::-1],o[:,::-1],swapped,blocks)
        np.testing.assert_array_equal(b,swap(a))
        for x,y in zip((ref,p,o),before):np.testing.assert_array_equal(x,y)

    def test_training_is_invariant_to_held_out_targets(self):
        labels=np.array(list('H'*20+'A'*20+'D'*10+'H'*50))
        train=np.arange(50);a=training_record(labels,train,1.)
        labels[50:]='X';b=training_record(labels,train,1.)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertAlmostEqual(a['weight'][:20].sum(),a['weight'][20:40].sum())

    def test_overlapping_blocks_rejected(self):
        ref,p,o,raw,_=self.fixture()
        with self.assertRaises(ValueError):repair(ref,p,o,raw,[(0,100),(90,160)])


if __name__=='__main__':unittest.main()
