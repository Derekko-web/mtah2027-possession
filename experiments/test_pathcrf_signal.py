"""Diagnostic proposals require temporal support and preserve D/X exactly."""
import unittest
import numpy as np
from audit_pathcrf_signal import overlay


class PathSignalTests(unittest.TestCase):
    def test_three_consecutive_cells_are_required_and_masks_are_protected(self):
        old=np.full(40,'H');old[12]='D';old[14]='X'
        frames=np.array([11190,11195,11200]);pred=np.array(['A','A','A'])
        two=overlay(old,frames,pred,np.array([True,True,False]));np.testing.assert_array_equal(two,old)
        result=overlay(old,frames,pred,np.ones(3,bool))
        self.assertEqual(result[12],'D');self.assertEqual(result[14],'X')
        self.assertTrue((result[3:10]=='A').all());self.assertEqual(result[0],'H')

    def test_missing_sampling_cell_breaks_a_proposal_run(self):
        old=np.full(40,'H');frames=np.array([11190,11195,11205]);pred=np.array(['A','A','A'])
        np.testing.assert_array_equal(overlay(old,frames,pred,np.ones(3,bool)),old)


if __name__=='__main__':unittest.main()
