"""Input integrity contracts for the short-control correction."""
import unittest
import numpy as np
from tabpfn_short_control import restore_supported_islands


class ShortControlContractTests(unittest.TestCase):
    def data(self):
        ref=np.array(list("H"*20+"A"*40+"H"*20))
        alt=np.array(list("H"*80));support=np.ones((80,2),bool)
        return ref,alt,support

    def test_inputs_are_immutable(self):
        values=self.data();before=[v.copy() for v in values]
        restore_supported_islands(*values,[(0,80)])
        for a,b in zip(values,before):np.testing.assert_array_equal(a,b)

    def test_mismatched_dead_mask_is_rejected(self):
        ref,alt,support=self.data();alt[35]="D"
        with self.assertRaises(ValueError):restore_supported_islands(ref,alt,support,[(0,80)])

    def test_mismatched_unknown_mask_is_rejected(self):
        ref,alt,support=self.data();alt[35]="X"
        with self.assertRaises(ValueError):restore_supported_islands(ref,alt,support,[(0,80)])

    def test_bad_support_shape_is_rejected(self):
        ref,alt,support=self.data()
        with self.assertRaises(ValueError):restore_supported_islands(ref,alt,support[:20],[(0,80)])

    def test_overlapping_blocks_are_rejected(self):
        with self.assertRaises(ValueError):restore_supported_islands(*self.data(),[(0,60),(40,80)])


if __name__=="__main__":unittest.main()
