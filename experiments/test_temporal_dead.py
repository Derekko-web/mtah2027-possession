"""Synthetic invariants for sequence isolation and D-only integration."""
import unittest
import numpy as np
from temporal_dead import normalization,transform,windows,binary_targets,replace_dead_probability,dead_overlay,RADIUS


class TemporalDeadTests(unittest.TestCase):
    def test_preprocessing_uses_only_eligible_rows_and_retains_missingness(self):
        values=np.array([[1,np.nan],[3,np.nan],[1000,2]],float)
        mean,scale=normalization(values,[0,1]);a=transform(values,mean,scale)
        values[2]=[-9999,9999];m2,s2=normalization(values,[0,1])
        np.testing.assert_array_equal(mean,m2);np.testing.assert_array_equal(scale,s2)
        np.testing.assert_array_equal(a[:2,3],[1,1]);self.assertEqual(a[0,0],-1)

    def test_half_reset_and_exact_context_radius(self):
        values=np.c_[np.arange(100),np.zeros(100)].astype(np.float32)
        block=windows(values,[50],[(0,50),(50,100)])
        self.assertTrue((block[0,1,:RADIUS]==1).all())
        before=windows(values,[50],[(0,50),(50,100)])
        values[:19]=999;values[82:]=999
        np.testing.assert_array_equal(before,windows(values,[50],[(0,50),(50,100)]))

    def test_frame_control_has_no_neighbor_information(self):
        values=np.c_[np.arange(100),np.zeros(100)].astype(np.float32)
        actual=windows(values,[50],[(0,100)],repeat=True)
        values[:50]=-500;values[51:]=500
        np.testing.assert_array_equal(actual,windows(values,[50],[(0,100)],repeat=True))

    def test_targets_ignore_held_labels_and_enforce_inclusive_buffer(self):
        labels=np.array(['H','D','A']+['X']*997)
        a=binary_targets(labels,[0,1,2],[400])
        labels[300:700]='D';b=binary_targets(labels,[0,1,2],[400])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        labels[150]='H'
        with self.assertRaises(ValueError):binary_targets(labels,[0,1,150],[400])

    def test_dead_composition_preserves_conditional_owner(self):
        original=np.array([[.6,.3,.1],[.2,.5,.3]],float)
        result=replace_dead_probability(original,[.8],[0])
        self.assertAlmostEqual(result[0,0]/result[0,1],2)
        np.testing.assert_array_equal(result[1],original[1])
        np.testing.assert_allclose(result.sum(axis=1),1)

    def test_both_decoders_preserve_existing_live_team_and_unknown(self):
        reference=np.array(list('HHDDAX'));alternative=np.array(list('ADAHHX'))
        np.testing.assert_array_equal(dead_overlay(reference,alternative,'union'),np.array(list('HDDDAX')))
        np.testing.assert_array_equal(dead_overlay(reference,alternative,'trim'),np.array(list('HHAHAX')))


if __name__=='__main__':unittest.main()
