"""Meaningful protection and supervision-mass checks for D-only adaptation."""
import unittest
import numpy as np
from research_dfl_dead_public import preserve_owners,fit_arrays
from research_dfl_dead_external import metrics,gates


class DeadIntegrationTests(unittest.TestCase):
    def test_ownership_is_protected_while_both_D_directions_are_available(self):
        old=np.array(list('HADDXH'));alternative=np.array(list('AHHADX'))
        np.testing.assert_array_equal(preserve_owners(old,alternative),np.array(list('HAHAXH')))
        np.testing.assert_array_equal(preserve_owners(old,old),old)

    def test_training_mass_and_held_label_independence(self):
        labels=np.array(list('HADDHAXX'));train=np.arange(4);weak_frames=np.array([6,7]);weak_target=np.array([0,1])
        y,w=fit_arrays(labels,train,weak_frames,weak_target)
        self.assertAlmostEqual(w[:4].sum(),4);self.assertAlmostEqual(w[4:].sum(),2)
        other=labels.copy();other[4:6]='D';yy,ww=fit_arrays(other,train,weak_frames,weak_target)
        np.testing.assert_array_equal(y,yy);np.testing.assert_array_equal(w,ww)
        with self.assertRaises(ValueError):fit_arrays(labels,train,np.array([0,7]),weak_target)

    def test_external_metrics_account_for_visible_and_hidden_stoppages(self):
        status=np.array([0,0,1,1]);p=np.array([.9,.2,.8,.1]);seen=np.array([True,False,True,False])
        m=metrics(status,p,seen)
        self.assertEqual(m['dead_precision'],.5);self.assertEqual(m['dead_recall'],.5)
        self.assertEqual(m['visible_dead_recall'],1);self.assertEqual(m['hidden_dead_recall'],0)
        self.assertEqual(m['live_false_dead_rate'],.5)


if __name__=='__main__':unittest.main()
