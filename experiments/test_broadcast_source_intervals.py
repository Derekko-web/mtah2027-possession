"""Source intervals require physical/model agreement and protect brief references."""

import unittest

import numpy as np
import pandas as pd

from broadcast_source_intervals import source_intervals


def features(owners):
    owners = np.asarray(owners)
    h,a = np.where(owners == 0,1.,4.),np.where(owners == 1,1.,4.)
    return pd.DataFrame(dict(bc_ball_seen=np.ones(len(owners)),h__bc_same_distance_1=h,a__bc_same_distance_1=a,
                              h__bc_same_control_share=(a+.1)/(h+a+.2),a__bc_same_control_share=(h+.1)/(h+a+.2)))


class BroadcastSourceIntervalTests(unittest.TestCase):
    def test_new_supported_island_requires_learned_agreement(self):
        reference = np.ones(600,int)
        owner = reference.copy(); owner[200:240] = 0
        x = features(owner)
        np.testing.assert_array_equal(source_intervals(reference,reference,x,2),reference)
        result = source_intervals(reference,owner,x,2)
        self.assertGreater((result == 0).sum(),30)
        np.testing.assert_array_equal(source_intervals(1-reference,1-owner,features(1-owner),2),1-result)

    def test_source_cannot_punch_through_brief_reference_island(self):
        reference = np.ones(600,int); reference[100:242] = 0
        owner = np.zeros(600,int); owner[140:180] = 1
        learned = reference.copy(); learned[140:180] = 1
        result,records = source_intervals(reference,learned,features(owner),5,True)
        np.testing.assert_array_equal(result,reference)
        self.assertIn("protected_reference_island",[r["reason"] for r in records])

    def test_dead_and_missing_stretches_remain_reference(self):
        reference = np.ones(600,int)
        owner = reference.copy(); owner[200:240] = 0
        x = features(owner)
        reference[220] = 2
        result = source_intervals(reference,owner,x,2)
        np.testing.assert_array_equal(result,reference)
        reference[220] = 1
        x.loc[215:224,"bc_ball_seen"] = 0
        result = source_intervals(reference,owner,x,2)
        self.assertTrue(np.all(result[215:225] == 1))
        np.testing.assert_array_equal(result == 2,reference == 2)


if __name__ == "__main__": unittest.main()
