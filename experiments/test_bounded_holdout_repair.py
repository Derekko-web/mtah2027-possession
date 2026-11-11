"""Pitch exit must not consume a live prefix;block-edge control is eligible."""
import unittest
import numpy as np
import test_holdout_possession_repair as fixtures
from bounded_holdout_repair import refine


class BoundedHoldoutTests(unittest.TestCase):
    def test_late_exit_preserves_original_live_prefix_between_dead_runs(self):
        data=fixtures.HoldoutRepairTests().fixture('D'*30+'A'*100+'D'*50)
        data[4].loc[100:,'ball_goal_line_distance']=-1
        result,_=refine(*data,stages=('pitch',))
        self.assertTrue(np.all(result[30:102]=='A'));self.assertTrue(np.all(result[102:]=='D'))

    def test_already_outside_short_dead_gap_is_fully_suppressed(self):
        data=fixtures.HoldoutRepairTests().fixture('D'*30+'H'*50+'D'*50)
        data[4].loc[30:,'ball_goal_line_distance']=-1
        result,_=refine(*data,stages=('pitch',));self.assertTrue(np.all(result=='D'))

    def test_cross_source_trim_at_first_run_uses_intrinsic_owner_evidence(self):
        data=fixtures.HoldoutRepairTests().fixture('A'*70+'H'*80)
        data[4].loc[:69,['h__distance_1','a__distance_1']]=[4.,1.]
        data[5].loc[:69,['bc_matched','bc_same_ball_seen','h__bc_same_distance_1','a__bc_same_distance_1']]=[1,1,1,4]
        data[2][:70]=[.2,.8];data[2][:10]=[.8,.2]
        data[3][:70]=[.2,.8];data[3][50:70]=[.8,.2]
        result,_=refine(*data,stages=('cross',))
        self.assertTrue(np.all(result[:10]=='H'));self.assertTrue(np.all(result[10:50]=='A'))
        self.assertTrue(np.all(result[50:]=='H'))


if __name__=='__main__':unittest.main()
