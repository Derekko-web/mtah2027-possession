"""Verify physical features and segment/context weighting used by the candidate."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import relative_motion as rm
from build_candidate import objective_weights
from evaluate_physics_context import gap_context


class SelectedPipelineTests(unittest.TestCase):
    def test_relative_velocity_detects_a_player_moving_with_the_ball(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tracking.txt';lines=[]
            for frame in range(80):
                ball=50+frame*.04*2
                lines.append(f'{frame+1}:{(ball+.5)/105.3},0.5,2;{70/105.3},0.5,0:{ball/105.3},0.5,NaN\n')
            path.write_text(''.join(lines))
            with patch.object(rm,'require',return_value=['meta',str(path)]), \
                 patch.object(rm,'team_of_each_track',return_value=np.array([0,1])):
                features=rm.build_relative_motion(80)
            np.testing.assert_allclose(features.h__nearest_relative_velocity.iloc[5:],0,atol=1e-5)
            np.testing.assert_allclose(features.a__nearest_relative_velocity.iloc[5:],2,atol=1e-5)

    def test_segment_weight_totals_are_equal_and_swaps_keep_weights(self):
        target=np.array([0,0,1,1,1,2])
        frames=np.array([10,11,12,13,14,30])
        augmented,weights=objective_weights(target,frames,True)
        self.assertAlmostEqual(weights[:2].sum(),weights[2:5].sum())
        self.assertAlmostEqual(weights[:2].sum(),weights[5:6].sum())
        np.testing.assert_array_equal(weights[:6],weights[6:])
        np.testing.assert_array_equal(augmented[6:],[1,1,0,0,0,2])

    def test_occlusion_continuity_requires_active_players(self):
        inputs=pd.DataFrame({'ball_seen':np.ones(300), 'h__control_share':np.full(300,.3),
                             'h__speed_median':np.full(300,2.),'a__speed_median':np.full(300,2.)})
        inputs.loc[50:149,['ball_seen','h__control_share']]=[0,np.nan]
        np.testing.assert_array_equal(gap_context(inputs)[50:150],np.ones(100,dtype=int))
        inputs[['h__speed_median','a__speed_median']]=1.
        self.assertTrue((gap_context(inputs)<0).all())

    def test_long_gap_does_not_import_distant_context(self):
        inputs=pd.DataFrame({'ball_seen':np.ones(400), 'h__control_share':np.full(400,.3),
                             'h__speed_median':np.full(400,2.),'a__speed_median':np.full(400,2.)})
        inputs.loc[50:349,['ball_seen','h__control_share']]=[0,np.nan]
        self.assertTrue((gap_context(inputs)<0).all())


if __name__=='__main__':unittest.main()
