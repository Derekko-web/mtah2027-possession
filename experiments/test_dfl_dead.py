"""Synthetic checks for the new direct-status observation audit."""
import unittest
import numpy as np
from prepare_dfl_dead import summary


class DeadSupportTests(unittest.TestCase):
    def test_status_missingness_and_team_geometry_are_distinct(self):
        status=np.array([0,0,1,0,np.nan,1]);ball=np.zeros((6,2));ball[1]=np.nan
        teams=np.r_[np.zeros(4,int),np.ones(4,int)];positions=np.zeros((6,8,2));positions[3,0]=np.nan
        r=summary(status,ball,positions,teams)
        self.assertEqual(r['dead'],3);self.assertEqual(r['live'],2);self.assertEqual(r['status_valid'],5)
        self.assertEqual(r['dead_intervals'],2);self.assertEqual(r['dead_ball_xy'],2);self.assertEqual(r['dead_geometry'],2)

    def test_observation_change_does_not_relabel_dead_play(self):
        status=np.array([0,1,0]);teams=np.r_[np.zeros(4,int),np.ones(4,int)]
        positions=np.zeros((3,8,2));a=summary(status,np.zeros((3,2)),positions,teams)
        b=summary(status,np.full((3,2),np.nan),positions,teams)
        self.assertEqual(a['dead'],b['dead']);self.assertEqual(a['dead_intervals'],b['dead_intervals'])
        self.assertEqual(a['dead_ball_xy'],2);self.assertEqual(b['dead_ball_xy'],0)


if __name__=='__main__':unittest.main()
