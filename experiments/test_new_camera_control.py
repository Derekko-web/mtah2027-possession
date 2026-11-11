"""Tests for new-camera evidence, stable release, symmetry and boundaries."""
import unittest
import numpy as np
import pandas as pd
from new_camera_control import recover


class NewCameraTests(unittest.TestCase):
    def fixture(self):
        n=400;ref=np.array(list('H'*n));p=np.tile([.9,.1],(n,1));p[100:106]=[.4,.6]
        raw=pd.DataFrame({'h__distance_1':np.full(n,1.),'a__distance_1':np.full(n,4.)})
        raw.loc[100:105,['h__distance_1','a__distance_1']]=[4.,1.]
        motion=pd.DataFrame({'h__tc_kernel':np.full(n,.9),'a__tc_kernel':np.full(n,.1)})
        motion.loc[95:150,'h__tc_kernel']=.1
        bc=raw.rename(columns={'h__distance_1':'h__bc_same_distance_1','a__distance_1':'a__bc_same_distance_1'}).copy()
        bc['bc_matched']=1.;bc['bc_same_ball_seen']=1.
        return ref,p,raw,motion,bc,np.zeros(n,bool),[(0,n)]

    def test_bilateral_contact_recovers_control_until_confirmed_release(self):
        data=self.fixture();actual,events=recover(*data)
        self.assertTrue(np.all(actual[100:164]=='A'));self.assertEqual(len(events),1)
        self.assertEqual(events[0]['broadcast_trigger_contacts'],6)

    def test_existing_camera_coverage_cannot_trigger_new_repair(self):
        data=list(self.fixture());data[5][:]=True
        actual,events=recover(*data);np.testing.assert_array_equal(actual,data[0]);self.assertEqual(events,[])

    def test_missing_second_camera_contact_refuses_tactical_only_trigger(self):
        data=list(self.fixture());data[4]['a__bc_same_distance_1']=5.
        actual,events=recover(*data);np.testing.assert_array_equal(actual,data[0]);self.assertEqual(events,[])

    def test_missing_camera_inside_episode_refuses_carry(self):
        data=list(self.fixture());data[4].loc[130,'bc_same_ball_seen']=0
        actual,_=recover(*data);np.testing.assert_array_equal(actual,data[0])

    def test_no_confirmed_release_produces_no_timeout_guess(self):
        data=list(self.fixture());data[3]['h__tc_kernel']=.1
        actual,_=recover(*data);np.testing.assert_array_equal(actual,data[0])

    def test_short_existing_spells_and_dead_unknown_boundaries_are_protected(self):
        for state in ('A','D','X'):
            data=list(self.fixture());data[0][180:]=state
            actual,_=recover(*data);np.testing.assert_array_equal(actual,data[0])

    def test_team_symmetry_and_input_immutability(self):
        data=self.fixture();ref,p,raw,m,bc,old,blocks=data;actual,_=recover(*data)
        before=[ref.copy(),p.copy(),raw.copy(),m.copy(),bc.copy(),old.copy()]
        swap=lambda a:np.where(a=='H','A',np.where(a=='A','H',a))
        reverse=lambda frame:frame.rename(columns={c:('a__'+c[3:] if c.startswith('h__') else 'h__'+c[3:] if c.startswith('a__') else c) for c in frame})
        other,_=recover(swap(ref),p[:,::-1],reverse(raw),reverse(m),reverse(bc),old,blocks)
        np.testing.assert_array_equal(other,swap(actual))
        for a,b in zip((ref,p,raw,m,bc,old),before):
            if isinstance(a,pd.DataFrame):pd.testing.assert_frame_equal(a,b)
            else:np.testing.assert_array_equal(a,b)

    def test_overlapping_blocks_and_invalid_probabilities_rejected(self):
        data=list(self.fixture());data[-1]=[(0,250),(240,400)]
        with self.assertRaises(ValueError):recover(*data)
        data=list(self.fixture());data[1][100]=np.nan
        with self.assertRaises(ValueError):recover(*data)


if __name__=='__main__':unittest.main()
