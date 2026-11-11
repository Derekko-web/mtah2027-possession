"""Boundary evidence must preserve topology, missingness, symmetry and inputs."""
import unittest
import numpy as np
import pandas as pd
from possession_boundary_precision import refine,spans


class BoundaryPrecisionTests(unittest.TestCase):
    def fixture(self):
        n=220;ref=np.array(list('A'*50+'H'*70+'A'*100));p=np.tile([.5,.5],(n,1))
        raw=pd.DataFrame({'h__distance_1':np.full(n,4.),'a__distance_1':np.full(n,1.)})
        bc=pd.DataFrame({'h__bc_same_distance_1':np.full(n,4.),'a__bc_same_distance_1':np.full(n,1.),'bc_matched':np.ones(n),'bc_same_ball_seen':np.ones(n)})
        bc.loc[70:90,['h__bc_same_distance_1','a__bc_same_distance_1']]=[1.,4.]
        return ref,p,raw,bc,np.ones(n,bool),[(0,n)],[]

    def test_distinct_arrival_and_release_padding(self):
        data=self.fixture();actual,_=refine(*data,leading=13,trailing=5)
        self.assertTrue(np.all(actual[57:96]=='H'));self.assertEqual(spans(actual),[(0,57,'A'),(57,96,'H'),(96,220,'A')])

    def test_new_camera_approach_requires_proven_recovery_and_bilateral_evidence(self):
        data=list(self.fixture());data[0]=np.array(list('H'*100+'A'*50+'H'*70));data[4][:]=False
        data[1][80:86]=[.2,.8];data[2].loc[80:85,['h__distance_1','a__distance_1']]=[4.,1.]
        data[3].loc[80:85,['h__bc_same_distance_1','a__bc_same_distance_1']]=[4.,1.]
        data[6]=[dict(start=100,end_exclusive=150,team='A')]
        actual,_=refine(*data,approach=25);self.assertTrue(np.all(actual[80:150]=='A'))
        data[6]=[];actual,_=refine(*data,approach=25);np.testing.assert_array_equal(actual,data[0])

    def test_missing_camera_retains_reference(self):
        data=list(self.fixture());data[3].loc[85,'bc_matched']=0
        actual,_=refine(*data,leading=13,trailing=5);np.testing.assert_array_equal(actual,data[0])

    def test_no_contact_or_dead_flank_retains_reference(self):
        for condition in ('contact','dead'):
            data=list(self.fixture())
            if condition=='contact':data[3]['h__bc_same_distance_1']=5.
            else:data[0][:50]='D'
            actual,_=refine(*data,leading=13,trailing=5);np.testing.assert_array_equal(actual,data[0])

    def test_minimum_island_length_and_bounded_movement(self):
        data=list(self.fixture());data[3]['h__bc_same_distance_1']=4.;data[3].loc[84:86,'h__bc_same_distance_1']=.1
        actual,_=refine(*data,leading=5,trailing=5)
        self.assertEqual([s[2] for s in spans(actual)],[s[2] for s in spans(data[0])])
        self.assertGreaterEqual(spans(actual)[1][1]-spans(actual)[1][0],8)
        self.assertLessEqual(spans(actual)[1][0]-50,25);self.assertLessEqual(120-spans(actual)[1][1],25)

    def test_team_symmetry_and_immutability(self):
        ref,p,raw,bc,old,blocks,events=self.fixture();before=[ref.copy(),p.copy(),raw.copy(),bc.copy(),old.copy()]
        a,_=refine(ref,p,raw,bc,old,blocks,events,leading=13,trailing=5)
        swap=lambda x:np.where(x=='H','A',np.where(x=='A','H',x))
        reverse=lambda d:d.rename(columns={c:('a__'+c[3:] if c.startswith('h__') else 'h__'+c[3:] if c.startswith('a__') else c) for c in d})
        b,_=refine(swap(ref),p[:,::-1],reverse(raw),reverse(bc),old,blocks,events,leading=13,trailing=5)
        np.testing.assert_array_equal(b,swap(a))
        for x,y in zip((ref,p,raw,bc,old),before):
            if isinstance(x,pd.DataFrame):pd.testing.assert_frame_equal(x,y)
            else:np.testing.assert_array_equal(x,y)

    def test_overlap_and_invalid_probability_rejected(self):
        data=list(self.fixture());data[5]=[(0,150),(140,220)]
        with self.assertRaises(ValueError):refine(*data,leading=13,trailing=5)
        data=list(self.fixture());data[1][60]=np.nan
        with self.assertRaises(ValueError):refine(*data,leading=13,trailing=5)


if __name__=='__main__':unittest.main()
