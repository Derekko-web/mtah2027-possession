"""Behavioral safeguards for learned short proposals and training exclusions."""

from unittest import TestCase,main
from unittest.mock import patch

import numpy as np
import pandas as pd

from consensus_control import consensus_team,decode_consensus
from control_refinement import contact_support,control_evidence,fit_control
from rich_features import swap_teams


def scene(n=250):
    return pd.DataFrame(dict(ball_seen=np.ones(n),ball_displacement_1s=np.full(n,6.),
                             h__distance_1=np.full(n,5.),a__distance_1=np.full(n,.5),
                             h__control_share=np.full(n,.1),a__control_share=np.full(n,.9),h__speed_median=np.full(n,2.),
                             a__speed_median=np.full(n,2.)))


class ConsensusTests(TestCase):
    def test_recovers_supported_brief_proposal_without_changing_other_ownership(self):
        current=np.tile([.09,.9,.01],(250,1));previous=current.copy()
        previous[100:120]=[.9,.09,.01]
        fast=np.tile([.1,.9],(250,1));fast[100:120]=[.8,.2]
        support=np.zeros((250,2),dtype=bool);support[103:115,0]=True
        owner=consensus_team(current,previous,fast,support)
        np.testing.assert_allclose(owner[100:120],fast[100:120])
        np.testing.assert_allclose(owner[:100],current[:100,:2]/.99)
        np.testing.assert_allclose(owner[120:],current[120:,:2]/.99)
        np.testing.assert_array_equal(current,np.tile([.09,.9,.01],(250,1)))

    def test_fast_model_alone_cannot_create_an_unsupported_possession(self):
        current=np.tile([.09,.9,.01],(250,1));previous=current.copy()
        previous[100:120]=[.9,.09,.01];fast=np.tile([.8,.2],(250,1))
        actual=consensus_team(current,previous,fast,np.zeros((250,2),dtype=bool))
        np.testing.assert_allclose(actual,current[:,:2]/.99)
        previous[100:120]=current[100:120]
        actual=consensus_team(current,previous,fast,np.ones((250,2),dtype=bool))
        np.testing.assert_allclose(actual,current[:,:2]/.99)

    def test_long_or_block_clipped_proposals_are_not_resurrected(self):
        current=np.tile([.09,.9,.01],(300,1));fast=np.tile([.8,.2],(300,1))
        for start,end in ((50,200),(0,20),(280,300)):
            previous=current.copy();previous[start:end]=[.9,.09,.01]
            actual=consensus_team(current,previous,fast,np.ones((300,2),dtype=bool))
            np.testing.assert_allclose(actual,current[:,:2]/.99)

    def test_missing_observations_never_supply_contact_evidence(self):
        raw=scene();raw.loc[:,['h__distance_1','a__distance_1']]=np.nan
        self.assertFalse(contact_support(raw).any())
        self.assertFalse(control_evidence(raw).any())
        raw=scene();raw['ball_displacement_1s']=0
        self.assertFalse(control_evidence(raw).any())

    def test_full_decoder_is_team_symmetric_and_preserves_abstention_gaps(self):
        raw=scene();raw.loc[100:119,['h__distance_1','a__distance_1']]=[.5,5]
        current=np.tile([.09,.9,.01],(250,1));previous=np.tile([.45,.54,.01],(250,1))
        previous[95:125]=[.99,0,.01];fast=np.tile([.1,.9],(250,1));fast[75:145]=[.85,.15]
        blocks=[(20,200)]
        a=decode_consensus(current,previous,fast,raw,blocks)
        b=decode_consensus(current[:,[1,0,2]],previous[:,[1,0,2]],fast[:,[1,0]],swap_teams(raw),blocks)
        expected=np.where(a=='H','A',np.where(a=='A','H',a))
        np.testing.assert_array_equal(b,expected)
        self.assertTrue(np.all(a[:20]=='X'));self.assertTrue(np.all(a[200:]=='X'))
        self.assertTrue(np.all(a[100:120]=='H'))

    def test_valid_pure_dead_probability_does_not_create_nan_emissions(self):
        raw=scene();raw['ball_displacement_1s']=0
        dead=np.tile([0.,0.,1.],(250,1));fast=np.tile([.5,.5],(250,1))
        result=decode_consensus(dead,dead,fast,raw,[(0,250)])
        self.assertTrue(np.all(result=='D'))

    def test_live_head_targets_ignore_dead_and_held_out_labels(self):
        x=pd.DataFrame(dict(h__x=np.arange(12),a__x=-np.arange(12)))
        labels=np.array(list('HADHHADHADHA'));calls=[]
        class Model:
            def __init__(self,**kwargs):pass
            def fit(self,features,target):calls.append((features.copy(),target.copy()))
        with patch('control_refinement.XGBClassifier',Model):
            fit_control(x,labels,np.arange(6))
            labels[6:]='X'
            fit_control(x,labels,np.arange(6))
        pd.testing.assert_frame_equal(calls[0][0],calls[1][0]);np.testing.assert_array_equal(calls[0][1],calls[1][1])
        self.assertEqual(len(calls[0][0]),10)

    def test_invalid_blocks_and_probabilities_are_rejected(self):
        raw=scene();p=np.tile([.2,.7,.1],(250,1));fast=np.tile([.3,.7],(250,1))
        with self.assertRaises(ValueError):decode_consensus(p,p,fast,raw,[(0,100),(90,200)])
        fast[40]=np.nan
        with self.assertRaises(ValueError):decode_consensus(p,p,fast,raw,[(0,100)])


if __name__=='__main__':main()
