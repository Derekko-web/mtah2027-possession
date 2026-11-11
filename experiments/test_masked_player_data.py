"""Observation-only corruption, identity-independent windows and context controls."""
import unittest
import numpy as np
from masked_player_data import observation_views,assemble


class MaskedPlayerDataTests(unittest.TestCase):
    def test_view_replay_normalization_and_missingness(self):
        xy=np.zeros((101,4,2),np.float32);xy[15,0]=np.nan;teams=np.array([0,0,1,1]);pitch=np.array([105.,68.])
        first=observation_views(xy,teams,pitch,5)
        second=observation_views(xy,teams,pitch,5)
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
        values,mask,groups=first
        self.assertTrue(np.isfinite(values).all());self.assertFalse(mask[:,15,0].any())
        self.assertFalse(mask[:,:,4:].any());self.assertFalse((mask[3]&~mask[1]).any())
        np.testing.assert_array_equal(values[0,0,:4],np.zeros((4,2)))

    def test_origin_translation_has_identical_observation_semantics(self):
        xy=np.zeros((101,4,2),np.float32);teams=np.array([0,0,1,1]);pitch=np.array([105.,68.])
        a=observation_views(xy,teams,pitch,17,centered=True)
        b=observation_views(xy+pitch/2,teams,pitch,17,centered=False)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_half_boundaries_and_neighbor_information_control(self):
        xy=np.zeros((4,100,2,2),np.float32);mask=np.ones((4,100,2),bool);teams=np.tile([0,1],(100,1));halves=np.repeat([0,1],50)
        with self.assertRaises(ValueError):assemble(xy,mask,teams,halves,[49],[0])
        a=assemble(xy,mask,teams,halves,[25],[0],repeat=True)
        xy[:,10:25]=.8;xy[:,26:41]=-.9
        b=assemble(xy,mask,teams,halves,[25],[0],repeat=True)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertFalse(np.array_equal(assemble(xy,mask,teams,halves,[25],[0])[0],a[0]))
        xy[:,:10]=15;xy[:,41:]=16
        c=assemble(xy,mask,teams,halves,[25],[0])
        np.testing.assert_array_equal(c[0][:,15],np.zeros((1,2,2)))


if __name__=='__main__':unittest.main()
