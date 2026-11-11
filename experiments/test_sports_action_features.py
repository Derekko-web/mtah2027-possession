from pathlib import Path
import tempfile
import unittest
import numpy as np
from sports_action_features import dense_actions,actor_row
from sports_action_sampling import clip_frames


class SportsActionFeatureTests(unittest.TestCase):
    def test_nearest_center_tie_and_missing_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            records=[]
            for center,logit in ((12000,2.),(12050,4.),(12200,6.)):
                values=np.zeros((100,31),np.float32)
                values[:,1]=logit
                name=f'{center}.npz'
                np.savez(Path(folder)/name,frames=clip_frames(center),im_feat=values,
                         team_feat=np.zeros(100),displ_feat=np.zeros(100))
                records.append(dict(center=center,file=name))
            dense=dense_actions(folder,records,12400)
            expected=np.exp(2)/(np.exp(2)+12)
            self.assertAlmostEqual(float(dense.sa_pass.iloc[12025]),expected,places=6)
            self.assertEqual(dense.sa_seen.iloc[12125],0)
            self.assertTrue(np.isnan(dense.sa_pass.iloc[12125]))
            self.assertEqual(dense.sa_seen.iloc[12001],1)

    def test_actor_mapping_exchanges_teams_and_rejects_insufficient_geometry(self):
        positions=np.array([[20,10],[21,12],[22,14],[70,10],[71,12],[72,14]],float)
        teams=np.array([0,0,0,1,1,1])
        matrix=np.diag([1920.,1080.,1.])
        first=actor_row(positions,teams,matrix,.8)
        second=actor_row(positions,1-teams,matrix,.8)
        np.testing.assert_allclose(first[0],second[0][::-1])
        np.testing.assert_allclose(first[1],second[1][::-1])
        np.testing.assert_allclose(first[1],[.2,.8])
        self.assertIsNone(actor_row(positions[:5],teams[:5],matrix,.8))


if __name__=='__main__':
    unittest.main()
