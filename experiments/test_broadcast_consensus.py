"""Coverage and source-mask invariants for the second-camera research pipeline."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from broadcast_consensus import decode_broadcast_consensus,short_broadcast_proposals
from broadcast_features import aligned_features
from test_broadcast_alignment import synthetic_sources


class BroadcastConsensusTests(unittest.TestCase):
    def test_replacement_preserves_original_head_where_broadcast_is_unseen(self):
        n = 20
        p = np.tile([.6,.3,.1],(n,1))
        fast = np.tile([.7,.3],(n,1))
        heads = np.full((n,2),.1)
        features = pd.DataFrame(dict(bc_ball_seen=np.r_[np.zeros(10),np.ones(10)]))
        for joint in (False,True):
            probability = np.tile([.1,.8,.1] if joint else [.1,.9],(n,1))
            with patch("broadcast_consensus.decode_stoppage_consensus",return_value=np.full(n,"H")) as decoder:
                decode_broadcast_consensus(p,p,fast,heads,probability,features,pd.DataFrame(index=range(n)),[(0,n)],joint,"replace_head")
                current,_,new_fast = decoder.call_args.args[:3]
                np.testing.assert_array_equal(current[:10],p[:10])
                np.testing.assert_array_equal(new_fast[:10],fast[:10])
                np.testing.assert_array_equal(current[10:] if joint else new_fast[10:],probability[10:])
        np.testing.assert_array_equal(p,np.tile([.6,.3,.1],(n,1)))
        np.testing.assert_array_equal(fast,np.tile([.7,.3],(n,1)))

    def test_proposal_requires_independent_contact_coverage_and_no_dead(self):
        n = 150
        reference = np.zeros(n,int)
        p = np.tile([.99,.01],(n,1)); p[55:85] = [.01,.99]
        support = np.zeros((n,2),bool); support[55:85,1] = True
        seen = np.zeros(n,bool)
        np.testing.assert_array_equal(short_broadcast_proposals(reference,p,support,seen),reference)
        seen[55:85] = True
        result = short_broadcast_proposals(reference,p,support,seen)
        self.assertGreater((result == 1).sum(),20)
        np.testing.assert_array_equal(short_broadcast_proposals(1-reference,p[:,::-1],support[:,::-1],seen),1-result)
        reference[70] = 2
        np.testing.assert_array_equal(short_broadcast_proposals(reference,p,support,seen),reference)

    def test_feature_team_symmetry_and_bounded_summary_padding(self):
        t,b,offset = synthetic_sources()
        n = len(t["positions"])
        t["speeds"],b["speeds"] = np.ones((n,12)),np.ones((len(b["positions"]),12))
        t["ball"] = t["positions"][:,0].copy()
        b["ball"] = np.full((len(b["positions"]),2),np.nan)
        from broadcast_alignment import transform
        b["ball"][offset:offset+n] = transform(t["ball"],3)
        mapping = dict(frames=np.arange(n)+offset,orientation=np.full(n,3),cost=np.full(n,.2),margin=np.ones(n))
        full = aligned_features(t,b,mapping)
        swapped_t,swapped_b = dict(t,teams=1-t["teams"]),dict(b,teams=1-b["teams"])
        swapped = aligned_features(swapped_t,swapped_b,mapping)
        prefix_t = {k:v[:300] if k in ("positions","speeds","ball","valid") else v for k,v in t.items()}
        prefix = aligned_features(prefix_t,b,{k:v[:300] for k,v in mapping.items()})
        for original,flipped,shorter in zip(full,swapped,prefix):
            pd.testing.assert_frame_equal(original.iloc[125:175],shorter.iloc[125:175])
            for name in original:
                other = ("a__" if name.startswith("h__") else "h__")+name[3:] if name.startswith(("h__","a__")) else name
                np.testing.assert_array_equal(original[name],flipped[other])


if __name__ == "__main__": unittest.main()
