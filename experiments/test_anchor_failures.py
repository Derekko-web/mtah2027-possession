"""Synthetic anchor provenance, conditional gates and inclusive-interval attribution."""
import unittest
import numpy as np
from broadcast_alignment import formation_anchors,map_anchors
from test_broadcast_alignment import synthetic_sources
from anchor_failures import (trace_anchors,trace_mapping,array_equal,ANCHOR_REASONS,
                             FRAME_REASONS,additional_consistency,attribution_summary)

PROFILES={'strict':dict(maximum_cost=2.5,minimum_margin=.5,maximum_fine_span=10),
          'relaxed':dict(maximum_cost=3.5,minimum_margin=.25,maximum_fine_span=15)}


def dummy_sources(n=200):
    rng=np.random.default_rng(101);teams=np.repeat([0,1],4)
    xy=rng.uniform([10,10],[90,58],(n,8,2))
    return dict(positions=xy,teams=teams,valid=np.ones(n,bool))


def fabricated(centers,source_frames,orientation=None,allowed=None):
    centers=np.array(centers);m=len(centers)
    coarse=np.c_[source_frames,np.zeros(m) if orientation is None else orientation,np.ones(m),np.ones(m)]
    fine=np.c_[source_frames,np.ones(m),np.zeros(m)]
    if allowed is not None:fine[~np.array(allowed,bool)]=np.nan
    anchors=dict(tactical_frames=centers,coarse=coarse,fine=fine)
    reason=np.where(np.isfinite(fine).all(axis=1),'allowed','strict_fine_geometry')
    return anchors,dict(reason=reason)


class AnchorFailureTests(unittest.TestCase):
    def test_joint_profile_trace_exactly_replays_original(self):
        tactical,broadcast,_=synthetic_sources()
        original=formation_anchors(tactical,broadcast,PROFILES)
        traced,trace=trace_anchors(tactical,broadcast,PROFILES)
        for key in original:self.assertTrue(array_equal(original[key],traced[key]),key)
        mapping,frames=trace_mapping(traced,trace,tactical,broadcast,PROFILES['strict'])
        expected=map_anchors(original,tactical,broadcast,PROFILES['strict'])
        for key in mapping:self.assertTrue(array_equal(mapping[key],expected[key]),key)
        self.assertEqual(sum((frames['reason']==r).sum() for r in FRAME_REASONS),len(tactical['positions']))

    def test_positive_infinite_margin_is_not_a_missing_neighborhood(self):
        tactical,_,_=synthetic_sources()
        broadcast=dict(positions=tactical['positions'].copy(),teams=tactical['teams'].copy(),valid=tactical['valid'].copy())
        anchors,trace=trace_anchors(tactical,broadcast,PROFILES)
        use=anchors['tactical_frames']==150
        self.assertTrue(np.isposinf(anchors['coarse'][use,3]).all())
        self.assertTrue(trace['coarse_neighbors_finite'][use].all())
        self.assertEqual(trace['reason'][use][0],'allowed')

    def test_shared_knot_considers_both_incident_intervals(self):
        source=dummy_sources()
        anchors,trace=fabricated([50,75,100],[50,75,100],allowed=[False,True,True])
        mapping,frames=trace_mapping(anchors,trace,source,source,PROFILES['strict'])
        self.assertEqual(frames['reason'][74],'no_allowed_endpoint_pair')
        self.assertEqual(frames['reason'][75],'mapped')
        self.assertEqual(mapping['frames'][75],75)
        self.assertTrue(frames['endpoint_reason_flags'][75].any()) # other incident interval rejected.

    def test_orientation_and_rate_are_separate_stages(self):
        source=dummy_sources()
        anchors,trace=fabricated([50,75,100],[50,75,115],orientation=[0,1,1])
        _,frames=trace_mapping(anchors,trace,source,source,PROFILES['strict'])
        self.assertEqual(frames['reason'][60],'no_common_orientation_after_allowed_pair')
        self.assertEqual(frames['reason'][90],'fine_rate_after_allowed_orientation')

    def test_final_broadcast_gate_does_not_silently_add_tactical_gate(self):
        tactical=dummy_sources();broadcast={k:v.copy() for k,v in tactical.items()}
        tactical['valid'][80]=False
        broadcast['valid'][81]=False
        anchors,trace=fabricated([50,75,100],[50,75,100])
        mapping,frames=trace_mapping(anchors,trace,tactical,broadcast,PROFILES['strict'])
        self.assertEqual(frames['reason'][80],'mapped')
        self.assertFalse(frames['tactical_query_usable'][80])
        self.assertEqual(mapping['frames'][81],-1)
        self.assertEqual(frames['reason'][81],'broadcast_source_unusable')

    def test_prefilter_respects_original_coarse_increment_limits(self):
        # Original synthetic cut ensures the trace follows all production gates,
        # including all missing fine rows, without assigning a later strict cause.
        tactical,broadcast,offset=synthetic_sources();broadcast['valid'][offset+125:offset+175]=False
        original=formation_anchors(tactical,broadcast,PROFILES)
        anchors,trace=trace_anchors(tactical,broadcast,PROFILES)
        self.assertTrue(array_equal(original['fine'],anchors['fine']))
        missing=~np.isfinite(anchors['fine']).all(axis=1)
        self.assertFalse(np.isin(trace['reason'][missing],['strict_fine_geometry','strict_fine_uncertainty_span','allowed']).any())

    def test_additional_checks_obey_context_guard_and_leave_candidates_unchanged(self):
        tactical,broadcast,_=synthetic_sources();anchors,trace=trace_anchors(tactical,broadcast,PROFILES)
        before={k:v.copy() for k,v in anchors.items()}
        plan=dict(additional_consistency=dict(unused_offsets=[-12,12],diagnostic_cutoff=2.5,quantiles=[0,.5,1],minimum_numeric_group=5),
                  reporting=dict(public_start=25,public_stop=325))
        table,summary=additional_consistency(anchors,trace,tactical,broadcast,[('half',0,349)],[],[],plan)
        self.assertGreater(len(table),0);self.assertTrue(summary['public_anchors_skipped_context_guard']>0)
        blocked,_=additional_consistency(anchors,trace,tactical,broadcast,[('half',0,349)],[dict(start=175,end=175)],[],plan)
        self.assertEqual(len(blocked),0)
        for key in before:self.assertTrue(array_equal(before[key],anchors[key]))

    def test_prefix_attribution_parity_on_stable_interior(self):
        tactical,broadcast,_=synthetic_sources();outputs=[]
        for n in (300,350):
            t={k:v[:n] if k in ('positions','valid') else v for k,v in tactical.items()}
            anchors,trace=trace_anchors(t,broadcast,PROFILES)
            mapping,frames=trace_mapping(anchors,trace,t,broadcast,PROFILES['strict'])
            outputs.append((anchors,trace,mapping,frames))
        for key in ('reason','endpoint_reason_flags','tactical_query_usable'):
            self.assertTrue(array_equal(outputs[0][3][key][50:225],outputs[1][3][key][50:225]),key)

    def test_overlap_counts_exclude_undefined_fine_gates(self):
        tactical,broadcast,offset=synthetic_sources();broadcast['valid'][offset+125:offset+175]=False
        anchors,trace=trace_anchors(tactical,broadcast,PROFILES)
        mapping,frames=trace_mapping(anchors,trace,tactical,broadcast,PROFILES['strict'])
        summary=attribution_summary(anchors,trace,mapping,frames,tactical,broadcast,dict(public_start=25,public_stop=325))
        self.assertEqual(summary['anchor_failed_predicates_overlapping']['fine_cost_strict']['evaluated'],int(trace['fine_finite'].sum()))
        self.assertEqual(sum(summary['anchor_first_failure'].values()),12)
        self.assertEqual(sum(summary['frame_first_failure'].values()),300)

    def test_no_broadcast_coarse_candidates_preserves_explicit_abort(self):
        tactical,broadcast,_=synthetic_sources();broadcast['valid'][:]=False
        with self.assertRaisesRegex(ValueError,'No eligible'):trace_anchors(tactical,broadcast,PROFILES)
        with self.assertRaisesRegex(ValueError,'No eligible'):formation_anchors(tactical,broadcast,PROFILES)


if __name__=='__main__':unittest.main()
