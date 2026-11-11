"""Synthetic contracts for local decisions, cuts, exclusions and prefix summaries."""
import unittest
import numpy as np
import pandas as pd
from alignment_coverage import inverse_schedule, query_plan, synthetic_feature_view, opportunity_gate, coverage_diagnostics
from broadcast_action_sampling import schedule


def source(n=19060):
    return dict(frames=np.arange(n)+3000, orientation=np.zeros(n, int))


def eligible(mapping, grid=50, n=19060, reserved=None, held=None, usable=None):
    return inverse_schedule(n, mapping, [('half',11185,30000)], reserved or [],
                            np.array([] if held is None else held),
                            np.ones(164487, bool) if usable is None else usable, grid)


class AlignmentCoverageTests(unittest.TestCase):
    def test_exact_content_survives_a_gap_outside_its_source_span(self):
        mapping=source();mapping['frames'][12115]=-1;mapping['orientation'][12115]=-1
        rows,_=eligible(mapping)
        target=next(r for r in rows if r['tactical_center']==12000)
        self.assertEqual((target['image_left'],target['image_right']),(11900,12098))
        old=schedule(19060,mapping,[('half',11185,30000)],[],[],broadcast_n=164487)
        self.assertNotIn(12000,[r['tactical_center'] for r in old])

    def test_source_crossing_gap_orientation_or_jump_is_rejected(self):
        for mode in ('gap','orientation','jump'):
            mapping=source()
            if mode=='gap':mapping['frames'][12050]=-1
            elif mode=='orientation':mapping['orientation'][12050]=1
            else:mapping['frames'][12050:]+=30
            rows,_=eligible(mapping)
            self.assertNotIn(12000,[r['tactical_center'] for r in rows],mode)

    def test_intervening_unsampled_source_frame_is_guarded(self):
        mapping=source();usable=np.ones(164487,bool);usable[15001]=False
        rows,counts=eligible(mapping,usable=usable)
        self.assertNotIn(12000,[r['tactical_center'] for r in rows])
        self.assertGreater(counts['source_unusable'],0)

    def test_complete_metadata_interval_excludes_sealed_and_held(self):
        rows,_=eligible(source(),reserved=[dict(start=13000,end=13010)],held=[15000])
        self.assertTrue(rows)
        for row in rows:
            a,b=row['decision_left'],row['decision_right']
            self.assertGreaterEqual(a,11185);self.assertLess(b,19060)
            self.assertFalse(a<=13260 and b>=12750)
            self.assertFalse(a<=15250 and b>=14750)
            self.assertLessEqual(row['tactical_center']-row['image_left'],125)
            self.assertLessEqual(row['image_right']-row['tactical_center'],125)

    def test_local_eligibility_does_not_scan_distant_component_endpoints(self):
        original=source();changed=source()
        outside=np.abs(np.arange(19060)-12000)>125
        changed['frames'][outside]=-1;changed['orientation'][outside]=-1
        first,_=eligible(original);second,_=eligible(changed)
        self.assertEqual(next(r for r in first if r['tactical_center']==12000),
                         next(r for r in second if r['tactical_center']==12000))

    def test_query_uses_an_eligible_alternative_and_keeps_earlier_ties(self):
        mapping=dict(frames=np.arange(300)+1000,orientation=np.zeros(300,int))
        rows=[dict(tactical_center=100,broadcast_center=1100,component_left=75,component_right=120),
              dict(tactical_center=125,broadcast_center=1125,component_left=121,component_right=150),
              dict(tactical_center=150,broadcast_center=1150,component_left=121,component_right=180)]
        queries=query_plan(rows,mapping,300)
        self.assertEqual(queries['tactical_center'][120],100)
        same=[dict(tactical_center=100,broadcast_center=1100,component_left=0,component_right=299),
              dict(tactical_center=150,broadcast_center=1150,component_left=0,component_right=299)]
        self.assertEqual(query_plan(same,mapping,300)['tactical_center'][125],100)
        self.assertFalse(queries['seen'][200])

    def test_actual_future_synthetic_features_are_prefix_equal(self):
        mapping=source();views=[];queries=[]
        for n in (18935,19060):
            rows,_=eligible(mapping,grid=25,n=n)
            q=query_plan(rows,mapping,n);queries.append(q)
            views.append(synthetic_feature_view(q,mapping,n,[('half',11185,30000)]).iloc[11185:18685])
        for key in queries[0]:
            np.testing.assert_array_equal(queries[0][key][11135:18735],queries[1][key][11135:18735])
        pd.testing.assert_frame_equal(*views,check_exact=True)
        self.assertEqual(len(views[0].columns),121)

    def test_grid_refinement_cannot_remove_query_support(self):
        mapping=source();mapping['frames'][12320:12335]=-1
        sparse,_=eligible(mapping,grid=50);dense,_=eligible(mapping,grid=25)
        sparse=query_plan(sparse,mapping,19060);dense=query_plan(dense,mapping,19060)
        self.assertFalse((sparse['seen'] & ~dense['seen']).any())

    def test_frozen_opportunity_gate_requires_every_condition(self):
        rule=dict(min_observed_frames=3000,min_net_additional_frames=750,min_original_coverage_retention=.98)
        result=dict(observed_public_frames=3100,net_additional_public_frames=900,original_coverage_retention=.99)
        self.assertTrue(opportunity_gate(result,rule,True)['passed'])
        for key,value in [('observed_public_frames',2999),('net_additional_public_frames',749),('original_coverage_retention',.979)]:
            self.assertFalse(opportunity_gate(dict(result,**{key:value}),rule,True)['passed'])
        self.assertFalse(opportunity_gate(result,rule,False)['passed'])

    def test_fixed_windows_are_label_free_and_duplicate_sources_count_once(self):
        q=dict(seen=np.ones(300,bool));before=dict(seen=np.zeros(300,bool));before['seen'][:150]=True
        result=coverage_diagnostics(q,before,[dict(broadcast_center=10),dict(broadcast_center=10)],{},0,300)
        self.assertEqual(result['unique_source_centers'],1)
        self.assertEqual(result['additional_public_frames'],150)
        for window in result['fixed_label_free_windows']:
            self.assertEqual(window['coverage_ge80'],window['windows'])


if __name__=='__main__':unittest.main()
