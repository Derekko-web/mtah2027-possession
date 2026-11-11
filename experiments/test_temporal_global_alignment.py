"""Known correspondences, aliases, missingness and complete dependency contracts."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from broadcast_alignment import formation_anchors, frame_usable, transform, map_anchors
from test_broadcast_alignment import synthetic_sources
from test_anchor_failures import PROFILES
from temporal_global_alignment import (temporal_search, temporal_costs, temporal_anchors,
    neighborhood_reason, inverse_schedule, MAPPING_CONTEXT, INVERSE_RADIUS, DECISION_RADIUS,
    safe_interval, extension, FIELDS, metadata_gate)
from alignment_coverage import inverse_schedule as historical_schedule, query_plan, synthetic_feature_view


def mapping(n=19060, rate=1.):
    return dict(frames=np.rint(np.arange(n)*rate).astype(int)+3000, orientation=np.zeros(n, int))


def schedule(m, n=19060, usable=None, reserved=None, held=None):
    return inverse_schedule(n, m, [('half',11185,30000)], reserved or [],
        np.array([] if held is None else held), np.ones(164487, bool) if usable is None else usable)


class TemporalGlobalTests(unittest.TestCase):
    def test_known_offset_reflection_permutation(self):
        t,b,offset=synthetic_sources()
        _,_,out=temporal_anchors(t,b,[('half',0,349)],[],np.array([]))
        use=out['frames']>=0
        self.assertGreater(int(use.sum()),150)
        np.testing.assert_array_equal(out['frames'][use],np.flatnonzero(use)+offset)
        self.assertTrue((out['orientation'][use]==3).all())

    def test_temporal_objective_replaces_false_spatial_winner(self):
        t,b,offset=synthetic_sources();b['teams']=t['teams'];b['positions'][offset:offset+350]=transform(t['positions'],3)
        b['positions'][275]+=0.2
        b['positions'][575]=transform(t['positions'][150],3)
        old=formation_anchors(t,b,PROFILES)
        row=old['coarse'][old['tactical_frames']==150][0]
        self.assertEqual(row[0],575)
        ut,ub=frame_usable(t),frame_usable(b)
        bf=np.arange(25,len(ub)-25,25)
        result,_,_,_=temporal_search(150,t,b,bf,ut,ub)
        self.assertEqual(result[0],275);self.assertEqual(result[1],3)

    def test_exact_far_temporal_copy_is_ambiguous(self):
        t,b,offset=synthetic_sources()
        b['positions']=np.concatenate([b['positions'],b['positions'][offset:offset+350]])
        b['valid']=np.ones(len(b['positions']),bool)
        _,_,out=temporal_anchors(t,b,[('half',0,349)],[],np.array([]))
        self.assertFalse((out['frames']>=0).any())

    def test_runner_uses_new_winner_all_reflections_and_strict_250(self):
        t,b,_=synthetic_sources();bf=np.array([25,275,300])
        def fake(frame,candidates,o,*args):
            score=np.array([0,.1,1.]) if o==0 else np.array([9,9,.4]) if o==2 else np.ones(3)*9
            return score,np.repeat(score[:,None],3,axis=1)
        with patch('temporal_global_alignment.temporal_costs',side_effect=fake):
            row,runner,_,_=temporal_search(100,t,b,bf,frame_usable(t),frame_usable(b))
        self.assertEqual(row[3],.4);self.assertEqual(tuple(runner[:2]),(300,2))

    def test_no_far_runner_infinite_margin_and_deterministic_ties(self):
        t,b,_=synthetic_sources();bf=np.array([25,50,75])
        with patch('temporal_global_alignment.temporal_costs',return_value=(np.zeros(3),np.zeros((3,3)))):
            row,runner,_,_=temporal_search(100,t,b,bf,frame_usable(t),frame_usable(b))
        self.assertEqual(tuple(row[:2]),(25,0));self.assertTrue(np.isposinf(row[3]));self.assertTrue(np.isnan(runner[0]))

    def test_all_infinite_is_missing_not_a_fake_winner(self):
        t,b,_=synthetic_sources();ut,ub=frame_usable(t),frame_usable(b);ut[75]=False
        row,_,_,count=temporal_search(100,t,b,np.array([25,50]),ut,ub)
        self.assertTrue(np.isnan(row).all());self.assertEqual(count,0)

    def test_full_triplet_usability_and_source_bounds(self):
        t,b,_=synthetic_sources();ut,ub=frame_usable(t),frame_usable(b);ub[300]=False
        score,_=temporal_costs(150,np.array([0,25,275,625]),3,t,b,ut,ub)
        self.assertTrue(np.isinf(score[[0,2,3]]).all());self.assertTrue(np.isfinite(score[1]))

    def test_average_cannot_hide_bad_individual_offset(self):
        row=np.array([[100,0,2.,1.],[125,0,2.,np.inf],[150,0,2.,1.]])
        individual=np.ones((3,3));individual[0,0]=3.
        self.assertLess(np.sqrt(np.mean(individual[0]**2)),2.5)
        self.assertEqual(neighborhood_reason(row,individual),'neighbor_individual_cost')

    def test_neighborhood_increments_margins_orientation(self):
        row=np.array([[100,0,1.,.5],[125,0,1.,np.inf],[150,0,1.,.5]]);individual=np.ones((3,3))
        self.assertEqual(neighborhood_reason(row,individual),'pass')
        for column,value,reason in [(2,2.51,'neighbor_temporal_cost'),(3,.49,'neighbor_temporal_margin'),(1,1,'orientation'),(0,176,'coarse_increment')]:
            changed=row.copy();changed[2,column]=value
            self.assertEqual(neighborhood_reason(changed,individual),reason)
        for centers in ([100,100,150],[100,150,150]):
            changed=row.copy();changed[:,0]=centers
            self.assertEqual(neighborhood_reason(changed,individual),'pass')

    def test_excluded_triplets_never_evaluate_geometry(self):
        t,b,_=synthetic_sources()
        with patch('temporal_global_alignment.temporal_search',side_effect=AssertionError('excluded geometry')):
            _,trace,out=temporal_anchors(t,b,[('half',0,349)],[dict(start=175,end=175)],np.array([]))
        self.assertFalse(trace['safe'].any());self.assertFalse((out['frames']>=0).any())

    def test_mapping_dependency_envelope_including_shared_knots(self):
        distances=[]
        for f in range(100,125):
            intervals=[(100,125)] if f>100 else [(75,100),(100,125)]
            dependencies={endpoint+neighbor+offset for interval in intervals for endpoint in interval
                          for neighbor in (-25,0,25) for offset in (-25,0,25)}
            distances.extend(abs(d-f) for d in dependencies)
        self.assertEqual(max(distances),MAPPING_CONTEXT)
        self.assertEqual(INVERSE_RADIUS+MAPPING_CONTEXT,DECISION_RADIUS)
        self.assertEqual(INVERSE_RADIUS+MAPPING_CONTEXT+25+50,250)

    def test_changing_geometry_outside_75_cannot_change_row_or_absence(self):
        t,b,_=synthetic_sources();changed=copy.deepcopy(t)
        outside=np.abs(np.arange(len(t['positions']))-175)>75
        changed['positions'][outside]+=100
        a=temporal_anchors(t,b,[('half',0,349)],[],np.array([]))[2]
        z=temporal_anchors(changed,b,[('half',0,349)],[],np.array([]))[2]
        for k in FIELDS:np.testing.assert_array_equal(a[k][175:176],z[k][175:176])

    def test_actual_synthetic_prefix_reconstruction(self):
        t,b,_=synthetic_sources();outputs=[]
        for n in (300,350):
            current={k:v[:n] if k in ('positions','valid') else v for k,v in t.items()}
            outputs.append(temporal_anchors(current,b,[('half',0,349)],[],np.array([]))[2])
        for k in FIELDS:np.testing.assert_array_equal(outputs[0][k][75:200],outputs[1][k][75:200])

    def test_rate_one_feasible_rate_point_eight_loses_support(self):
        rows,_=schedule(mapping());self.assertIn(12000,[r['tactical_center'] for r in rows])
        slow=mapping(rate=.8);rows,_=schedule(slow)
        self.assertNotIn(12000,[r['tactical_center'] for r in rows])
        old,_=historical_schedule(19060,slow,[('half',11185,30000)],[],np.array([]),np.ones(164487,bool),25)
        self.assertIn(12000,[r['tactical_center'] for r in old])

    def test_failed_inverse_and_missingness_ignore_beyond_100(self):
        a=mapping();z=mapping();outside=np.abs(np.arange(19060)-12000)>100
        z['frames'][outside]=-1;z['orientation'][outside]=-1
        first,_=schedule(a);second,_=schedule(z)
        self.assertEqual(next(r for r in first if r['tactical_center']==12000),next(r for r in second if r['tactical_center']==12000))

    def test_unsampled_source_frame_cut_and_video_end_are_rejected(self):
        usable=np.ones(164487,bool);usable[14999]=False
        rows,_=schedule(mapping(),usable=usable)
        self.assertNotIn(12000,[r['tactical_center'] for r in rows])
        last=mapping();last['frames']+=149390
        rows,_=schedule(last)
        self.assertNotIn(12000,[r['tactical_center'] for r in rows])

    def test_full121_future_features_prefix_equal(self):
        outputs=[]
        for n in (18935,19060):
            m=mapping(n);rows,_=schedule(m,n);q=query_plan(rows,m,n)
            outputs.append(synthetic_feature_view(q,m,n,[('half',11185,30000)]).iloc[11185:18685])
        pd.testing.assert_frame_equal(*outputs,check_exact=True);self.assertEqual(outputs[0].shape,(7500,121))

    def test_extension_preserves_fields_and_discontinuous_join(self):
        old=mapping();old.update(cost=np.zeros(19060),margin=np.ones(19060));old['frames'][12000:12050]=-1;old['orientation'][12000:12050]=-1
        new=copy.deepcopy(old);new['frames'][12000:12050]=np.arange(50)+9000;new['orientation'][12000:12050]=1
        out=extension(old,new);use=old['frames']>=0
        for k in FIELDS:self.assertEqual(old[k][use].tobytes(),out[k][use].tobytes())
        self.assertNotIn('anchor_allowed',out)
        rows,_=schedule(out);self.assertNotIn(12000,[r['tactical_center'] for r in rows])

    def test_checkpoints_guard75_and_inclusive_exclusions(self):
        args=(19060,[('half',11185,30000)])
        self.assertFalse(safe_interval(12000,75,*args,[],np.array([12325])))
        self.assertTrue(safe_interval(12000,75,*args,[],np.array([12326])))

    def test_extra_gate_cannot_accept_bad_individual_geometry(self):
        from broadcast_alignment import formation_cost as real_cost
        t,b,_=synthetic_sources()
        def cost(a,frames,*args):
            return np.full(1,2.51) if len(frames)==1 else real_cost(a,frames,*args)
        with patch('temporal_global_alignment.formation_cost',side_effect=cost):
            _,trace,out=temporal_anchors(t,b,[('half',0,349)],[],np.array([]))
        self.assertIn('extra_cost',trace['reason']);self.assertFalse(out['anchor_allowed'].any())

    def test_fine_individual_residual_and_disconnected_span_are_gated(self):
        t,b,_=synthetic_sources();real=temporal_costs
        for mode,expected in [('individual','fine_individual_cost'),('span','fine_span')]:
            def costs(frame,candidates,o,*args):
                if len(candidates)!=51:return real(frame,candidates,o,*args)
                score=np.full(51,9.);individual=np.full((51,3),9.)
                if mode=='individual':score[25]=np.sqrt(3.);individual[25]=[3.,0.,0.]
                else:score[[0,-1]]=0.;individual[[0,-1]]=0.
                return score,individual
            with patch('temporal_global_alignment.temporal_costs',side_effect=costs):
                _,trace,out=temporal_anchors(t,b,[('half',0,349)],[],np.array([]))
            self.assertIn(expected,trace['reason']);self.assertFalse(out['anchor_allowed'].any())

    def test_no_global_usable_triplets_aborts_explicitly(self):
        t,b,_=synthetic_sources();b['valid'][:]=False
        with self.assertRaisesRegex(ValueError,'No eligible temporal global'):
            temporal_anchors(t,b,[('half',0,349)],[],np.array([]))

    def test_public_prefix_projection_keeps_every_public_metric(self):
        from prepare_temporal_global_alignment import public_summary
        a=dict(variants=dict(baseline=dict(eligible_associations=1,unique_source_centers=1,
            eligibility_counts=dict(boundary=20),observed_public_frames=100,public_feature_associations=1)))
        b=copy.deepcopy(a);b['variants']['baseline'].update(eligible_associations=2,unique_source_centers=2,eligibility_counts=dict(boundary=19))
        self.assertEqual(public_summary(a),public_summary(b))
        b['variants']['baseline']['observed_public_frames']+=1
        self.assertNotEqual(public_summary(a),public_summary(b))


if __name__=='__main__':unittest.main()
