"""Synthetic temporal correspondences and strict safeguards for the frozen trial."""
import unittest
import copy
from unittest.mock import patch
import numpy as np
from broadcast_alignment import formation_anchors, frame_usable, transform
from subgrid_alignment import (FIELDS, coarse_search, local_refinement, neighborhood_reason,
    refine_anchors, extension, safe_interval, metadata_gate)
from test_broadcast_alignment import synthetic_sources
from test_anchor_failures import PROFILES
from alignment_coverage import inverse_schedule


class SubgridTests(unittest.TestCase):
    def test_prefix_aggregate_projection_ignores_only_whole_prefix_rejection_counts(self):
        from prepare_subgrid_alignment import public_summary
        a=dict(variants={n:dict(observed_public_frames=2328,eligible_associations=90,
             eligibility_counts=dict(grid=10,boundary_or_prefix=5,sealed=0,eligible=5))
             for n in ('baseline','candidate','extension')},checkpoints=dict(usable=50))
        b=copy.deepcopy(a)
        for value in b['variants'].values():value['eligibility_counts'].update(boundary_or_prefix=0,sealed=5)
        self.assertEqual(public_summary(a),public_summary(b))
        b['variants']['extension']['observed_public_frames']+=1
        self.assertNotEqual(public_summary(a),public_summary(b))
        self.assertIn('eligibility_counts',a['variants']['baseline'])

    def test_local_refinement_recovers_true_non_grid_offset_and_reflection(self):
        t, b, _ = synthetic_sources()
        b['positions'][132:482] = transform(t['positions'], 3)
        b['teams'] = t['teams']
        win, counts, censored = local_refinement(150, 275, 3, t, b, frame_usable(b))
        self.assertEqual(int(win[0]), 282)
        self.assertAlmostEqual(win[1], 0)
        self.assertEqual(counts, (25, 25)); self.assertFalse(censored)

    def test_halfcell_never_reaches_beyond_12_and_masks_unusable(self):
        t, b, _ = synthetic_sources(); b['teams'] = t['teams']
        b['positions'][288] = transform(t['positions'][150], 3)
        b['valid'][263:288] = False
        win, counts, _ = local_refinement(150, 275, 3, t, b, frame_usable(b))
        self.assertFalse(np.isfinite(win[0])); self.assertEqual(counts, (25, 0))

    def test_bounds_and_disconnected_span(self):
        t, b, _ = synthetic_sources()
        def flat(*args): return np.zeros(len(args[1]))
        with patch('subgrid_alignment.formation_cost', side_effect=flat):
            win, counts, censored = local_refinement(150, 25, 0, t, b, frame_usable(b))
            self.assertTrue(censored); self.assertEqual(counts, (13, 13)); self.assertEqual(win[2], 12)
        def separated(*args):
            cost = np.ones(len(args[1])); cost[[0, -1]] = 0; return cost
        with patch('subgrid_alignment.formation_cost', side_effect=separated):
            win, _, _ = local_refinement(150, 275, 0, t, b, frame_usable(b))
            self.assertEqual(win[0], 263); self.assertEqual(win[2], 24)

    def test_global_far_runner_uses_strict_250_and_all_reflections(self):
        t, b, _ = synthetic_sources(); bf = np.array([25, 275, 300])
        def costs(_t, _b, _tt, _bt, o):
            return np.array([0, .1, 1.]) if o == 0 else np.array([5, 5, .4]) if o == 2 else np.ones(3)*9
        with patch('subgrid_alignment.formation_cost', side_effect=costs):
            coarse, runner = coarse_search(100, t, b, bf)
        self.assertEqual(coarse[0], 25); self.assertEqual(coarse[3], .4)
        self.assertEqual(tuple(runner[:2]), (300, 2))

    def test_absent_far_runner_keeps_infinite_margin(self):
        t, b, _ = synthetic_sources()
        coarse, runner = coarse_search(100, t, b, np.array([25, 50, 75]))
        self.assertTrue(np.isposinf(coarse[3])); self.assertTrue(np.isnan(runner[0]))

    def test_neighborhood_retains_every_strict_predicate(self):
        a = np.array([[100, 1, 1, .5], [125, 1, 1, np.inf], [150, 1, 1, .5]])
        spans = np.zeros(3)
        self.assertEqual(neighborhood_reason(a, spans), 'pass')
        for col, value, reason in [(2, 2.5001, 'neighbor_cost'), (3, .4999, 'neighbor_margin'),
                                   (1, 2, 'orientation'), (0, 176, 'coarse_increment')]:
            changed = a.copy(); changed[2, col] = value
            self.assertEqual(neighborhood_reason(changed, spans), reason)
        spans[0] = 11
        self.assertEqual(neighborhood_reason(a, spans), 'neighbor_halfcell_span')
        for centers in ([100, 100, 150], [100, 150, 150]):
            changed = a.copy(); changed[:, 0] = centers
            self.assertEqual(neighborhood_reason(changed, np.zeros(3)), 'pass')

    def test_extra_gate_cannot_bypass_a_bad_or_unavailable_offset(self):
        t, b, _ = synthetic_sources(); original = formation_anchors(t, b, PROFILES)
        def zero_cost(_t, frames, _tt, _bt, _o): return np.zeros(len(frames))
        # Force fine proposals while retaining true coarse replay and independent neighborhoods.
        def bad_extra(_t, frames, _tt, _bt, _o): return np.full(len(frames), 2.51)
        with patch('subgrid_alignment.neighborhood_reason', return_value='pass'), \
             patch('subgrid_alignment.fine_search', return_value=np.array([200., 0., 0.])), \
             patch('subgrid_alignment.formation_cost', side_effect=bad_extra), \
             patch('subgrid_alignment.coarse_search', side_effect=lambda f,*args:(original['coarse'][np.flatnonzero(original['tactical_frames']==f)[0]], np.array([np.nan,np.nan,np.inf]))), \
             patch('subgrid_alignment.local_refinement', return_value=(np.array([200.,0.,0.]),(25,25),False)):
            _, trace, mapping = refine_anchors(t,b,original,[('half',0,349)],[],np.array([]))
        self.assertIn('extra_cost', trace['reason']); self.assertFalse(mapping['anchor_allowed'].any())

    def test_excluded_context_never_calls_new_geometry(self):
        t,b,_ = synthetic_sources(); original=formation_anchors(t,b,PROFILES)
        with patch('subgrid_alignment.coarse_search', side_effect=AssertionError('Excluded geometry called')):
            _,trace,mapping=refine_anchors(t,b,original,[('half',0,349)],[dict(start=175,end=175)],np.array([]))
        self.assertFalse(trace['safe'].any()); self.assertFalse((mapping['frames']>=0).any())

    def test_sourceprefix_reconstruction_preserves_interior(self):
        t,b,_ = synthetic_sources(); results=[]
        for n in (300,350):
            current={k:v[:n] if k in ('positions','valid') else v for k,v in t.items()}
            old=formation_anchors(current,b,PROFILES)
            results.append(refine_anchors(current,b,old,[('half',0,349)],[],np.array([])))
        for k in FIELDS:
            self.assertTrue(np.array_equal(results[0][2][k][50:225],results[1][2][k][50:225],equal_nan=True))

    def test_refined_mapping_recovers_known_non_grid_trajectory(self):
        t,b,_=synthetic_sources(); b['positions'][132:482]=transform(t['positions'],3);b['teams']=t['teams']
        old=formation_anchors(t,b,PROFILES)
        _,_,mapped=refine_anchors(t,b,old,[('half',0,349)],[],np.array([]))
        use=mapped['frames']>=0
        self.assertGreater(int(use.sum()),100)
        self.assertLessEqual(int(np.max(np.abs(mapped['frames'][use]-np.flatnonzero(use)-132))),1)
        self.assertTrue((mapped['orientation'][use]==3).all())

    def test_far_duplicate_cannot_gain_acceptance_from_refinement(self):
        t,b,offset=synthetic_sources()
        b['positions']=np.concatenate([b['positions'],b['positions'][offset:offset+350]])
        b['valid']=np.ones(len(b['positions']),bool)
        old=formation_anchors(t,b,PROFILES)
        _,_,mapped=refine_anchors(t,b,old,[('half',0,349)],[],np.array([]))
        self.assertFalse((mapped['frames']>=0).any())

    def test_refined_runner_can_veto_despite_good_original_margin(self):
        t,b,_=synthetic_sources();old=formation_anchors(t,b,PROFILES)
        def search(f,*args):
            row=old['coarse'][np.flatnonzero(old['tactical_frames']==f)[0]].copy()
            row[2:]=[1.,1.]
            return row,np.array([600.,0.,2.])
        def refine(f,c,*args):
            return np.array([c,.1 if c==600 else .2,0.]),(25,25),False
        with patch('subgrid_alignment.coarse_search',side_effect=search), \
             patch('subgrid_alignment.equal'), \
             patch('subgrid_alignment.local_refinement',side_effect=refine):
            anchors,trace,mapped=refine_anchors(t,b,old,[('half',0,349)],[],np.array([]))
        self.assertTrue((anchors['coarse'][:,3]<.5).all())
        self.assertIn('neighbor_margin',trace['reason']);self.assertFalse(mapped['anchor_allowed'].any())

    def test_extension_preserves_all_fields_and_omits_composite_anchor_mask(self):
        old=dict(frames=np.array([100,-1,102]),orientation=np.array([0,-1,0]),cost=np.array([.7,np.nan,.8]),margin=np.array([np.inf,np.nan,.9]),anchor_allowed=np.array([True]))
        new=dict(frames=np.array([900,101,902]),orientation=np.ones(3,int),cost=np.ones(3),margin=np.ones(3),anchor_allowed=np.array([False]))
        result=extension(old,new)
        for k in FIELDS: self.assertTrue(np.array_equal(result[k][[0,2]],old[k][[0,2]],equal_nan=True))
        self.assertEqual(result['frames'][1],101); self.assertNotIn('anchor_allowed',result)

    def test_extension_join_does_not_force_clip_bridge(self):
        n=19060; old=dict(frames=np.arange(n)+3000,orientation=np.zeros(n,int),cost=np.zeros(n),margin=np.ones(n))
        old['frames'][12000:12050]=-1; old['orientation'][12000:12050]=-1
        new={k:v.copy() for k,v in old.items()}; new['frames'][12000:12050]=np.arange(50)+9000;new['orientation'][12000:12050]=0
        combined=extension(old,new)
        rows,_=inverse_schedule(n,combined,[('half',11185,30000)],[],np.array([]),np.ones(164487,bool),25)
        self.assertNotIn(12000,[r['tactical_center'] for r in rows])

    def test_context_intersection_is_inclusive(self):
        args=(19060,[('half',11185,30000)])
        self.assertFalse(safe_interval(12000,25,*args,[dict(start=12275,end=12275)],np.array([])))
        self.assertFalse(safe_interval(12000,25,*args,[],np.array([12275])))
        self.assertTrue(safe_interval(12000,25,*args,[],np.array([12276])))

    def test_every_metadata_gate_is_required_and_empty_fails(self):
        cover=dict(observed_public_frames=3100,net_additional_public_frames=772,original_coverage_retention=1.)
        quality=dict(baseline=dict(usable=100,bad_fraction=.01),new_only=dict(usable=50,bad_fraction=.03,source_availability=.98))
        rule=dict(min_observed_frames=3000,min_net_additional_frames=750,min_original_coverage_retention=.98)
        ck=dict(minimum_new_usable=50,minimum_new_source_availability=.98,maximum_excess_bad_fraction=.02)
        self.assertTrue(metadata_gate(cover,quality,rule,ck,True)['passed'])
        for key,value in [('usable',49),('bad_fraction',.031),('source_availability',.979),('usable',0)]:
            bad=dict(quality,new_only=dict(quality['new_only'],**{key:value}))
            self.assertFalse(metadata_gate(cover,bad,rule,ck,True)['passed'])
        self.assertFalse(metadata_gate(cover,quality,rule,ck,False)['passed'])


if __name__ == '__main__': unittest.main()
