"""Synthetic evidence for matching, fixed oracle limits and continuation rules."""
import copy
import unittest
import numpy as np
from dfl_signal_audit import (matching,teacher_diagnostic,conditional_probability,
    oracle_probability,keep_reference_dead_unknown,confidence_summary,oracle_effect,continuation,native_prefix_lengths)

RULE=dict(minimum_balanced_accuracy=.70,minimum_confident_corrections=25,
    minimum_correction_precision=.55,minimum_case_probability=.60,minimum_common_cases=1)


def protocol(case_id=3):
    case=dict(gt_id=case_id,reference_matched=False,frames=71,teacher_matched=True,
        teacher_correct_probability=.7,oracle={str(slot):dict(target_gt_matched=True,
            retained_all_previously_matched=True,no_extra_spurious=True) for slot in (0,4)})
    return dict(confidence=dict(balanced_accuracy=.8,confident_corrections=30,correction_precision=.6),truth_cases=[case])


class DFLSignalTests(unittest.TestCase):
    def test_native_observation_prefix_is_distinct_from_decoder_cache(self):
        actual=native_prefix_lengths(18935,19060)
        self.assertEqual(actual,dict(decoder_rows=18935,short_rows=18935,long_rows=19060))
    def test_official_matching_includes_half_iou_and_predictor_identity(self):
        m=matching(np.array(list('HHHHAAAA')),np.array(list('HHAAAAAA')))
        self.assertEqual(m['pairs'][0][:2],(1,1))
        self.assertTrue(m['gt_flags'].all());self.assertTrue(m['pred_flags'].all())
        self.assertEqual(m['pairs'][1][2],.5)
    def test_unknown_gaps_split_spells(self):
        m=matching(np.array(list('HHXHH')),np.array(list('HHHHH')))
        self.assertEqual(len(m['truth']),2);self.assertEqual(len(m['predicted']),2)
        self.assertTrue(m['gt_flags'].all())
    def test_predicted_abstention_filling_matches_official_temporal_policy(self):
        m=matching(np.array(list('HHHH')),np.array(list('XHXX')))
        self.assertEqual(m['pairs'],[(0,0,1.)])
    def test_teacher_resets_at_dead_unknown_and_eval_boundaries(self):
        reference=np.array(list('HHHDDXXXAAAAHHHH'))
        p=np.r_[np.ones(3)*.1,np.ones(5)*.9,np.ones(4)*.1,np.ones(4)*.9]
        result=teacher_diagnostic(reference,p,[(0,12),(12,16)])
        np.testing.assert_array_equal(result,np.array(list('AAADDXXXAAAAHHHH')))
    def test_live_overlay_protects_dead_unknown_and_alternative_dead(self):
        actual=keep_reference_dead_unknown(np.array(list('HADXH')),np.array(list('AHAAD')))
        np.testing.assert_array_equal(actual,np.array(list('AHDXH')))
    def test_oracle_preserves_mass_dead_and_untargeted_rows(self):
        original=np.array([[.2,.3,.5],[.1,.8,.1],[0.,0.,1.],[.7,.2,.1]])
        after=oracle_probability(original,[0,2],[1,0])
        np.testing.assert_array_equal(after[:,2],original[:,2])
        np.testing.assert_array_equal(after[[1,3]],original[[1,3]])
        np.testing.assert_allclose(after[:,:2].sum(axis=1),original[:,:2].sum(axis=1))
        self.assertAlmostEqual(after[0,1],.495);self.assertEqual(after[2,0],0)
    def test_oracle_rejects_wrong_target_or_frame(self):
        for frames,target in [([0],[2]),([-1],[0]),([1],[0]),([0,0],[0,1])]:
            with self.assertRaises(ValueError):oracle_probability(np.array([[.3,.5,.2]]),frames,target)
    def test_conditional_transfer_preserves_original_dead_mass(self):
        old=np.array([[.2,.3,.5],[0.,0.,1.]])
        actual=conditional_probability(old,np.array([[.9,.1,0],[0,0,1.]]))
        np.testing.assert_allclose(actual,[[.45,.05,.5],[0,0,1.]])
        np.testing.assert_array_equal(old,[[.2,.3,.5],[0,0,1.]])
    def test_confident_disagreements_partition_correction_and_harm(self):
        m=confidence_summary(np.array(list('HHAADX')),np.array(list('AHAHHA')),[.8,.2,.8,.2,.9,.9])
        self.assertEqual(m['confident_corrections'],2);self.assertEqual(m['confident_harms'],2)
        self.assertEqual(m['correction_precision'],.5);self.assertEqual(m['balanced_accuracy'],.5)
    def test_zero_disagreement_is_undefined_and_gate_fails(self):
        m=confidence_summary(np.array(list('HA')),np.array(list('HA')),[.9,.1])
        self.assertIsNone(m['correction_precision'])
        p=protocol();p['confidence'].update(m)
        self.assertFalse(continuation({'p':p},RULE)['passed'])
    def test_common_case_required_not_different_cases_in_each_protocol(self):
        self.assertFalse(continuation({'a':protocol(1),'b':protocol(2)},RULE)['passed'])
    def test_priority_and_all_protocol_conjunction(self):
        data={'a':protocol(),'b':protocol()}
        self.assertEqual(continuation(data,RULE)['selected_slot'],4)
        data['b']['truth_cases'][0]['oracle']['4']['no_extra_spurious']=False
        self.assertEqual(continuation(data,RULE)['selected_slot'],0)
        data['a']['confidence']['balanced_accuracy']=.69
        self.assertFalse(continuation(data,RULE)['passed'])
    def test_greedy_reassignment_does_not_count_as_clean_new_capacity(self):
        before=dict(gt_flags=np.array([True,False]),pred_flags=np.array([True,False]))
        after=dict(gt_flags=np.array([False,True]),pred_flags=np.array([False,True]))
        effect=oracle_effect(before,after,1)
        self.assertTrue(effect['target_gt_matched']);self.assertFalse(effect['retained_all_previously_matched'])
        self.assertEqual(effect['lost_matched_gt_ids'],[0])


if __name__=='__main__':unittest.main()
