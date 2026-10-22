"""Promotion must reject score, stopping, preservation and completeness regressions."""

import unittest

import pandas as pd

from finalize_motion_dead_trim import promotion_checks


class MotionDeadTrimPromotionTests(unittest.TestCase):
    def fixture(self):
        plan = dict(protocols=["buffered_minutes","purged_spells","buffered_100_seconds"],seeds=[42,43,44])
        scores,segments,official,tables = [],[],[],{}
        for protocol in plan["protocols"]:
            table = pd.DataFrame(dict(ground_truth=["H","H","D","A"]))
            for seed in plan["seeds"]:
                for new in (False,True):
                    name = f"motion_dead_trim_seed{seed}" if new else f"current_seed{seed}"
                    scores.append(dict(protocol=protocol,model=name,spellF1=.7+(.01 if new and protocol == "purged_spells" else 0),
                        macroF1=.8+(.001 if new else 0),short_live_mean_best_iou=.3+(.01 if new else 0),
                        false_dead_frames=0 if new else 1,dead_recall=.75,missed_stoppages=0,short_live_matched_iou50=2))
                    segments.append(dict(protocol=protocol,model=name,label="H",frames=19,start=0,end=18,recall=.7 if new else .6))
                    official.append(dict(protocol=protocol,model=name,spurious_iou50=4,missed_iou50=4))
                    table[name] = ["H","H","D","A"] if new else ["H","D","D","A"]
            tables[protocol] = table
        return pd.DataFrame(scores),pd.DataFrame(segments),pd.DataFrame(official),tables,plan

    def test_incremental_pass_does_not_claim_original_substantial_gates(self):
        result = promotion_checks(*self.fixture())
        self.assertTrue(result["passed"])
        self.assertFalse(result["original_substantial_primary_criterion_met"])
        self.assertFalse(result["original_extra_buffered_short_match_criterion_met"])

    def test_each_critical_metric_and_official_count_guard_rejects(self):
        for metric,value in (("spellF1",.69),("macroF1",.799),("short_live_mean_best_iou",.29),
                             ("false_dead_frames",1),("dead_recall",.74),("missed_stoppages",1)):
            with self.subTest(metric=metric):
                args = self.fixture();args[0].loc[1,metric] = value
                self.assertFalse(promotion_checks(*args)["passed"])
        for metric in ("spurious_iou50","missed_iou50"):
            args = self.fixture();args[2].loc[1,metric] = 5
            self.assertFalse(promotion_checks(*args)["passed"])
        args = self.fixture();args[1].loc[1,"recall"] = .4
        self.assertFalse(promotion_checks(*args)["passed"])

    def test_frame_preservation_and_incomplete_evidence_reject(self):
        for row,label in ((0,"A"),(0,"D"),(2,"H")):
            args = self.fixture();args[3]["buffered_minutes"].loc[row,"motion_dead_trim_seed42"] = label
            self.assertFalse(promotion_checks(*args)["passed"])
        args = list(self.fixture());args[0] = args[0].iloc[:-1]
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            promotion_checks(*args)


if __name__ == "__main__": unittest.main()
