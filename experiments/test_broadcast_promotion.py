"""Incremental primary gains cannot conceal secondary or false-spell losses."""

import unittest

import pandas as pd

from finalize_broadcast_closure import promotion_checks


class BroadcastPromotionTests(unittest.TestCase):
    def test_every_incremental_and_secondary_guard_is_required(self):
        plan = dict(protocols=["buffered_minutes"],seeds=[42])
        base = dict(protocol="buffered_minutes",spellF1=.73,macroF1=.87,short_live_mean_best_iou=.4)
        scores = pd.DataFrame([dict(base,model="current_seed42"),dict(base,model="broadcast_closure_seed42",spellF1=.745,macroF1=.871,short_live_mean_best_iou=.45)])
        segments = pd.DataFrame([dict(protocol="buffered_minutes",model=name,label="H",frames=19,start=10,end=28,recall=1.)
                                 for name in ("current_seed42","broadcast_closure_seed42")])
        official = pd.DataFrame([dict(protocol="buffered_minutes",model=name,spurious_iou50=4,missed_iou50=4)
                                 for name in ("current_seed42","broadcast_closure_seed42")])
        result = promotion_checks(scores,segments,official,plan)
        self.assertTrue(result["passed"])
        self.assertFalse(result["original_buffered_substantial_criterion_met"])
        for field,value in (("spellF1",.739),("macroF1",.868),("short_live_mean_best_iou",.39)):
            changed = scores.copy(); changed.loc[1,field] = value
            self.assertFalse(promotion_checks(changed,segments,official,plan)["passed"],field)
        for field in ("spurious_iou50","missed_iou50"):
            changed = official.copy(); changed.loc[1,field] = 5
            self.assertFalse(promotion_checks(scores,segments,changed,plan)["passed"],field)
        changed = segments.copy(); changed.loc[1,"recall"] = 0.
        self.assertFalse(promotion_checks(scores,changed,official,plan)["passed"])


if __name__ == "__main__": unittest.main()
