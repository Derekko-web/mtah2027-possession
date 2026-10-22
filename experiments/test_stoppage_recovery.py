"""Weak exclusion, train-only preprocessing, and bounded conservative recovery."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from neural_control import fit_neural
from finalize_stoppage_recovery import promotion_checks
from rich_features import swap_teams
from stoppage_consensus import stoppage_emissions
from stoppage_context import context_features, fit_context, weak_samples
from stoppage_hysteresis import hysteresis_dead


class StoppageRecoveryTests(unittest.TestCase):
    def test_secondary_promotion_cannot_trade_false_D_or_short_recovery_for_recall(self):
        plan = dict(protocols=["buffered_minutes"], seeds=[42], methods=["current", "stoppage_consensus"])
        base = dict(protocol="buffered_minutes", spellF1=.73, macroF1=.84, false_dead_frames=92,
                    dead_recall=.67, missed_stoppages=1, short_live_mean_recall=.61)
        better = dict(base, macroF1=.87, dead_recall=.75, missed_stoppages=0)
        scores = pd.DataFrame([dict(base, model="current_seed42"), dict(better, model="stoppage_consensus_seed42")])
        spans = pd.DataFrame([dict(protocol="buffered_minutes", model=name, label="H", frames=19,
                                   start=10, end=28, recall=1.) for name in ("current_seed42", "stoppage_consensus_seed42")])
        self.assertTrue(promotion_checks(scores, spans, plan)["passed"])
        for field, value in (("spellF1", .72), ("macroF1", .841), ("false_dead_frames", 93),
                             ("dead_recall", .71), ("missed_stoppages", 1), ("short_live_mean_recall", .60)):
            changed = scores.copy()
            changed.loc[1, field] = value
            self.assertFalse(promotion_checks(changed, spans, plan)["passed"], field)
        spans.loc[1, "recall"] = 0.
        self.assertFalse(promotion_checks(scores, spans, plan)["passed"])

    def test_hysteresis_needs_sustained_core_and_stops_at_motion(self):
        head = np.full(400, .6)
        head[150:200] = .9
        visibility, moving = np.ones(400), np.zeros(400)
        prior = hysteresis_dead(head, visibility, moving, .5)
        self.assertTrue(np.all(prior[25:325] > 0))
        self.assertTrue(np.all(prior[:25] == 0))
        self.assertTrue(np.all(prior[325:] == 0))
        moving[130] = 1
        prior = hysteresis_dead(head, visibility, moving, .5)
        self.assertTrue(np.all(prior[:131] == 0))
        head[150] = .6
        self.assertFalse(hysteresis_dead(head, visibility, moving, .5).any())

    def test_weak_rows_exclude_public_and_all_reserved_windows(self):
        n = 22000
        raw = pd.DataFrame(dict(ball_displacement_1s=np.zeros(n), h__distance_1=np.full(n, 3.), a__distance_1=np.full(n, 4.)))
        physical = pd.DataFrame(dict(ball_observed_radius25=np.ones(n), ball_iqr_radius25=np.zeros(n)))
        restarts = pd.DataFrame(dict(restart_all_active=np.ones(n)))
        windows = [dict(start=20000, end=20749)]
        frames, target = weak_samples(raw, physical, restarts, windows)
        self.assertTrue(len(frames) > 0)
        self.assertFalse(np.any((frames >= 10935) & (frames <= 18934)))
        self.assertFalse(np.any((frames >= 19750) & (frames <= 20999)))
        self.assertTrue(np.all(frames % 10 == 0))
        self.assertTrue(np.all(target == 1))

    def test_context_is_team_invariant_and_prefix_matches_with_padding(self):
        n = 500
        rng = np.random.default_rng(21)
        base = pd.DataFrame(dict(ball_seen=np.ones(n), ball_displacement_1s=rng.uniform(0, 5, n)))
        raw = pd.DataFrame(index=base.index)
        for team in ("h__", "a__"):
            for name in ("distance_1", "speed_median", "moving_fraction"):
                base[team+name] = rng.uniform(0, 4, n)
            for name in ("spread_x", "spread_y", "center_ball_distance", "distance_median", "speed_p90"):
                raw[team+name] = rng.uniform(0, 8, n)
        ball = rng.normal(size=(n, 2))
        with patch("stoppage_context.load_tags", return_value=[]):
            a = context_features(raw, base, ball)
            b = context_features(swap_teams(raw), swap_teams(base), ball)
            prefix = context_features(raw.iloc[:350], base.iloc[:350], ball[:350])
        for x, y, z in zip(a, b, prefix):
            pd.testing.assert_frame_equal(x, y)
            pd.testing.assert_frame_equal(x.iloc[125:225], z.iloc[125:225])

    def test_neural_preprocessing_never_fits_heldout_features_or_labels(self):
        x = pd.DataFrame(dict(h__x=np.arange(12.), a__x=-np.arange(12.)))
        labels = np.array(list("HADHADHADHAD"))
        with patch("neural_control.MLPClassifier"):
            first = fit_neural(x, labels, np.arange(6), False)
            changed = x.copy()
            changed.iloc[6:] = 1e9
            labels[6:] = "X"
            second = fit_neural(changed, labels, np.arange(6), False)
        np.testing.assert_array_equal(first.named_steps["impute"].statistics_, second.named_steps["impute"].statistics_)
        np.testing.assert_array_equal(first.named_steps["scale"].mean_, second.named_steps["scale"].mean_)
        np.testing.assert_array_equal(first.named_steps["scale"].scale_, second.named_steps["scale"].scale_)

    def test_weak_training_ignores_nontraining_public_targets(self):
        x = pd.DataFrame(dict(value=np.arange(12.)))
        labels = np.array(list("HADHADHADHAD"))
        train, weak_frames, weak_target = np.arange(6), np.array([10, 11]), np.array([0, 1])
        with patch("stoppage_context.XGBClassifier") as constructor:
            fit_context(x, labels, train, weak_frames, weak_target, .5)
            first_x, first_y = constructor.return_value.fit.call_args.args
            weight = constructor.return_value.fit.call_args.kwargs["sample_weight"].copy()
            labels[6:] = "X"
            fit_context(x, labels, train, weak_frames, weak_target, .5)
            second_x, second_y = constructor.return_value.fit.call_args.args
            pd.testing.assert_frame_equal(first_x, second_x)
            np.testing.assert_array_equal(first_y, second_y)
            np.testing.assert_array_equal(weight, constructor.return_value.fit.call_args.kwargs["sample_weight"])
        with self.assertRaises(ValueError):
            fit_context(x, labels, train, np.array([1, 11]), weak_target, .5)

    def test_consensus_preserves_conditional_ownership_and_cannot_lower_D(self):
        reference = np.tile([.6, .3, .1], (300, 1))
        heads = np.full((300, 2), .1)
        heads[70:230, 0] = .99
        emission = stoppage_emissions(reference, heads, np.ones(300), np.zeros(300))
        self.assertTrue(np.all(emission[:, 2] >= reference[:, 2]))
        np.testing.assert_allclose(emission[:, 0]/emission[:, 1], 2.)
        self.assertGreater(emission[150, 2], .8)


if __name__ == "__main__":
    unittest.main()
