"""Synthetic checks for bounded inputs, slot invariance and validation exclusions."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from robust_motion import MAX_CONTEXT_FRAMES, set_relative_features, stationarity_features
from freeze_annotation_windows import sample_windows
from run_continuation import fit_ensemble, purged_spell_blocks
from sequence_diagnostics import sequence_details


class ContinuationTests(unittest.TestCase):
    def test_robust_displacement_has_physical_units(self):
        ball = np.column_stack((np.arange(200)*2/25, np.ones(200)))
        features = stationarity_features(ball)
        self.assertAlmostEqual(features.ball_robust_speed_past_1s.iloc[100], 2)
        self.assertAlmostEqual(features.ball_robust_speed_future_1s.iloc[100], 2)
        self.assertAlmostEqual(features.ball_iqr_1s.iloc[100], 1)
        stationary = stationarity_features(np.ones((200, 2)))
        self.assertEqual(stationary.ball_iqr_2s.iloc[100], 0)
        self.assertEqual(stationary.ball_stationary_step_fraction_1s.iloc[100], 1)

    def test_missing_ball_does_not_become_a_stationary_ball(self):
        ball = np.ones((250, 2))
        ball[50:200] = np.nan
        features = stationarity_features(ball)
        self.assertTrue(np.isnan(features.ball_iqr_2s.iloc[125]))
        self.assertTrue(np.isnan(features.ball_robust_speed_past_1s.iloc[125]))
        self.assertTrue(np.isnan(features.ball_stationary_step_fraction_1s.iloc[125]))
        self.assertEqual(features.ball_observed_fraction_2s.iloc[125], 0)

    def test_feature_context_is_bounded_in_both_directions(self):
        rng = np.random.default_rng(42)
        ball = rng.normal(size=(300, 2))
        changed = ball.copy()
        changed[:150-MAX_CONTEXT_FRAMES] = 1000
        changed[151+MAX_CONTEXT_FRAMES:] = -1000
        pd.testing.assert_series_equal(stationarity_features(ball).iloc[150],
                                       stationarity_features(changed).iloc[150])

    def test_relative_motion_ignores_slot_permutations_and_uses_metres(self):
        ball = np.column_stack((50+np.arange(100)*2/25, np.full(100, 34)))
        positions = np.zeros((100, 4, 2))
        positions[:, 0] = ball + [.5, 0]
        positions[:, 1] = [10, 10]
        positions[:, 2] = [70, 34]
        positions[:, 3] = [90, 10]
        permuted = positions.copy()
        permuted[1::2] = positions[1::2, [1, 0, 3, 2]]
        expected = set_relative_features(positions, ball, [0, 0, 1, 1])
        actual = set_relative_features(permuted, ball, [0, 0, 1, 1])
        pd.testing.assert_frame_equal(expected, actual)
        np.testing.assert_allclose(actual.h__nearest_set_relative_velocity.iloc[5:], 0, atol=1e-5)
        np.testing.assert_allclose(actual.a__nearest_set_relative_velocity.iloc[5:], 2, atol=1e-5)
        positions[30:60, :2] = np.nan
        missing = set_relative_features(positions, ball, [0, 0, 1, 1])
        self.assertTrue(missing.h__nearest_set_relative_velocity.iloc[40:50].isna().all())

    def test_purging_removes_whole_spells_including_the_buffer_survivors(self):
        labels = np.array(list("A"*10 + "H"*40 + "D"*10 + "A"*40))
        train = np.r_[np.arange(28), np.arange(42, 100)]
        folds = [(1, train, np.arange(30, 40))]
        _, purged, validation = purged_spell_blocks(labels, folds)[0]
        np.testing.assert_array_equal(purged, np.r_[np.arange(10), np.arange(50, 100)])
        np.testing.assert_array_equal(validation, np.arange(30, 40))

    def test_equidistant_set_matches_do_not_depend_on_slot_order(self):
        ball = np.column_stack((49.5+np.arange(20)/25, np.full(20, 34)))
        positions = np.zeros((20, 3, 2))
        positions[:, 0] = [49, 34]
        positions[:, 1] = [51, 34]
        positions[:, 2] = [80, 34]
        positions[5, 0] = [50, 34]
        permuted = positions.copy()
        permuted[:5, :2] = positions[:5, [1, 0]]
        a = set_relative_features(positions, ball, [0, 0, 1])
        b = set_relative_features(permuted, ball, [0, 0, 1])
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(np.isnan(a.h__nearest_set_relative_velocity.iloc[5]))

    def test_fit_targets_and_weights_do_not_read_validation_labels(self):
        inputs = pd.DataFrame({"h__x": np.arange(9), "a__x": -np.arange(9)})
        labels = np.array(list("HHADADHHA"))
        train, validation = np.arange(6), np.arange(6, 9)
        calls = []

        class FakeModel:
            def __init__(self, **kwargs):
                pass

            def fit(self, x, y, sample_weight):
                calls.append((x.copy(), y.copy(), sample_weight.copy()))

            def predict_proba(self, x):
                return np.tile([.4, .3, .3], (len(x), 1))

            def save_model(self, path):
                pass

        with patch("run_continuation.XGBClassifier", FakeModel):
            fit_ensemble(inputs, labels, train, validation, Path("."), 1)
            labels[validation] = "D"
            fit_ensemble(inputs, labels, train, validation, Path("."), 1)
        for first, second in zip(calls[:2], calls[2:]):
            pd.testing.assert_frame_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            np.testing.assert_array_equal(first[2], second[2])

    def test_sequence_diagnostic_catches_missed_stoppages_and_false_fragments(self):
        truth = np.array(list("HHHHHDDDAAA"))
        prediction = np.array(list("HHAHHHHHAAA"))
        summary, _ = sequence_details(np.arange(len(truth)), truth, prediction)
        self.assertEqual(summary["missed_stoppages"], 1)
        self.assertEqual(summary["dead_recall"], 0)
        self.assertEqual(summary["false_dead_frames"], 0)
        self.assertGreater(summary["live_run_fragments"], 0)

    def test_annotation_sampling_is_fixed_and_has_no_training_audit_overlap(self):
        halves = [(1, 11185, 90651), (2, 110808, 190197)]
        events = [dict(code="SET PIECES", start_frame=i) for i in
                  list(range(22000, 88000, 4000)) + list(range(114000, 188000, 4000))]
        a = sample_windows(events, halves, np.random.default_rng(42))
        b = sample_windows(events, halves, np.random.default_rng(42))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)
        for split in ("train", "audit"):
            for stratum in ("uniform", "restart"):
                self.assertEqual(sum(w["split"] == split and w["stratum"] == stratum for w in a), 6)
        for i, window in enumerate(a):
            self.assertEqual(window["end"]-window["start"]+1, 750)
            self.assertGreater(window["start"], 18684+250)
            for other in a[i+1:]:
                self.assertTrue(window["end"]+250 < other["start"] or window["start"] > other["end"]+250)


if __name__ == "__main__":
    unittest.main()
