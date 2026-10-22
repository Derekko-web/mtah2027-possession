"""Behavior checks for the promoted observation and fitting path."""

from unittest import TestCase, main
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from rich_features import swap_teams
from build_temporal_candidate import main as build_candidate
from temporal_candidate import decode_blocks, fit_models, ordered_inputs, predict_models
from temporal_control import contact_memory, temporal_taps
from validate_temporal_candidate import long_blocks


def observations(n=500):
    rng = np.random.default_rng(12)
    base = pd.DataFrame({"ball_seen": np.ones(n), "ball_displacement_1s": rng.random(n),
                         "ball_goal_line_distance": rng.random(n),
                         "ball_touchline_distance": rng.random(n)})
    for team in ("h__", "a__"):
        for name in ("distance_1", "control_share", "nearest_speed"):
            base[team+name] = rng.random(n)*5
    return base.astype(np.float32)


class TemporalCandidateTests(TestCase):
    def test_alternative_default_cannot_overwrite_selected_candidate(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root/"data/candidate_temporal_xgb"
            selected.mkdir(parents=True)
            sentinel = selected/"candidate.csv"
            sentinel.write_text("preserved selected candidate")
            with patch("build_temporal_candidate.ROOT", root), patch(
                "sys.argv", ["build_temporal_candidate.py", "--method", "ordered_catboost_frame"]
            ), patch("build_temporal_candidate.build_inputs", side_effect=RuntimeError("before data")):
                with self.assertRaisesRegex(RuntimeError, "before data"):
                    build_candidate()
            self.assertEqual(sentinel.read_text(), "preserved selected candidate")
            self.assertTrue((root/"data/candidate_temporal").is_dir())

    def test_features_ignore_inputs_outside_the_declared_context(self):
        base = observations()
        changed = base.copy()
        changed.iloc[:125] = 100
        changed.iloc[301:] = -100
        # At row 250 the memory reaches 125 frames back; future taps reach 50.
        pd.testing.assert_series_equal(ordered_inputs(base).iloc[250], ordered_inputs(changed).iloc[250])

    def test_longer_observation_table_does_not_change_interior_features(self):
        base = observations()
        short = ordered_inputs(base.iloc[:400])
        full = ordered_inputs(base)
        pd.testing.assert_frame_equal(short.iloc[:350], full.iloc[:350])

    def test_missing_ball_is_no_contact_and_history_expires(self):
        base = observations(300)
        base.loc[:, "h__distance_1"] = np.nan
        base.loc[:, "a__distance_1"] = np.nan
        base.loc[10, ["h__distance_1", "a__distance_1"]] = [.5, 4]
        memory = contact_memory(base)
        self.assertEqual(memory.h__recent_contact_decay.iloc[10], 1)
        self.assertEqual(memory.h__recent_contact_decay.iloc[136], 0)
        self.assertEqual(memory.h__contact_fraction_past_25.iloc[40], 0)
        self.assertEqual(memory.a__recent_contact_decay.max(), 0)
        self.assertEqual(memory.h__next_contact_decay.iloc[10], 1)

    def test_ordered_observations_distinguish_approach_from_departure(self):
        base = observations(200)
        base["h__distance_1"] = np.arange(200, dtype=float)
        taps = temporal_taps(base)
        self.assertEqual(taps.h__distance_1_lag_25.iloc[100], 75)
        self.assertEqual(taps["h__distance_1_lag_-25"].iloc[100], 125)
        self.assertTrue(np.isnan(taps.h__distance_1_lag_25.iloc[0]))

    def test_team_swap_commutes_with_feature_construction(self):
        base = observations()
        pd.testing.assert_frame_equal(ordered_inputs(swap_teams(base)), swap_teams(ordered_inputs(base)))

    def test_fit_never_reads_held_out_labels_for_targets_or_weights(self):
        base = observations(12)
        labels = np.array(list("HHADDADHAHDA"))
        calls = []

        class Model:
            def __init__(self, **kwargs):
                pass

            def fit(self, x, y, sample_weight):
                calls.append((x.copy(), y.copy(), sample_weight.copy()))

        with patch("temporal_candidate.XGBClassifier", Model):
            fit_models("ordered_xgboost_frame", base, labels, np.arange(6))
            labels[6:] = "X"
            fit_models("ordered_xgboost_frame", base, labels, np.arange(6))
        pd.testing.assert_frame_equal(calls[0][0], calls[1][0])
        np.testing.assert_array_equal(calls[0][1], calls[1][1])
        np.testing.assert_array_equal(calls[0][2], calls[1][2])
        self.assertEqual(len(calls[0][0]), 12)

    def test_symmetric_inference_even_for_asymmetric_model(self):
        class Model:
            classes_ = np.arange(3)

            def predict_proba(self, x):
                h = .1+.1*x.h__distance_1.to_numpy()
                return np.column_stack((h, np.full(len(x), .1), .9-h))

        base = observations()
        np.testing.assert_allclose(predict_models([Model()], base),
                                   predict_models([Model()], swap_teams(base))[:, [1, 0, 2]])

    def test_decode_abstains_outside_blocks_and_resets_at_each_boundary(self):
        p = np.tile([.01, .01, .98], (200, 1))
        p[20:40] = [.9, .05, .05]
        p[120:140] = [.05, .9, .05]
        result = decode_blocks(p, np.full(200, -1), [(20, 40), (120, 140)])
        self.assertTrue(np.all(result[20:40] == "H"))
        self.assertTrue(np.all(result[120:140] == "A"))
        self.assertEqual(np.count_nonzero(result == "X"), 160)
        with self.assertRaises(ValueError):
            decode_blocks(p, np.full(200, -1), [(20, 40), (30, 50)])

    def test_long_blocks_hold_out_every_frame_once_with_full_buffers(self):
        public = np.arange(11185, 18685)
        folds = long_blocks(public)
        np.testing.assert_array_equal(np.concatenate([v for _, _, v in folds]), public)
        for _, train, validation in folds:
            self.assertEqual(len(validation), 2500)
            self.assertTrue(np.all((train < validation[0]-250) | (train > validation[-1]+250)))


if __name__ == "__main__":
    main()
