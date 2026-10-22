"""Algorithmic and leakage checks for prospective boundary experiments."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from boundary_refinement import decode_refinement, fit_micro, micro_inputs, refine_block, relocate_boundaries, runs
from finalize_boundary_refinement import promotion_checks
from rich_features import swap_teams


class BoundaryTests(unittest.TestCase):
    def test_promotion_rejects_secondary_regressions_and_lost_recovery(self):
        plan = dict(methods=["current", "anchored_rescue"], protocols=["buffered_minutes"], seeds=[42])
        base = dict(protocol="buffered_minutes", spellF1=.68, dead_recall=.67, missed_stoppages=1,
                    macroF1=.84, false_dead_frames=140, short_live_mean_best_iou=.32)
        better = dict(base, spellF1=.73, false_dead_frames=92, short_live_mean_best_iou=.40)
        scores = pd.DataFrame([dict(base, model="current_seed42"), dict(better, model="anchored_rescue_seed42")])
        segments = pd.DataFrame([dict(protocol="buffered_minutes", label="H", frames=19, start=10, end=28,
                                      model=name, recall=1.) for name in ("current_seed42", "anchored_rescue_seed42")])
        self.assertEqual(promotion_checks(scores, segments, plan)["selected"], "anchored_rescue")
        for column, value in (("spellF1", .67), ("dead_recall", .64), ("missed_stoppages", 2),
                              ("macroF1", .83), ("false_dead_frames", 140), ("short_live_mean_best_iou", .32)):
            changed = scores.copy()
            changed.loc[1, column] = value
            self.assertIsNone(promotion_checks(changed, segments, plan)["selected"], column)
        lost = segments.copy()
        lost.loc[1, "recall"] = 0.
        self.assertIsNone(promotion_checks(scores, lost, plan)["selected"])

    @staticmethod
    def rescue_scene():
        current = np.tile([.02, .96, .02], (300, 1))
        current[100:140] = [.96, .02, .02]
        current[140:260] = [.02, .02, .96]
        previous = np.tile([.02, .96, .02], (300, 1))
        previous[95:195] = [.96, .02, .02]
        previous[195:260] = [.02, .02, .96]
        fast = np.tile([.05, .95], (300, 1))
        fast[80:210] = [.85, .15]
        return current, previous, fast, np.ones((300, 2), bool), np.zeros((300, 2))

    def test_rescues_a_supported_live_tail_at_dead_boundary(self):
        arrays = self.rescue_scene()
        before = refine_block(*arrays, "current")
        after = refine_block(*arrays, "anchored_rescue")
        self.assertGreater(np.count_nonzero(after[130:195] == 0), np.count_nonzero(before[130:195] == 0))
        self.assertTrue(np.all(after[220:250] == 2))
        swapped = [arrays[0][:, [1, 0, 2]], arrays[1][:, [1, 0, 2]],
                   arrays[2][:, [1, 0]], arrays[3][:, [1, 0]], arrays[4][:, [1, 0]]]
        np.testing.assert_array_equal(refine_block(*swapped, "anchored_rescue"), np.array([1, 0, 2])[after])

    def test_no_rescue_without_anchor_fast_agreement_or_contacts(self):
        original = self.rescue_scene()
        scenes = []
        no_anchor = [v.copy() for v in original]
        no_anchor[0][:] = [.02, .02, .96]
        scenes.append(no_anchor)
        no_fast = [v.copy() for v in original]
        no_fast[2][:] = [.5, .5]
        scenes.append(no_fast)
        no_contact = [v.copy() for v in original]
        no_contact[3][:] = False
        scenes.append(no_contact)
        for arrays in scenes:
            np.testing.assert_array_equal(refine_block(*arrays, "anchored_rescue"), refine_block(*arrays, "current"))

    def test_shared_decoder_resets_blocks_and_preserves_abstention(self):
        current, previous, fast, _, _ = self.rescue_scene()
        raw = pd.DataFrame(dict(ball_seen=np.ones(300), ball_displacement_1s=np.zeros(300),
                                h__distance_1=np.full(300, .5), a__distance_1=np.full(300, 5.),
                                h__control_share=np.ones(300), a__control_share=np.zeros(300),
                                h__speed_median=np.ones(300), a__speed_median=np.ones(300)))
        result = decode_refinement(current, previous, fast, raw, [(20, 260)])
        self.assertTrue(np.all(result[:20] == "X"))
        self.assertTrue(np.all(result[260:] == "X"))
        mutated = current.copy()
        mutated[260:] = [.99, .005, .005]
        np.testing.assert_array_equal(result, decode_refinement(mutated, previous, fast, raw, [(20, 260)]))
        with self.assertRaises(ValueError):
            decode_refinement(current, previous, fast, raw, [(20, 260), (200, 280)])
        invalid = fast.copy()
        invalid[100] = np.nan
        with self.assertRaises(ValueError):
            decode_refinement(current, previous, invalid, raw, [(20, 260)])

    def test_matches_exhaustive_constrained_likelihood(self):
        rng = np.random.default_rng(4)
        for _ in range(15):
            probability = rng.dirichlet([1, 1, 1], 12)
            states = np.repeat([0, 2, 1, 0], 3)
            result = relocate_boundaries(states, probability, 2)
            loss = -np.log(probability)
            actual = loss[np.arange(12), result].sum()
            feasible = []
            for cuts in itertools.product(range(1, 6), range(4, 9), range(7, 12)):
                if not 0 < cuts[0] < cuts[1] < cuts[2] < 12:
                    continue
                trial = np.repeat([0, 2, 1, 0], np.diff([0, *cuts, 12]))
                feasible.append(loss[np.arange(12), trial].sum())
            self.assertAlmostEqual(actual, min(feasible), places=8)
            self.assertEqual([s[2] for s in runs(result)], [0, 2, 1, 0])

    def test_ties_and_tiny_adjacent_runs_preserve_topology(self):
        states = np.array([0, 1, 0, 2, 1, 0])
        np.testing.assert_array_equal(relocate_boundaries(states, np.ones((6, 3)), 25), states)
        np.testing.assert_array_equal(relocate_boundaries(states, np.ones((6, 3)), 0), states)

    def test_team_swap_equivariance(self):
        rng = np.random.default_rng(7)
        probability = rng.dirichlet([1, 1, 1], 20)
        states = np.repeat([0, 2, 1, 0], 5)
        swap = np.array([1, 0, 2])
        np.testing.assert_array_equal(relocate_boundaries(swap[states], probability[:, swap], 3),
                                      swap[relocate_boundaries(states, probability, 3)])

    def test_micro_features_are_symmetric_and_use_padded_context(self):
        base = pd.DataFrame({"ball_seen": np.ones(100), "ball_speed": np.ones(100),
                             "h__distance_1": np.arange(100.), "a__distance_1": np.arange(100.)[::-1],
                             "h__control_share": np.linspace(0, 1, 100),
                             "a__control_share": np.linspace(1, 0, 100)})
        pd.testing.assert_frame_equal(micro_inputs(swap_teams(base)), swap_teams(micro_inputs(base)))
        pd.testing.assert_frame_equal(micro_inputs(base).iloc[20:70], micro_inputs(base.iloc[:83]).iloc[20:70])

    def test_fit_ignores_all_nontraining_targets(self):
        inputs = pd.DataFrame({"h__distance_1": np.arange(8.), "a__distance_1": np.arange(8.)[::-1]})
        train = np.array([0, 1, 4, 5])
        first = np.array(list("HADDHAXX"))
        second = first.copy()
        second[[2, 3, 6, 7]] = "H"
        with patch("boundary_refinement.XGBClassifier") as constructor:
            fit_micro(inputs, first, train, True)
            _, a = constructor.return_value.fit.call_args
            first_x, first_y = constructor.return_value.fit.call_args.args
            fit_micro(inputs, second, train, True)
            second_x, second_y = constructor.return_value.fit.call_args.args
            pd.testing.assert_frame_equal(first_x, second_x)
            np.testing.assert_array_equal(first_y, second_y)
            np.testing.assert_array_equal(a["sample_weight"], constructor.return_value.fit.call_args.kwargs["sample_weight"])


if __name__ == "__main__":
    unittest.main()
