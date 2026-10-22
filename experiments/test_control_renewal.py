"""Leakage, geometry and transition-preservation checks for new research."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from control_chronology import contact_chronology, receiver_geometry
from handoff_hazard import event_targets, relocate_handoffs
from research_control_renewal import fit_renewal
from rich_features import swap_teams
from transition_penalty import decode


class RenewalTests(unittest.TestCase):
    def test_contact_chronology_is_symmetric_bounded_and_missing_safe(self):
        base = pd.DataFrame(dict(h__distance_1=np.full(400, .5), a__distance_1=np.full(400, 5.)))
        base.loc[140:260] = np.nan
        a = contact_chronology(base)
        pd.testing.assert_frame_equal(contact_chronology(swap_teams(base)), swap_teams(a))
        changed = base.copy()
        changed.loc[326:] = [5., .5]
        pd.testing.assert_frame_equal(a.iloc[:200], contact_chronology(changed).iloc[:200])
        self.assertEqual(a.loc[200, "h__touch2_bracket_25"], 0.)
        base[:] = np.nan
        self.assertTrue(np.all(contact_chronology(base).to_numpy() == 0))

    def test_receiver_geometry_is_slot_permutation_and_team_equivariant(self):
        rng = np.random.default_rng(19)
        positions = rng.normal(size=(90, 6, 2))
        ball = rng.normal(size=(90, 2))
        teams = np.array([0, 0, 0, 1, 1, 1])
        a = receiver_geometry(positions, ball, teams)
        perm = [2, 0, 1, 5, 3, 4]
        pd.testing.assert_frame_equal(a, receiver_geometry(positions[:, perm], ball, teams[perm]))
        pd.testing.assert_frame_equal(swap_teams(a), receiver_geometry(positions, ball, 1-teams))
        pd.testing.assert_frame_equal(a.iloc[5:65], receiver_geometry(positions[:90], ball[:90], teams).iloc[5:65])

    def test_event_targets_ignore_heldout_and_discontinuous_labels(self):
        labels = np.array(list("HHHHHHHHAAAAAAAADDDDDDDDHHHHHHHHAAAAAAAADDDDDDDD"))
        train = np.r_[np.arange(20), np.arange(28, len(labels))]
        a = event_targets(labels, train)
        changed = labels.copy()
        changed[20:28] = "H"
        b = event_targets(changed, train)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        for frame in a[0]:
            self.assertTrue(np.isin(np.arange(frame-4, frame+4), train).all())
        self.assertTrue(a[2].any())

    def test_handoff_relocation_preserves_dead_spans_and_team_symmetry(self):
        states = np.repeat([0, 1, 0, 2, 1, 0], [15, 8, 15, 20, 10, 12])
        probability = np.full((80, 3), .1)
        probability[np.arange(80), states] = .8
        hazard = np.full((80, 2), .01)
        hazard[12, 1], hazard[27, 0], hazard[70, 0] = .99, .99, .99
        result = relocate_handoffs(states, probability, hazard, 13, 10)
        np.testing.assert_array_equal(result == 2, states == 2)
        swap = np.array([1, 0, 2])
        np.testing.assert_array_equal(relocate_handoffs(swap[states], probability[:, swap], hazard[:, ::-1], 13, 10), swap[result])
        np.testing.assert_array_equal(relocate_handoffs(states, probability, hazard, 0, 10), states)

    def test_model_fit_uses_only_training_targets_and_weights(self):
        x = pd.DataFrame(dict(h__x=np.arange(12), a__x=-np.arange(12)))
        labels = np.array(list("HADHHADHADHA"))
        with patch("research_control_renewal.XGBClassifier") as constructor:
            fit_renewal(x, labels, np.arange(6), "runs", 42)
            first_x, first_y = constructor.return_value.fit.call_args.args
            first_weights = constructor.return_value.fit.call_args.kwargs["sample_weight"].copy()
            labels[6:] = "X"
            fit_renewal(x, labels, np.arange(6), "runs", 42)
            second_x, second_y = constructor.return_value.fit.call_args.args
            pd.testing.assert_frame_equal(first_x, second_x)
            np.testing.assert_array_equal(first_y, second_y)
            np.testing.assert_array_equal(first_weights, constructor.return_value.fit.call_args.kwargs["sample_weight"])


if __name__ == "__main__":
    unittest.main()
