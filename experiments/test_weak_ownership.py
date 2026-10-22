"""Check label exclusions, weight provenance, and D-preserving ownership gates."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from weak_ownership import fit_weak_ownership, ownership_prediction, sample_weak_ownership


class WeakOwnershipTests(unittest.TestCase):
    def test_weak_exclusions_and_team_swap(self):
        n = 22000
        owner = np.arange(n)//2000 % 2
        raw = pd.DataFrame(dict(ball_displacement_1s=np.full(n, 6),
                                h__distance_1=np.where(owner == 0, 1., 4.),
                                a__distance_1=np.where(owner == 1, 1., 4.)))
        phase = pd.DataFrame(dict(h__phase_control_support=(owner == 0).astype(float),
                                  a__phase_control_support=(owner == 1).astype(float)))
        physical = pd.DataFrame(dict(ball_observed_radius25=np.ones(n), ball_iqr_radius25=np.full(n, 3.)))
        restart = pd.DataFrame(dict(restart_all_active=np.zeros(n)))
        windows = [dict(start=20000, end=20749)]
        with patch("weak_ownership.half_video_blocks", return_value=[(1, 0, n-1)]):
            first = sample_weak_ownership(raw, physical, restart, phase, windows)
            flipped_raw = raw.rename(columns={"h__distance_1":"a__distance_1", "a__distance_1":"h__distance_1"})
            flipped_phase = phase.rename(columns={"h__phase_control_support":"a__phase_control_support", "a__phase_control_support":"h__phase_control_support"})
            second = sample_weak_ownership(flipped_raw, physical, restart, flipped_phase, windows)
        for name, (frames, target) in first.items():
            self.assertEqual(set(target), {0, 1})
            self.assertFalse(np.any((frames >= 10935) & (frames <= 18934)))
            self.assertFalse(np.any((frames >= 19750) & (frames <= 20999)))
            self.assertTrue(np.all(frames % 10 == 0))
            np.testing.assert_array_equal(frames, second[name][0])
            np.testing.assert_array_equal(target, 1-second[name][1])

    def test_fit_never_reads_nontraining_targets_and_weak_mass_is_fixed(self):
        x = pd.DataFrame(dict(h__x=np.arange(12.), a__x=-np.arange(12.)))
        labels = np.array(list("HADHADHADHAD"))
        train, frames, target = np.arange(6), np.array([10, 11]), np.array([0, 1])
        with patch("weak_ownership.XGBClassifier") as constructor:
            fit_weak_ownership(x, labels, train, frames, target, .5)
            x1, y1 = constructor.return_value.fit.call_args.args
            weight = constructor.return_value.fit.call_args.kwargs["sample_weight"].copy()
            self.assertAlmostEqual(weight[4:6].sum()/weight[:4].sum(), .5)
            labels[6:] = "X"
            fit_weak_ownership(x, labels, train, frames, target, .5)
            x2, y2 = constructor.return_value.fit.call_args.args
            pd.testing.assert_frame_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)
            np.testing.assert_array_equal(weight, constructor.return_value.fit.call_args.kwargs["sample_weight"])
        with self.assertRaises(ValueError):
            fit_weak_ownership(x, labels, train, np.array([1, 11]), target, .5)

    def test_proposals_need_support_and_read_unchanged_reference(self):
        n = 120
        reference = np.zeros(n, int)
        p = np.tile([.99, .01], (n, 1))
        p[45:75] = [.01, .99]
        support = np.zeros((n, 2), bool)
        phase = np.zeros((n, 2))
        np.testing.assert_array_equal(ownership_prediction(reference, p, support, phase, "short_proposals"), reference)
        phase[45:75, 1] = .5
        result = ownership_prediction(reference, p, support, phase, "short_proposals")
        self.assertGreater((result == 1).sum(), 20)
        np.testing.assert_array_equal(ownership_prediction(1-reference, p[:, ::-1], support[:, ::-1],
                                                         phase[:, ::-1], "short_proposals"), 1-result)
        reference[60] = 2
        np.testing.assert_array_equal(ownership_prediction(reference, p, support, phase, "short_proposals"), reference)

    def test_live_replacement_cannot_change_D_or_bridge_D(self):
        reference = np.r_[np.zeros(20, int), np.full(10, 2), np.ones(20, int)]
        p = np.tile([.01, .99], (len(reference), 1))
        result = ownership_prediction(reference, p, np.zeros_like(p), np.zeros_like(p), "replace_live")
        np.testing.assert_array_equal(reference == 2, result == 2)
        self.assertTrue(np.all(result[:20] == 1))
        self.assertTrue(np.all(result[30:] == 1))


if __name__ == "__main__":
    unittest.main()
