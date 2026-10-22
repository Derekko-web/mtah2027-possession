import unittest

import numpy as np
import pandas as pd

from ownership_graph import fitting_weights, graph_views, matched_velocity, native_graph


def synthetic_source(reverse=False):
    n = 71
    xy = np.array([[2., 0], [6, 3], [-4, -5], [-2, 1], [7, -4], [-6, 5]])
    positions = np.repeat(xy[None], n, axis=0)
    ball = np.column_stack(((np.arange(n)-35)*(-.1 if reverse else .1), np.zeros(n)))
    return dict(positions=positions, ball=ball, teams=np.array([0, 0, 0, 1, 1, 1]), valid=np.ones(n, bool))


class OwnershipGraphTests(unittest.TestCase):
    def test_direction_distinguishes_equal_speed_and_distance(self):
        forward, _ = native_graph(synthetic_source(), [35])
        reverse, _ = native_graph(synthetic_source(True), [35])
        self.assertGreater(forward.h__radial_in_b2.iloc[0], 0)
        self.assertLess(reverse.h__radial_in_b2.iloc[0], 0)
        np.testing.assert_allclose(forward.h__tangent_in_b2, reverse.h__tangent_in_b2)

    def test_slot_rotation_reflection_and_team_symmetry(self):
        source = synthetic_source()
        original, info = native_graph(source, np.arange(10, 61))
        perm = np.array([4, 2, 0, 5, 1, 3])
        moved = {name:value.copy() for name, value in source.items()}
        matrix = np.array([[0., -1], [-1, 0]])
        moved['positions'] = source['positions'][:, perm]@matrix+np.array([30., -12])
        moved['teams'] = source['teams'][perm]
        moved['ball'] = source['ball']@matrix+np.array([30., -12])
        actual, actual_info = native_graph(moved, np.arange(10, 61))
        pd.testing.assert_frame_equal(original, actual, atol=1e-5, rtol=1e-5)
        pd.testing.assert_frame_equal(info, actual_info, atol=1e-5, rtol=1e-5)
        swapped = {**source, 'teams':1-source['teams']}
        actual, _ = native_graph(swapped, np.arange(10, 61))
        from rich_features import swap_teams
        pd.testing.assert_frame_equal(actual, swap_teams(original))

    def test_ambiguous_matching_and_cut_are_missing(self):
        current = np.array([[[0., 0], [0, 0], [5, 0]]])
        value = matched_velocity(current, current.copy(), .2, np.array([True]))
        self.assertTrue(np.isnan(value[0, :2]).all())
        np.testing.assert_array_equal(value[0, 2], [0., 0])
        source = synthetic_source()
        source['valid'][32] = False
        features, info = native_graph(source, [35])
        self.assertEqual(info.seen.iloc[0], 0)
        self.assertTrue(features.drop(columns='seen').isna().all().all())

    def test_broadcast_enclosure_resets_at_gap(self):
        source = synthetic_source()
        mapping = dict(frames=np.arange(71), orientation=np.zeros(71, int))
        features, info = graph_views(source, source, mapping)
        self.assertEqual(features.shape[1], 140)
        self.assertTrue((info.bg_seen.iloc[:10] == 0).all())
        self.assertEqual(info.bg_seen.iloc[35], 1)
        mapping['frames'][30] = -1
        _, info = graph_views(source, source, mapping)
        self.assertTrue((info.bg_seen.iloc[20:41] == 0).all())

    def test_weights_only_read_eligible_contiguous_targets(self):
        labels = np.array(list(('H'*60+'A'*60+'D'*60)*2))
        train = np.r_[np.arange(180), np.arange(240, 360)]
        y, weight = fitting_weights(labels, train, True)
        hidden = labels.copy()
        hidden[180:240] = 'X'
        other_y, other_weight = fitting_weights(hidden, train, True)
        np.testing.assert_array_equal(y, other_y)
        np.testing.assert_array_equal(weight, other_weight)
        for team in range(3):
            self.assertAlmostEqual(weight[y == team].sum(), len(y)/3)
        _, plain = fitting_weights(labels, train, False)
        self.assertFalse(np.array_equal(weight, plain))
        with self.assertRaises(ValueError):
            fitting_weights(labels, train[::-1], True)


if __name__ == '__main__':
    unittest.main()
