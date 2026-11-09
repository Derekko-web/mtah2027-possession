import unittest

import numpy as np
import pandas as pd

from short_control_recovery import recover


class ShortControlRecoveryTests(unittest.TestCase):
    def setUp(self):
        n = 330
        self.reference = np.full(n, 'A')
        self.owner = np.tile([.05, .95], (n, 1))
        self.owner[151:168] = [.85, .15]
        self.owner[150] = [.43, .57]
        self.owner[168] = [.28, .72]
        self.fast = np.tile([.2, .8], (n, 1))
        self.fast[150:169] = [.8, .2]
        self.eligible = np.ones(n, dtype=bool)
        self.raw = pd.DataFrame(dict(
            h__distance_1=np.full(n, 3.), a__distance_1=np.full(n, 1.),
            ball_seen=np.ones(n), ball_goal_line_distance=np.ones(n),
            ball_touchline_distance=np.ones(n),
        ))
        self.broadcast = pd.DataFrame(dict(
            h__bc_same_distance_1=np.full(n, 3.), a__bc_same_distance_1=np.full(n, 1.),
            bc_matched=np.ones(n), bc_same_ball_seen=np.ones(n),
        ))
        self.raw.iloc[162:169, :2] = [1., 3.]
        self.broadcast.iloc[162:169, :2] = [1., 3.]
        self.blocks = [(0, n)]

    def predict(self):
        return recover(self.reference, self.owner, self.fast, self.eligible,
                       self.raw, self.broadcast, self.blocks)[0]

    def test_recovers_short_control_with_supported_edges_for_either_team(self):
        expected = self.reference.copy()
        expected[150:169] = 'H'
        np.testing.assert_array_equal(self.predict(), expected)

        self.reference[:] = 'H'
        self.owner = self.owner[:, ::-1]
        self.fast = self.fast[:, ::-1]
        self.raw.iloc[:, :2] = self.raw.iloc[:, :2].to_numpy()[:, ::-1]
        self.broadcast.iloc[:, :2] = self.broadcast.iloc[:, :2].to_numpy()[:, ::-1]
        np.testing.assert_array_equal(self.predict(), np.where(expected == 'H', 'A', 'H'))

    def test_missing_broadcast_and_fold_edges_prevent_recovery(self):
        self.broadcast.loc[162:168, 'bc_same_ball_seen'] = 0
        np.testing.assert_array_equal(self.predict(), self.reference)

        self.broadcast.loc[162:168, 'bc_same_ball_seen'] = 1
        self.blocks = [(0, 160), (160, len(self.reference))]
        np.testing.assert_array_equal(self.predict(), self.reference)

    def test_existing_control_and_dead_or_ungraded_frames_are_preserved(self):
        self.reference[150:169] = 'H'
        self.reference[200:210] = 'D'
        self.reference[220:230] = 'X'
        np.testing.assert_array_equal(self.predict(), self.reference)


if __name__ == '__main__':
    unittest.main()
