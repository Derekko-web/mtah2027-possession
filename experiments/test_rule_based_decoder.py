import unittest

import numpy as np

from rule_based_decoder import decode


class RuleDecoderTests(unittest.TestCase):
    def test_loose_ball_and_occlusion_keep_the_last_owner_until_control(self):
        ball = np.column_stack((np.arange(170) * .15, np.zeros(170)))
        players = np.stack((ball + [0, 5], ball + [0, -5]), axis=1)
        players[:30, 0] = ball[:30] + [0, .3]
        players[40, 1] = ball[40]
        players[100:, 1] = ball[100:] + [0, -.3]
        ball[50:70] = np.nan

        result = decode(players, [0, 1], ball, [105, 68])
        self.assertTrue(np.all(result[:100] == 'H'))
        self.assertTrue(np.all(result[100:] == 'A'))
        swapped = decode(players, [1, 0], ball, [105, 68])
        np.testing.assert_array_equal(swapped, np.where(result == 'H', 'A', 'H'))

    def test_stationary_ball_marks_a_stop_and_restart_can_change_owner(self):
        distance = np.concatenate((np.arange(50), np.full(75, 50), np.arange(50, 100))) * .15
        ball = np.column_stack((distance, np.zeros(len(distance))))
        players = np.stack((ball + [0, .2], ball + [0, 5]), axis=1)
        players[125:, 0] = ball[125:] + [0, 5]
        players[125:, 1] = ball[125:] + [0, .2]

        result = decode(players, [0, 1], ball, [105, 68])
        self.assertTrue(np.all(result[:35] == 'H'))
        self.assertTrue(np.all(result[65:110] == 'D'))
        self.assertTrue(np.all(result[140:] == 'A'))

    def test_moving_ball_past_the_line_marks_dead(self):
        ball = np.column_stack((np.linspace(45, 56, 60), np.zeros(60)))
        players = np.stack((ball + [0, .3], ball + [0, 5]), axis=1)
        result = decode(players, [0, 1], ball, [105, 68])
        self.assertTrue(np.all(result[:35] == 'H'))
        self.assertTrue(np.all(result[45:] == 'D'))

    def test_no_observations_remain_unknown(self):
        ball = np.full((3, 2), np.nan)
        players = np.zeros((3, 2, 2))
        np.testing.assert_array_equal(decode(players, [0, 1], ball, [105, 68]), ['X'] * 3)


if __name__ == '__main__':
    unittest.main()
