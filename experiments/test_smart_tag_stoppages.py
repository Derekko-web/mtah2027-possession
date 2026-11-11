import unittest

import numpy as np

from smart_tag_stoppages import refine, restart_windows


PITCH = (105, 68)


class SmartTagStoppageTests(unittest.TestCase):
    def test_touchline_exit_pairs_with_throw_in_across_missing_ball_frames(self):
        ball = np.full((640, 2), np.nan)
        ball[:120] = np.column_stack((np.linspace(-30, 0, 120), np.zeros(120)))
        ball[120:150] = np.column_stack((np.zeros(30), np.linspace(0, 35, 30)))
        ball[150:155] = [0, 35]
        ball[390:400] = [0, 34.2]
        ball[400:450] = np.column_stack((np.zeros(50), np.linspace(34.2, 20, 50)))
        ball[450:] = [0, 20]
        baseline = np.full(len(ball), 'H')
        result, events = refine(baseline, ball, PITCH, [(370, 430, 'throw-in')])

        self.assertEqual(events[0]['kind'], 'exit_and_tag')
        self.assertTrue(events[0]['motion_confirmed'])
        self.assertTrue(np.all(result[155:389] == 'D'))
        self.assertTrue(np.all(result[415:] == 'H'))
        self.assertTrue(np.all(result[:140] == 'H'))
        self.assertTrue(np.all(baseline == 'H'))

    def test_tag_only_stoppage_respects_preceding_live_motion_and_bounded_window(self):
        ball = np.full((500, 2), np.nan)
        ball[:220] = np.column_stack((np.linspace(-35, 35, 220), np.zeros(220)))
        baseline = np.full(len(ball), 'A')
        result, events = refine(baseline, ball, PITCH, [(300, 400, 'free kick')])

        self.assertEqual(events[0]['kind'], 'restart_tag')
        self.assertFalse(events[0]['motion_confirmed'])
        self.assertTrue(np.all(result[:200] == 'A'))
        self.assertTrue(np.all(result[230:350] == 'D'))
        self.assertTrue(np.all(result[350:] == 'A'))

    def test_unconfirmed_tag_during_continuous_play_and_isolated_exit_are_ignored(self):
        ball = np.column_stack((np.linspace(-35, 35, 400), np.zeros(400)))
        ball[50, 1] = 35
        baseline = np.full(len(ball), 'H')
        result, events = refine(baseline, ball, PITCH, [(120, 200, 'corner kick')])

        np.testing.assert_array_equal(result, baseline)
        self.assertEqual(events, [])

    def test_untagged_exit_holds_through_a_bounded_gap(self):
        ball = np.full((450, 2), np.nan)
        ball[:100] = np.column_stack((np.zeros(100), np.linspace(0, 35, 100)))
        baseline = np.full(len(ball), 'A')
        baseline[300] = 'D'
        result, events = refine(baseline, ball, PITCH, [])

        self.assertEqual(events[0]['kind'], 'pitch_exit')
        self.assertEqual(events[0]['end_exclusive'], 250)
        self.assertTrue(np.all(result[100:250] == 'D'))
        self.assertTrue(np.all(result[250:300] == 'A'))
        self.assertEqual(result[300], 'D')

    def test_restart_windows_clip_to_half_and_filter_non_restart_tags(self):
        def event(code, kind, start, end):
            return dict(code=code, tags={'Type': kind}, start_frame=start, end_frame=end)

        events = [event('SET PIECES', 'Throw-in', 90, 120),
                  event('BUILD UP', 'Free Kick', 140, 170),
                  event('SET PIECES', 'Goal Kick', 190, 215)]
        self.assertEqual(restart_windows(events, 100, 200), [(0, 20, 'throw-in'), (90, 100, 'goal kick')])


if __name__ == '__main__':
    unittest.main()
