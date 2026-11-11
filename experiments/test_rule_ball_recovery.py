import unittest

import numpy as np

from rule_ball_recovery import recover, recover_possession
from rule_based_decoder import decode_evidence, tracking_evidence


class BallRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.pitch = [105, 68]
        self.ball = np.column_stack((np.arange(100) * .16, np.zeros(100)))
        self.live = np.ones(len(self.ball), dtype=bool)

    def test_interpolation_needs_nearby_in_pitch_endpoints_in_the_same_live_run(self):
        observed = self.ball.copy()
        observed[:2] = observed[8:12] = observed[24:29] = observed[42:50] = np.nan
        observed[69] = [54, 0]
        observed[70:72] = np.nan
        self.live[26] = False
        fused, counts = recover(observed, None, self.pitch, self.live, fps=25, variant='interpolate')
        expected = observed.copy()
        expected[8:12] = self.ball[8:12]
        np.testing.assert_allclose(fused, expected)
        self.assertEqual(counts, dict(interpolate=4, broadcast=0))

    def test_broadcast_uses_agreeing_observations_without_overwriting_tactical(self):
        observed = self.ball.copy()
        observed[10:20] = np.nan
        donor = self.ball + [.25, 0]
        fused, counts = recover(observed, donor, self.pitch, self.live, fps=25, variant='broadcast')
        np.testing.assert_array_equal(fused[:10], observed[:10])
        np.testing.assert_array_equal(fused[10:20], donor[10:20])
        np.testing.assert_array_equal(fused[20:], observed[20:])
        self.assertEqual(counts, dict(interpolate=0, broadcast=10))

    def test_broadcast_rejects_cut_and_disagreement(self):
        observed = self.ball.copy()
        observed[10:20] = np.nan
        for donor in (self.ball + [4, 0], self.ball.copy()):
            donor[10:20] += [12, 0]
            fused, counts = recover(observed, donor, self.pitch, self.live, fps=25, variant='broadcast')
            np.testing.assert_array_equal(fused, observed)
            self.assertEqual(counts['broadcast'], 0)

    def test_recovered_contact_can_switch_owner_but_leaves_stoppages_intact(self):
        observed = self.ball.copy()
        observed[30:60] = np.nan
        players = np.stack((self.ball + [0, 4], self.ball + [0, -4]), axis=1)
        players[:24, 0] = self.ball[:24] + [0, .3]
        players[32:54, 1] = self.ball[32:54] + [0, -.3]
        evidence = tracking_evidence(players, [0, 1], observed, self.pitch)
        reference = decode_evidence(evidence)
        reference[80:90] = 'D'
        labels, _ = recover_possession(players, [0, 1], observed, self.ball, self.pitch, evidence,
                                       reference, variant='broadcast')
        self.assertTrue(np.all(reference[35:54] == 'H'))
        self.assertTrue(np.all(labels[35:54] == 'A'))
        np.testing.assert_array_equal(labels == 'D', reference == 'D')


if __name__ == '__main__':
    unittest.main()
