import unittest

import numpy as np

from boundary_risk import (
    PROBABILITY_FLOOR, boundary_posterior, conditional_live_probability,
    decode_boundary_risk, expected_boundary_utilities, minimum_risk_boundary,
)


def direct_posterior(probability, left_owner):
    weights = np.array([
        np.prod(probability[:boundary, left_owner])
        * np.prod(probability[boundary:, 1-left_owner])
        for boundary in range(1, len(probability))
    ])
    return weights/weights.sum()


def direct_utility(posterior):
    """Independent enumeration using integer intersection/union comparisons."""
    length = len(posterior)+1
    result = []
    for boundary in range(1, length):
        total = 0.
        for true_boundary, probability in enumerate(posterior, 1):
            for numerator in (3, 4, 5, 6, 7):
                total += probability * (
                    10*min(boundary, true_boundary)
                    >= numerator*max(boundary, true_boundary))
                total += probability * (
                    10*min(length-boundary, length-true_boundary)
                    >= numerator*max(length-boundary, length-true_boundary))
        result.append(total/10.)
    return np.array(result)


def transcript(labels):
    return labels[np.r_[True, labels[1:] != labels[:-1]]] if len(labels) else labels


class BoundaryRiskTests(unittest.TestCase):
    def test_posterior_and_cdf_utility_against_direct_enumeration(self):
        random = np.random.default_rng(20261028)
        for length in range(2, 13):
            for _ in range(20):
                raw = random.dirichlet([1., 1., 1.], size=length)
                live = conditional_live_probability(raw)
                for owner in (0, 1):
                    posterior = boundary_posterior(live, owner)
                    np.testing.assert_allclose(posterior, direct_posterior(live, owner),
                                               rtol=2e-13, atol=2e-15)
                    utility = expected_boundary_utilities(posterior)
                    direct = direct_utility(posterior)
                    np.testing.assert_allclose(utility, direct, rtol=2e-14, atol=2e-15)
                    chosen = minimum_risk_boundary(posterior, length//2)
                    self.assertAlmostEqual(float(direct[chosen-1]), float(direct.max()),
                                           places=13)

    def test_all_rational_threshold_edges_against_point_masses(self):
        for length in range(2, 24):
            for true_boundary in range(1, length):
                posterior = np.zeros(length-1)
                posterior[true_boundary-1] = 1.
                np.testing.assert_array_equal(expected_boundary_utilities(posterior),
                                              direct_utility(posterior))

    def test_point_mass_plateau_exact_ties_and_outside_anchor(self):
        posterior = np.zeros(9)
        posterior[4] = 1.
        utility = expected_boundary_utilities(posterior)
        np.testing.assert_array_equal(np.flatnonzero(utility == utility.max())+1,
                                      [4, 5, 6])
        for anchor, expected in ((1, 4), (5, 5), (9, 6), (-10, 4), (20, 6)):
            self.assertEqual(minimum_risk_boundary(posterior, anchor), expected)

    def test_equal_distance_tie_uses_smallest_and_near_tie_is_not_a_tie(self):
        posterior = [.5, 0., 0., 0., .25, 0., .25]
        utility = expected_boundary_utilities(posterior)
        np.testing.assert_array_equal(np.flatnonzero(utility == utility.max())+1, [1, 3])
        self.assertEqual(minimum_risk_boundary(posterior, 2), 1)
        almost_tied = [.500000000001, .499999999999]
        utility = expected_boundary_utilities(almost_tied)
        self.assertGreater(utility[0], utility[1])
        self.assertEqual(minimum_risk_boundary(almost_tied, 2), 1)

    def test_uniform_evidence_and_uniform_boundary_utility(self):
        probability = np.full((5, 2), .5)
        posterior = boundary_posterior(probability, 0)
        np.testing.assert_array_equal(posterior, np.full(4, .25))
        utility = expected_boundary_utilities(posterior)
        np.testing.assert_array_equal(utility, direct_utility(posterior))
        winners = np.flatnonzero(utility == utility.max())+1
        for anchor in range(-1, 7):
            expected = min(winners, key=lambda value: (abs(value-anchor), value))
            self.assertEqual(minimum_risk_boundary(posterior, anchor), expected)

    def test_conditional_ignores_dead_mass_and_clips_after_conditioning(self):
        raw = np.array([[.2, .3, .5], [.04, .06, .9], [0., .5, .5], [1., 0., 0.]])
        live = conditional_live_probability(raw)
        np.testing.assert_allclose(live[:2], [[.4, .6], [.4, .6]], rtol=0, atol=1e-15)
        self.assertEqual(PROBABILITY_FLOOR, 1e-12)
        np.testing.assert_array_equal(live[2], np.array([1e-12, 1.])/(1.+1e-12))
        np.testing.assert_array_equal(live[3], np.array([1., 1e-12])/(1.+1e-12))
        np.testing.assert_array_equal(raw[2], [0., .5, .5])

    def test_positive_runs_team_equivariance_and_determinism(self):
        reference = np.array(list("HHHAAHAHHHAAAAHA"))
        random = np.random.default_rng(513)
        raw = random.dirichlet([.7, .7, .7], size=len(reference))
        expected = decode_boundary_risk(reference, raw, [(0, len(reference))])
        np.testing.assert_array_equal(transcript(expected), transcript(reference))
        swapped_reference = np.where(reference == "H", "A", "H")
        swapped = decode_boundary_risk(swapped_reference, raw[:, [1, 0, 2]],
                                       [(0, len(reference))])
        np.testing.assert_array_equal(swapped, np.where(expected == "H", "A", "H"))
        np.testing.assert_array_equal(decode_boundary_risk(reference, raw,
                                      [(0, len(reference))]), expected)
        np.testing.assert_array_equal(reference, list("HHHAAHAHHHAAAAHA"))

    def test_one_pass_uses_updated_left_and_original_tie_anchor(self):
        # The first update changes the second update's left conditioning
        # endpoint. The original reference endpoint remains its tie anchor.
        reference = np.array(list("HAAHHHHHHH"))
        raw = np.tile([.95, .04, .01], (len(reference), 1))
        raw[6:] = [.04, .95, .01]
        live = conditional_live_probability(raw)
        endpoints, original, owners = [0, 1, 3, 10], [0, 1, 3, 10], [0, 1, 0]
        for index in (1, 2):
            left, right = endpoints[index-1], endpoints[index+1]
            posterior = boundary_posterior(live[left:right], owners[index-1])
            endpoints[index] = left+minimum_risk_boundary(posterior, original[index]-left)
        expected = np.array(list("H"*10))
        expected[endpoints[1]:endpoints[2]] = "A"
        actual = decode_boundary_risk(reference, raw, [(0, len(reference))])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(transcript(actual), list("HAH"))

    def test_each_final_boundary_matches_its_chronological_direct_objective(self):
        random = np.random.default_rng(91027)
        for _ in range(50):
            sizes = random.integers(1, 5, size=5)
            original = np.r_[0, np.cumsum(sizes)]
            reference = np.repeat(list("HAHAH"), sizes)
            raw = random.dirichlet([.5, .5, .5], size=len(reference))
            live = conditional_live_probability(raw)
            actual = decode_boundary_risk(reference, raw, [(0, len(reference))])
            final = np.r_[0, np.flatnonzero(actual[1:] != actual[:-1])+1, len(actual)]
            self.assertEqual(len(final), len(original))
            for index in range(1, len(original)-1):
                # At update j, previous boundaries already have their final
                # values and following boundaries still have original values.
                left, right = int(final[index-1]), int(original[index+1])
                owner = (index-1) % 2
                posterior = direct_posterior(live[left:right], owner)
                utility = direct_utility(posterior)
                chosen = int(final[index])-left
                self.assertGreaterEqual(chosen, 1)
                self.assertLess(chosen, right-left)
                self.assertAlmostEqual(utility[chosen-1], utility.max(), places=13)

    def test_nonlive_outside_blocks_and_resets_remain_exact(self):
        reference = np.array(list("HHAA DAAHH XHHAAHHAA".replace(" ", "")))
        raw = np.tile([.1, .8, .1], (len(reference), 1))
        blocks = [(0, 9), (10, 12), (12, 14)]
        actual = decode_boundary_risk(reference, raw, blocks)
        for left, right in ((0, 4), (5, 9), (10, 12), (12, 14)):
            separately = decode_boundary_risk(reference[left:right], raw[left:right],
                                               [(0, right-left)])
            np.testing.assert_array_equal(actual[left:right], separately)
        visited = np.zeros(len(reference), bool)
        for left, right in blocks:
            visited[left:right] = True
        preserved = ~visited | ~np.isin(reference, ["H", "A"])
        np.testing.assert_array_equal(actual[preserved], reference[preserved])
        # Block iteration order cannot couple independently decoded components.
        np.testing.assert_array_equal(decode_boundary_risk(reference, raw, blocks[::-1]),
                                      actual)

    def test_empty_and_single_run_components(self):
        np.testing.assert_array_equal(decode_boundary_risk(np.array([], dtype="<U1"),
                                      np.empty((0, 3)), []), np.array([], dtype="<U1"))
        reference = np.array(list("HHDDXAAAA"))
        raw = np.tile([.1, .8, .1], (len(reference), 1))
        np.testing.assert_array_equal(decode_boundary_risk(reference, raw,
                                      [(0, len(reference))]), reference)

    def test_invalid_reference_probability_and_blocks_rejected(self):
        reference = np.array(list("HHAA"))
        valid = np.tile([.2, .5, .3], (4, 1))
        for raw in (valid[:-1], np.ones((4, 2))/2., np.ones((4, 3)),
                    np.full((4, 3), np.nan), np.full((4, 3), np.inf),
                    np.tile([-.1, .5, .6], (4, 1)), np.tile([0., 0., 1.], (4, 1))):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    decode_boundary_risk(reference, raw, [(0, 4)])
        outside_nan = valid.copy()
        outside_nan[-1] = np.nan
        with self.assertRaises(ValueError):
            decode_boundary_risk(reference, outside_nan, [(0, 2)])
        for bad in (list("HHAQ"), np.array([list("HHAA")])):
            with self.assertRaises(ValueError):
                decode_boundary_risk(bad, valid, [(0, 4)])
        for blocks in ([(0, 3), (2, 4)], [(-1, 4)], [(0, 5)], [(0, 0)],
                       [(0., 4)], [(True, 4)], [(0,)], [None], None):
            with self.subTest(blocks=blocks):
                with self.assertRaises(ValueError):
                    decode_boundary_risk(reference, valid, blocks)

    def test_invalid_local_distributions_rejected(self):
        for posterior in ([], [0., 0.], [np.nan, 1.], [-.1, 1.1], [[1.]]):
            with self.assertRaises(ValueError):
                expected_boundary_utilities(posterior)
        with self.assertRaises(ValueError):
            minimum_risk_boundary([1.], .5)
        for probability, owner in (([[.5, .5]], 0), ([[0., 1.], [1., 0.]], 0),
                                   ([[.5, .5], [.5, .5]], 2)):
            with self.assertRaises(ValueError):
                boundary_posterior(probability, owner)


if __name__ == "__main__":
    unittest.main()
