import unittest

import numpy as np

from graph_conditional import refine_source_extent


class GraphConditionalTests(unittest.TestCase):
    def fixture(self):
        reference = np.ones(200, int)
        reference[30:150] = 0
        reference[170:] = 2
        baseline = np.ones(200, int)
        baseline[170:] = 2
        p = np.repeat([[.01, .99]], 200, axis=0)
        p[70:110] = [.99, .01]
        return reference, baseline, p, np.ones(200, bool)

    def test_shorten_added_component_and_preserve_D_and_original_labels(self):
        reference, baseline, p, observed = self.fixture()
        result = refine_source_extent(reference, baseline, p, observed)
        self.assertLess((result == 0).sum(), (reference == 0).sum())
        self.assertGreaterEqual((result == 0).sum(), 8)
        self.assertTrue((result[80:100] == 0).all())
        np.testing.assert_array_equal(result == 2, reference == 2)
        np.testing.assert_array_equal(result[reference == baseline], reference[reference == baseline])

    def test_missing_observations_or_no_added_component_leave_reference(self):
        reference, baseline, p, observed = self.fixture()
        observed[:] = False
        np.testing.assert_array_equal(refine_source_extent(reference, baseline, p, observed), reference)
        observed[:] = True
        np.testing.assert_array_equal(refine_source_extent(reference, reference, p, observed), reference)

    def test_team_swap_invariance(self):
        reference, baseline, p, observed = self.fixture()
        swap = lambda value:np.where(value == 2, 2, 1-value)
        expected = swap(refine_source_extent(reference, baseline, p, observed))
        np.testing.assert_array_equal(refine_source_extent(swap(reference), swap(baseline), p[:, ::-1], observed), expected)


if __name__ == '__main__':
    unittest.main()
