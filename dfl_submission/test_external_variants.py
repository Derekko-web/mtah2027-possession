"""Regression checks for sequence decoding, team symmetry, and external resampling."""

from itertools import product
import unittest

import numpy as np
import pandas as pd

from dfl_submission import models  # adds the shared experiments modules to the import path
from dfl_submission.path_decoder import binary_path
from dfl_submission.robust_features import extras, invariant
from dfl_submission.skillcorner import at_25hz
from dfl_transfer_features import swap


class ExternalVariantTests(unittest.TestCase):
    def test_binary_decoder_matches_exhaustive_search(self):
        evidence = np.array([-1.0, 2.0, 1.5, -3.0, .2, -1.0])
        for cost in (0, .5, 2, 5):
            def value(path):
                return evidence @ path - cost * np.count_nonzero(path[1:] != path[:-1])

            best = max(value(np.asarray(path)) for path in product((0, 1), repeat=len(evidence)))
            self.assertAlmostEqual(value(binary_path(evidence, cost)), best)

    def test_features_swap_exactly_with_team_metadata(self):
        rng = np.random.default_rng(12)
        positions = rng.normal(size=(120, 6, 2))
        ball = rng.normal(size=(120, 2))
        ball[30:40] = np.nan
        teams = np.repeat([0, 1], 3)
        original = extras(positions, ball, teams, [105, 68])
        mirrored = extras(positions, ball, 1 - teams, [105, 68])
        pd.testing.assert_frame_equal(swap(original), mirrored)
        pd.testing.assert_frame_equal(invariant(original), invariant(mirrored))

    def test_resampling_keeps_missing_tracking_missing(self):
        xy = np.array([[0, 0], [1, 1], [np.nan, np.nan], [3, 3]])
        points, labels = at_25hz(xy, np.array(["H", "A", "D", "H"]))
        np.testing.assert_array_equal(labels, list("HHHAADDD"))
        np.testing.assert_allclose(points[:3, 0], [0, .4, .8])
        self.assertTrue(np.isnan(points[3:]).all())
