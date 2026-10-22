"""Synthetic checks for nested exclusions, case matching and timing constraints."""

import itertools
import unittest

import numpy as np

from boundary_refinement import runs
from endpoint_calibration import (calibrate_endpoints, contiguous_components, eligible_islands,
                                  estimate_correction, nested_folds, timing_examples)


class EndpointCalibrationTests(unittest.TestCase):
    def test_nested_fitting_and_case_labels_stay_inside_outer_training(self):
        outer = np.r_[np.arange(11185,13900), np.arange(15900,18685)]
        covered = []
        for block, train, validation in nested_folds(outer):
            start = 11185+(block-1)*2500
            self.assertTrue(np.isin(train, outer).all())
            self.assertTrue(np.isin(validation, outer).all())
            self.assertFalse(np.any((train >= start-250) & (train < start+2750)))
            self.assertFalse(np.intersect1d(train, validation).size)
            covered.extend(validation)
        np.testing.assert_array_equal(np.sort(covered), outer)
        self.assertEqual(contiguous_components([1,2,5,6,7]), [(1,3),(5,8)])

    def test_matching_excludes_clipped_truth_and_fits_medians(self):
        predicted = np.r_[np.zeros(20,int), np.ones(20,int), np.zeros(150,int), np.ones(20,int), np.zeros(20,int)]
        truth = np.r_[np.zeros(22,int), np.ones(13,int), np.zeros(157,int), np.ones(13,int), np.zeros(25,int)]
        cases = timing_examples(predicted, truth, 100)
        self.assertEqual(len(cases), 2)
        self.assertEqual([case["end_error"] for case in cases], [-5,-5])
        self.assertEqual(estimate_correction(cases, 2)["end"], -5.)
        self.assertFalse(estimate_correction(cases[:1], 2)["sufficient"])
        self.assertEqual(estimate_correction(cases[:1], 2)["end"], 0.)
        self.assertEqual(estimate_correction([dict(start_error=99,end_error=-99)],1)["start"],25.)
        self.assertEqual(timing_examples(predicted,np.zeros(len(predicted),int)), [])

    def test_endpoint_solution_matches_exhaustive_objective(self):
        states = np.r_[np.zeros(10,int), np.ones(10,int), np.zeros(10,int)]
        correction = dict(start=3.,end=-4.)
        result = calibrate_endpoints(states, correction, True)
        spans = runs(result)
        self.assertEqual([r[2] for r in spans], [0,1,0])
        actual = (spans[0][1]-13)**2+(spans[1][1]-16)**2
        best = min((left-13)**2+(right-16)**2 for left,right in itertools.product(range(1,29),repeat=2)
                   if right-left >= 8)
        self.assertEqual(actual,best)

    def test_adjacency_dead_reset_symmetry_and_noop(self):
        states = np.r_[np.full(9,2),np.zeros(30,int),np.ones(10,int),np.zeros(10,int),np.ones(30,int),np.full(9,2)]
        correction = dict(start=25.,end=-25.)
        for both in (False,True):
            result = calibrate_endpoints(states,correction,both)
            np.testing.assert_array_equal(result==2, states==2)
            self.assertEqual([r[2] for r in runs(result)], [r[2] for r in runs(states)])
            spans,eligible = eligible_islands(states)
            changed = runs(result)
            self.assertTrue(all(changed[i][1]-changed[i][0] >= 8 for i in eligible))
            swapped = np.where(states==2,2,1-states)
            np.testing.assert_array_equal(calibrate_endpoints(swapped,correction,both),np.where(result==2,2,1-result))
        np.testing.assert_array_equal(calibrate_endpoints(states,dict(start=0.,end=0.),True),states)


if __name__ == "__main__":
    unittest.main()
