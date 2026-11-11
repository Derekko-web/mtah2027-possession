import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extratrees_latent import fit_extra, predict_extra, training_record


class ExtraTreesLatentTests(unittest.TestCase):
    def test_train_only_imputer_and_exact_model_reload(self):
        inputs = pd.DataFrame(dict(
            h__x=np.r_[np.arange(12, dtype=float), 100000., 100000.],
            a__x=np.r_[np.arange(12, dtype=float), -100000., -100000.],
            h__missing=np.full(14, np.nan), a__missing=np.full(14, np.nan)))
        labels = np.array(list("HADHADHADHADXX"))
        train, validation = np.arange(12), np.arange(12, 14)
        bundle = fit_extra(inputs, labels, train, validation, 42)
        imputer, model = bundle
        np.testing.assert_array_equal(imputer.statistics_, [5.5, 5.5, 0., 0.])
        np.testing.assert_array_equal(imputer.indicator_.features_, [2, 3])
        self.assertEqual(model.n_jobs, 1)
        expected = predict_extra(bundle, inputs.iloc[validation])
        np.testing.assert_allclose(expected.sum(1), 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(bundle, path)
            np.testing.assert_array_equal(expected, predict_extra(joblib.load(path), inputs.iloc[validation]))

    def test_training_record_ignores_validation_targets(self):
        labels = np.array(list("HADHAD"))
        train, validation = np.arange(3), np.arange(3, 6)
        first = training_record(labels, train, validation)
        labels[validation] = "X"
        second = training_record(labels, train, validation)
        for key, value in first.items():
            np.testing.assert_array_equal(value, second[key])
        np.testing.assert_array_equal(first["target"], [0, 1, 2, 1, 0, 2])
        np.testing.assert_array_equal(first["weight"], np.ones(6))

    def test_invalid_training_indices_or_classes_rejected(self):
        with self.assertRaises(ValueError):
            training_record(np.array(list("HAD")), [0, 1], [1, 2])
        with self.assertRaises(ValueError):
            training_record(np.array(list("HAD")), [0, 1], [2])


if __name__ == "__main__":
    unittest.main()
