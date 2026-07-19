"""Tests for regularization-path construction (log/linear/user-defined).

Covers the user-defined and linearly-spaced lambda/mu paths added so that a
custom path (e.g. the linear 1.00, 0.95, ..., 0.10 grid) can be supplied rather
than only a log-spaced min/max/count grid.
"""
import unittest

import numpy as np

try:
    from q2_gglasso.utils import get_range, get_hyperparameters
except ImportError:
    raise ImportWarning("Qiime2 not installed.")


class TestGetRange(unittest.TestCase):
    def test_log_scale_default(self):
        out = get_range(1e-3, 1.0, 4)
        np.testing.assert_allclose(out, [1e-3, 1e-2, 1e-1, 1.0])

    def test_linear_scale(self):
        out = np.round(get_range(1.0, 0.1, 19, scale="linear"), 2)
        self.assertEqual(out[0], 1.0)
        self.assertEqual(out[-1], 0.1)
        self.assertEqual(len(out), 19)

    def test_unknown_scale_raises(self):
        with self.assertRaises(ValueError):
            get_range(0.1, 1.0, 3, scale="quadratic")


class TestGetHyperparameters(unittest.TestCase):
    def test_explicit_lambda_path_overrides_grid(self):
        h = get_hyperparameters(
            lambda1_min=0.001, lambda1_max=1, lambda2_min=None, lambda2_max=None,
            mu1_min=None, mu1_max=None, n_lambda1=50,
            lambda1_path=[1.0, 0.8, 0.6],
        )
        np.testing.assert_allclose(np.asarray(h["lambda1"], dtype=float), [1.0, 0.8, 0.6])
        # multiple values -> model selection
        self.assertTrue(h["model_selection"])

    def test_single_value_path_is_scalar_no_model_selection(self):
        h = get_hyperparameters(
            lambda1_min=None, lambda1_max=None, lambda2_min=None, lambda2_max=None,
            mu1_min=None, mu1_max=None,
            lambda1_path=[0.8],
        )
        self.assertFalse(h["model_selection"])

    def test_linear_path_scale(self):
        h = get_hyperparameters(
            lambda1_min=1.0, lambda1_max=0.1, lambda2_min=None, lambda2_max=None,
            mu1_min=None, mu1_max=None, n_lambda1=19, path_scale="linear",
        )
        lam = np.round(np.asarray(h["lambda1"], dtype=float), 2)
        self.assertEqual(lam[0], 1.0)
        self.assertEqual(lam[-1], 0.1)


if __name__ == "__main__":
    unittest.main()
