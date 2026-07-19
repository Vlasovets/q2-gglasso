"""Tests for the summarize visualizer, focusing on single-fit solutions.

A single fit (n_lambda1 == n_mu1 == 1, solved via P.solve()) writes no
"modelselect_stats" group. `_make_stats` must handle that without raising,
falling back to a single-model summary instead of a model-selection table.
"""
import os
import unittest

import numpy as np
import zarr

try:
    from q2_gglasso._summarize._visualizer import _make_stats
except ImportError:
    raise ImportWarning("Qiime2 not installed.")


class TestSummarizeSingleFit(unittest.TestCase):
    def _single_fit_solution(self, p=4):
        """In-memory zarr group mimicking a single-fit solution: a precision
        matrix but no modelselect_stats group."""
        root = zarr.group()
        prec = np.eye(p) + 0.1 * (np.ones((p, p)) - np.eye(p))
        root.create_dataset("solution/precision_", data=prec)
        return root, p

    def test_no_modelselect_group_present(self):
        sol, _ = self._single_fit_solution()
        self.assertNotIn("modelselect_stats", sol)

    def test_make_stats_single_fit_does_not_raise(self):
        sol, p = self._single_fit_solution()
        labels = {i: f"ASV-{i + 1}" for i in range(p)}
        layout = _make_stats(sol, labels_dict=labels)
        self.assertIsNotNone(layout)

    def test_make_stats_multi_fit_regression(self):
        """If the packaged multi-fit fixture is available, ensure the
        model-selection path still summarizes (no regression)."""
        fixture = os.path.join(os.path.dirname(__file__), "data", "test_solution.zip")
        if not os.path.exists(fixture):
            self.skipTest("multi-fit fixture not available")
        sol = zarr.open(fixture, mode="r")
        if "modelselect_stats" not in sol:
            self.skipTest("fixture has no modelselect_stats group")
        p = np.asarray(sol["solution/precision_"]).shape[0]
        labels = {i: f"ASV-{i + 1}" for i in range(p)}
        layout = _make_stats(sol, labels_dict=labels)
        self.assertIsNotNone(layout)


if __name__ == "__main__":
    unittest.main()
