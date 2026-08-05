"""Regression tests for get_seq_depth's orientation contract.

get_seq_depth used to guess which axis to sum over with `if p >= n`, i.e. it
assumed there were at least as many features as samples. For the 13-feature x
50-sample tutorial table that guess is wrong and it silently returned
per-FEATURE sums, which the pca visualizer then tried to join onto sample
metadata. It now requires samples x features and sums over features.
"""

import unittest

import numpy as np
import pandas as pd

from q2_gglasso.utils import get_seq_depth


def _table(n_samples, n_features, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.integers(1, 100, size=(n_samples, n_features)),
        index=[f"S{i}" for i in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )


class TestGetSeqDepth(unittest.TestCase):
    def test_returns_one_value_per_sample(self):
        df = _table(n_samples=50, n_features=13)
        depth = get_seq_depth(df)
        self.assertEqual(len(depth), 50)
        self.assertListEqual(list(depth.index), list(df.index))

    def test_fewer_features_than_samples(self):
        """The case the old `if p >= n` heuristic got wrong."""
        df = _table(n_samples=50, n_features=13)
        depth = get_seq_depth(df)
        # Must be indexed by sample, not by feature.
        self.assertNotIn("F0", depth.index)
        self.assertIn("S0", depth.index)

    def test_more_features_than_samples(self):
        df = _table(n_samples=10, n_features=300)
        depth = get_seq_depth(df)
        self.assertEqual(len(depth), 10)

    def test_scaled_to_unit_interval(self):
        depth = get_seq_depth(_table(40, 20))
        self.assertAlmostEqual(float(depth.min()), 0.0)
        self.assertAlmostEqual(float(depth.max()), 1.0)

    def test_ordering_follows_raw_totals(self):
        df = _table(30, 12)
        depth = get_seq_depth(df)
        self.assertEqual(depth.idxmax(), df.sum(axis=1).idxmax())
        self.assertEqual(depth.idxmin(), df.sum(axis=1).idxmin())

    def test_constant_depth_does_not_produce_nan(self):
        """Equal row sums used to divide by zero and yield NaN everywhere."""
        df = pd.DataFrame(
            np.ones((5, 4)), index=[f"S{i}" for i in range(5)],
            columns=[f"F{j}" for j in range(4)],
        )
        depth = get_seq_depth(df)
        self.assertFalse(np.isnan(np.asarray(depth, dtype=float)).any())


if __name__ == "__main__":
    unittest.main()
