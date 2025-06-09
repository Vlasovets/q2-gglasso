"""Tests for the correlation functionality in q2-gglasso.

This module contains tests for covariance and correlation calculation utilities.
"""

import unittest
import pandas as pd
import numpy as np

try:
    from q2_gglasso._func import calculate_covariance

except ImportError:
    raise ImportWarning("Qiime2 not installed.")


class TestUtil(unittest.TestCase):
    """Test suite for correlation utility functions."""

    def test_scaling_to_correlation(self):
        """Test that covariance matrix can be converted to correlation matrix."""
        table = pd.DataFrame(
            [[1, 1, 7, 3], [2, 6, 2, 4], [5, 5, 3, 3], [3, 2, 8, 1]],
            index=["s1", "s2", "s3", "s4"],
            columns=["o1", "o2", "o3", "o4"],
        )

        S = calculate_covariance(
            table, method="scaled", bias=True
        )  # covariance matrix S
        S_values = np.round(S.values, decimals=10)

        self.assertFalse((abs(S_values) > 1).any())


if __name__ == "__main__":
    unittest.main()
