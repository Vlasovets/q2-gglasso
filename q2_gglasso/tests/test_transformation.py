"""Tests for the data transformation utilities in the q2-gglasso plugin.

This module tests the data transformation functionalities of the q2-gglasso plugin,
focusing on the zero imputation transformation for compositional data analysis.
"""

import unittest
import numpy as np
from biom.table import Table

try:
    from q2_gglasso.utils import zero_imputation

except ImportError:
    raise ImportWarning("Qiime2 not installed.")


class TestUtil(unittest.TestCase):
    """Test case for the data transformation utilities.

    This test class verifies that the data transformation functions in the
    q2-gglasso plugin work correctly, particularly focusing on zero imputation.
    """

    def test_zero_imputation(self, equal=False):
        """Test the zero imputation function for compositional data.

        This test verifies that the zero_imputation function maintains
        subcomposition coherence by checking that the column sums remain
        unchanged after imputation.

        Parameters
        ----------
        equal : bool, default=False
            A flag that gets set to True if the column sums before and
            after imputation are equal.

        """
        data = np.array([[0, 0.1], [0.5, 0.1], [0.3, 0], [0.2, 0.8]])
        sample_ids = ["S%d" % i for i in range(2)]
        observ_ids = ["O%d" % i for i in range(4)]
        sample_metadata = [{"environment": "A"}, {"environment": "B"}]
        observ_metadata = [
            {"taxonomy": ["Bacteria", "Firmicutes"]},
            {"taxonomy": ["Bacteria", "Firmicutes"]},
            {"taxonomy": ["Bacteria", "Proteobacteria"]},
            {"taxonomy": ["Bacteria", "Proteobacteria"]},
        ]
        table = Table(
            data,
            observ_ids,
            sample_ids,
            observ_metadata,
            sample_metadata,
            table_id="Example Table",
        )

        X = table.to_dataframe()
        X = X.sparse.to_dense()

        original_sum = X.sum(axis=0)
        X = zero_imputation(X)
        imputeted_sum = X.sum(axis=0)

        if (original_sum == imputeted_sum).all():
            equal = True

        self.assertTrue(equal, msg="Subcomposition coherence not fulfilled.")


class TestRenameIndex(unittest.TestCase):
    """Verify feature ordering / labelling in rename_index_with_sum."""

    def _frame(self):
        import pandas as pd

        # total abundance per feature: hashA=15, hashB=1, hashC=6
        # -> ascending order is hashB, hashC, hashA (differs from input order)
        return pd.DataFrame(
            {"s1": [7, 1, 3], "s2": [8, 0, 3]},
            index=["hashA", "hashB", "hashC"],
        )

    def test_keeps_original_ids_by_default(self):
        from q2_gglasso.utils import rename_index_with_sum

        out = rename_index_with_sum(self._frame())
        # original identifiers preserved, ordered by ascending total abundance
        self.assertEqual(list(out.index), ["hashB", "hashC", "hashA"])

    def test_legacy_asv_labels_when_disabled(self):
        from q2_gglasso.utils import rename_index_with_sum

        out = rename_index_with_sum(self._frame(), keep_original_id=False)
        self.assertEqual(list(out.index), ["ASV-1", "ASV-2", "ASV-3"])


if __name__ == "__main__":
    unittest.main()
