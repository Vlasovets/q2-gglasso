"""Tests for heatmap visualization functionality in q2-gglasso.

This module tests the heatmap generation and visualization functions.
"""

import unittest
import numpy as np
import pandas as pd
import zarr
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN

try:
    from q2_gglasso._summarize._visualizer import (
        _make_heatmap,
        _get_order,
        _get_labels,
    )
except ImportError:
    raise ImportWarning("QIIME 2 not installed.")


class TestHeatmap(unittest.TestCase):
    """Test cases for heatmap generation functionality."""

    def test_heatmap_generation(self):
        """Test that heatmaps are correctly generated."""
        sol = zarr.load(store="data/test_solution.zip")
        S = pd.DataFrame(sol["covariance"])

        labels_dict, _ = _get_labels(solution=sol, clustered=False)

        asv_part = S.iloc[:-4, :-4]
        clust_order = _get_order(asv_part, method="average", metric="euclidean")

        re_asv_part = asv_part.iloc[clust_order, clust_order]
        cov_asv_part = S.iloc[:-4, -4:].iloc[clust_order, :]
        cov_part = S.iloc[-4:, -4:]

        res = np.block(
            [
                [re_asv_part.values, cov_asv_part.values],
                [cov_asv_part.T.values, cov_part.values],
            ]
        )

        labels_dict_ordered = {i: labels_dict[key] for i, key in enumerate(clust_order)}
        labels_covs = {k + len(clust_order): v for k, v in enumerate(cov_part.columns)}
        labels_dict_ordered.update(labels_covs)
        labels_dict_reversed = {
            len(labels_dict_ordered) - 1 - k: v for k, v in labels_dict_ordered.items()
        }

        labels = list(re_asv_part.columns) + list(cov_part.columns)
        re_data = pd.DataFrame(res, index=labels, columns=labels)

        # The _make_heatmap function now handles string conversion internally
        heatmap_html = _make_heatmap(
            data=re_data,
            title="Sample covariance",
            width=500,
            height=50,
            label_size="25pt",
            labels_dict=labels_dict_ordered,
            labels_dict_reversed=labels_dict_reversed,
        )

        html = file_html(heatmap_html, CDN, "test heatmap")

        self.assertIsInstance(heatmap_html, figure)
        self.assertIn("<div", html)
        self.assertGreater(len(html), 1000)


if __name__ == "__main__":
    unittest.main()
