"""Tests for the heatmap visualization functionality in the q2-gglasso plugin.

This module tests the heatmap creation and clustering functionality of the q2-gglasso
plugin, verifying that the heatmap generation, labeling, and clustering functions 
work as expected.
"""

import unittest
import numpy as np
import pandas as pd
import zarr
from tempfile import TemporaryDirectory

try:
    from q2_gglasso._summarize._visualizer import (_make_heatmap, _get_order, _get_labels,
                                                  hierarchical_clustering)
    from gglasso.helper.data_generation import generate_precision_matrix, sample_covariance_matrix
except ImportError:
    raise ImportWarning('Qiime2 or GGLasso not installed.')


class TestUtil(unittest.TestCase):
    """Test case for heatmap visualization functionality."""

    def setUp(self):
        """Set up test data."""
        # Create test data
        self.n_features = 10
        self.n_covariates = 2
        np.random.seed(42)
        
        # Create a symmetric covariance matrix
        cov_matrix = np.random.randn(self.n_features, self.n_features)
        cov_matrix = cov_matrix @ cov_matrix.T  # Make it symmetric
        
        # Create synthetic data that matches expected format
        p = self.n_features
        M = 1  # Single matrix for this test
        N = 10  # Sample size
        
        # Generate labels with ASVs and covariates
        self.labels = [f'ASV_{i}' for i in range(self.n_features - self.n_covariates)]
        self.labels.extend([f'Cov_{i}' for i in range(self.n_covariates)])
        
        # Create DataFrame with test data
        self.test_data = pd.DataFrame(
            cov_matrix, 
            columns=self.labels, 
            index=self.labels
        )
        
    def test_heatmap_components(self):
        """Test individual components of heatmap visualization."""
        with TemporaryDirectory() as temp_dir:
            # Create a temporary Zarr store with test data
            store = zarr.DirectoryStore(f"{temp_dir}/test.zarr")
            root = zarr.group(store=store)
            
            # Store test data and labels
            root.create_dataset('covariance', data=self.test_data.values)
            root.attrs['labels'] = self.labels
            
            # Test label dictionary generation
            labels_dict, labels_dict_reversed = _get_labels(solution=root, clustered=False)
            self.assertEqual(len(labels_dict), self.n_features)
            self.assertEqual(len(labels_dict_reversed), self.n_features)
            
            # Test clustering order generation (ASVs only)
            asv_data = self.test_data.iloc[:-self.n_covariates, :-self.n_covariates]
            clust_order = _get_order(asv_data, method='average', metric='euclidean')
            self.assertEqual(len(clust_order), self.n_features - self.n_covariates)
            
            # Test hierarchical clustering
            clustered = hierarchical_clustering(
                data=self.test_data,
                clust_order=clust_order,
                n_covariates=self.n_covariates
            )
            self.assertEqual(clustered.shape, self.test_data.shape)
            
            # Test heatmap visualization
            # Convert matrix to long format for heatmap
            df = pd.DataFrame(self.test_data.stack(), columns=['covariance']).reset_index()
            df.columns = ["taxa_y", "taxa_x", "covariance"]
            df = df.replace({"taxa_x": labels_dict, "taxa_y": labels_dict})
            
            plot = _make_heatmap(
                data=df,
                title="Test covariance matrix",
                width=500,
                height=500,
                label_size="10pt",
                labels_dict=labels_dict,
                labels_dict_reversed=labels_dict_reversed
            )
            self.assertIsNotNone(plot)


if __name__ == '__main__':
    unittest.main()
