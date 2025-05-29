import pandas as pd
import numpy as np
import zarr
import os
import itertools
from scipy import stats
import qiime2
from qiime2.plugins import feature_table as ft_plugin
import warnings
from sklearn import preprocessing

from q2_gglasso.utils import PCA, remove_biom_header, calculate_seq_depth, correlated_PC
from gglasso.helper.basic_linalg import adjacency_matrix
from gglasso.helper.basic_linalg import scale_array_by_diagonal

!python setup.py install

!qiime dev refresh-cache

!qiime gglasso summarize \
    --i-solution data/atacama-solution-sgl.qza \
    --p-label-size 25pt \
    --o-visualization data/sgl-summary.qzv


!qiime gglasso summarize \
    --i-solution data/atacama-solution-slr.qza \
    --p-label-size 25pt \
    --o-visualization data/slr-summary.qzv

!qiime gglasso summarize \
    --i-solution data/atacama-solution-adapt.qza \
    --p-label-size 25pt \
    --p-n-cov 4 \
    --o-visualization data/adapt-summary.qzv


def rename_index_with_sum(df: pd.DataFrame):
    """
    Rename the index of a DataFrame based on the relative abundance.
    New index values are generated with the format "ASV" followed by the top abundance among all the features.

    Parameters:
    - df: pandas DataFrame. The DataFrame to be modified.

    Returns:
    - df: pandas DataFrame. The modified DataFrame with the renamed index.
    """

    # Calculate the sum of each row and rename the index
    row_sum = df.sum(axis=1)
    df.rename(index=row_sum, inplace=True)

    # Sort the index
    df.sort_index(inplace=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Rename the index using the new index values
    df.rename(index=lambda x: f"ASV-{x + 1}", inplace=True)

    return df


def normalize(X):
    """
    transforms to the simplex
    X should be of a pd.DataFrame of form (p,N)
    """
    return (X / X.sum(axis=0)).round(12)

def geometric_mean(x, positive=False):
    """
    calculates the geometric mean of a vector
    """
    assert not np.all(x == 0)

    if positive:
        x = x[x > 0]
    a = np.log(x)
    g = np.exp(a.sum() / len(a))
    return g

def log_transform(X, transformation=str, eps=0.1):
    """
    log transform, scaled with geometric mean
    X should be a pd.DataFrame of form (p,N)
    """
    if transformation == "clr":
        assert not np.any(X.values == 0), "Add pseudo count before using clr"
        g = X.apply(geometric_mean)
        Z = np.log(X / g)
    elif transformation == "mclr":
        g = X.apply(geometric_mean, positive=True)
        X_pos = X[X > 0]
        Z = np.log(X_pos / g)
        Z = Z + abs(np.nanmin(Z.values)) + eps
        Z = Z.fillna(0)
    return Z.round(12)


def scale_array_by_diagonal(X, d=None):
    """
    scales a 2d-array X with 1/sqrt(d), i.e.

    X_ij/sqrt(d_i*d_j)
    in matrix notation: W^-1 @ X @ W^-1 with W^2 = diag(d)

    if d = None, use square root diagonal, i.e. W^2 = diag(X)
    see (2.4) in https://fan.princeton.edu/papers/09/Covariance.pdf
    """
    assert len(X.shape) == 2
    if d is None:
        d = np.diag(X)
    else:
        assert len(d) == X.shape[0]

    scale = np.tile(np.sqrt(d), (X.shape[0], 1))
    scale = scale.T * scale

    return X / scale

def _get_order(data: pd.DataFrame, method: str = 'average', metric: str = 'euclidean'):
    """
    Performs hierarchical clustering on the input DataFrame and returns the cluster order.

    Args:
        data (pd.DataFrame): The input DataFrame.
        method (str, optional): The clustering method. Defaults to 'average'.
        metric (str, optional): The distance metric. Defaults to 'euclidean'.

    Returns:
        list: The cluster order as a list of indices.
    """
    grid = sns.clustermap(data, method=method, metric=metric, robust=True)
    plt.close()
    clust_order = grid.dendrogram_row.reordered_ind

    return clust_order


def hierarchical_clustering(data: pd.DataFrame, clust_order: list, n_covariates: int = None):
    if n_covariates is None:
        re_data = data.iloc[clust_order, clust_order]

    else:
        asv_part = data.iloc[:-n_covariates, :-n_covariates]
        re_asv_part = asv_part.iloc[clust_order, clust_order]
        cov_asv_part = data.iloc[:-n_covariates, -n_covariates:].iloc[clust_order, :]
        cov_part = data.iloc[-n_covariates:, -n_covariates:]

        res = np.block([[re_asv_part.values, cov_asv_part.values],
                        [cov_asv_part.T.values, cov_part.values]])

        labels = list(re_asv_part.columns) + list(cov_part.columns)
        re_data = pd.DataFrame(res, index=labels, columns=labels)

    return re_data

counts = pd.read_csv("data/org_named_counts_T.tsv", sep="\t", index_col=0)
# this method transpose data because the dataframe is assumed to N,p shape
table_artifact = qiime2.Artifact.import_data('FeatureTable[Frequency]', counts)
table_artifact.save('data/pandas-counts.qza')

mclr = pd.read_csv("data/atacama-mclr/feature-table.tsv", sep="\t", index_col=0)

corr = pd.read_csv("data/corr_table.tsv", sep="\t", index_col=0)
corr_artifact = qiime2.Artifact.import_data('FeatureTable[Frequency]', corr)
corr_artifact.save('data/pandas-corr.qza')

!qiime tools export \
  --input-path data/atacama-table-mclr.qza \
  --output-path data/atacama-table-mclr
#
!biom convert -i data/atacama-table-mclr/feature-table.biom -o data/atacama-table-mclr/feature-table.tsv \
                                                          --to-tsv
#remove biom header from the files
remove_biom_header(file_path="data/atacama-table-mclr/feature-table.tsv")

q2_ggasso_mclr = pd.read_csv("data/atacama-table-mclr/feature-table.tsv", sep="\t",
                          index_col=0)


!qiime tools export \
  --input-path data/q2-classo-mclr_count_table.qza \
  --output-path data/q2-classo-mclr_count_table
#
!biom convert -i data/q2-classo-mclr_count_table/feature-table.biom -odata/q2-classo-mclr_count_table/feature-table.tsv \
                                                          --to-tsv
#remove biom header from the files
remove_biom_header(file_path="data/q2-classo-mclr_count_table/feature-table.tsv")

q2_classo_mclr = pd.read_csv("data/q2-classo-mclr_count_table/feature-table.tsv", sep="\t",
                          index_col=0)

corr = pd.read_csv("data/atacama-table-corr/pairwise_comparisons.tsv", sep="\t", index_col=0)

import biom
from biom import load_table
import seaborn as sns
from matplotlib import pyplot as plt

def _get_labels(solution: pd.DataFrame(), clustered: bool = False, clust_order: list = None):
    labels_dict = dict()
    labels_dict_reversed = dict()
    p = solution.shape[0]
    # p = np.array(solution['p']).item()
    for i in range(0, p):
        labels_dict[i] = solution.index[i]
        labels_dict_reversed[p - 1] = solution.index[i]
        p -= 1

    return labels_dict, labels_dict_reversed


biom_file_path = 'data/pandas-counts/feature-table.biom'

# Load the Biom table
biom_table = load_table(biom_file_path)

X = biom_table.to_dataframe()
X = X.sparse.to_dense()
X = rename_index_with_sum(X)

X = normalize(X)
X = log_transform(X, transformation="mclr")

S = np.cov(X.values, bias=True)

S_scaled = scale_array_by_diagonal(S)
X = pd.DataFrame(S_scaled, index=X.index, columns=X.index).round(10)

clust_order = _get_order(X, method='average', metric='euclidean')

sample_covariance = hierarchical_clustering(X, clust_order=clust_order, n_covariates=None)

labels_dict, labels_dict_reversed = _get_labels(sample_covariance)

labels_dict = {key: labels_dict[value] for key, value in enumerate(clust_order)}
labels_dict_reversed = {len(labels_dict) - 1 - key: value for key, value in labels_dict.items()}

df = pd.DataFrame(sample_covariance.stack(), columns=['covariance']).reset_index()
df.columns = ["taxa_y", "taxa_x", "covariance"]
df = df.replace({"taxa_x": labels_dict, "taxa_y": labels_dict})

X.to_csv('data/pandas-counts/q2_corr.tsv', sep="\t", index=True)

mclr = pd.read_csv("data/mclr_T.tsv", sep="\t", index_col=0)
np.array_equal(X.values, q2_mclr.values)


!qiime tools export \
  --input-path data/atacama-solution-sgl.qza \
  --output-path data/solution-sgl

sol = zarr.load(store="data/solution-sgl/problem.zip")

sample_covariance = pd.DataFrame(sol['covariance'])

clust_order = _get_order(sample_covariance, method='average', metric='euclidean')

sample_covariance = sample_covariance.iloc[clust_order, clust_order]

df = pd.DataFrame(sample_covariance.stack(), columns=['covariance']).reset_index()
df.columns = ["taxa_y", "taxa_x", "covariance"]
df = df.replace({"taxa_x": labels_dict, "taxa_y": labels_dict})


!qiime gglasso transform-features \
     --p-transformation mclr \
     --p-add-metadata False \
     --p-scale-metadata False \
     --i-table data/pandas-counts.qza \
     --i-taxonomy data/classification.qza \
     --m-sample-metadata-file data/selected-atacama-sample-metadata.tsv \
     --o-transformed-table data/atacama-table-mclr.qza \
     --verbose


!qiime gglasso calculate-covariance \
     --p-method scaled \
     --i-table data/atacama-table-mclr.qza \
     --o-covariance-matrix data/atacama-table-corr.qza

### SGL
!qiime gglasso solve-problem \
     --p-n-samples 50 \
     --p-lambda1-min 0.001 \
     --p-lambda1-max 1 \
     --p-n-lambda1 50 \
     --p-gamma 0.01 \
     --p-latent False \
     --i-covariance-matrix data/atacama-table-corr.qza \
     --o-solution data/atacama-solution-sgl.qza \
     --verbose


!qiime gglasso summarize \
    --i-solution data/atacama-solution-sgl.qza \
    --p-label-size 25pt \
    --o-visualization data/sgl-summary.qzv


### SLR
!qiime gglasso solve-problem \
     --p-n-samples 50 \
     --p-lambda1-min 0.001 \
     --p-lambda1-max 1 \
     --p-mu1-min 0.50118723 \
     --p-mu1-max 0.79432823 \
     --p-n-lambda1 50 \
     --p-n-mu1 50 \
     --p-gamma 0.01 \
     --p-latent True \
     --i-covariance-matrix data/atacama-table-corr.qza \
     --o-solution data/atacama-solution-slr.qza \
     --verbose

!qiime gglasso summarize \
    --i-solution data/atacama-solution-slr.qza \
    --p-label-size 25pt \
    --o-visualization data/slr-summary.qzv


# Adaptive
!qiime gglasso transform-features \
     --p-transformation mclr \
     --p-add-metadata True \
     --i-table data/pandas-counts.qza \
     --i-taxonomy data/classification.qza \
     --m-sample-metadata-file data/atacama-selected-covariates.tsv \
     --o-transformed-table data/atacama-table-mclr-meta.qza \
     --verbose


!qiime gglasso calculate-covariance \
     --p-method scaled \
     --i-table data/atacama-table-mclr-meta.qza \
     --o-covariance-matrix data/atacama-table-corr-meta.qza


!qiime gglasso solve-problem \
     --p-n-samples 50 \
     --p-lambda1-min 0.001 \
     --p-lambda1-max 1 \
     --p-n-lambda1 2 \
     --p-gamma 0.01 \
     --p-latent False \
     --p-adapt-lambda1 elevation 0.01 ph 0.01 average-soil-relative-humidity 0.01 average-soil-temperature 0.01  \
     --i-covariance-matrix data/atacama-table-corr-meta.qza \
     --o-solution data/atacama-solution-adapt.qza \
     --verbose


!qiime gglasso summarize \
    --i-solution data/atacama-solution-adapt.qza \
    --p-label-size 25pt \
    --o-visualization data/adapt-summary.qzv