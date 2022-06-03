import zarr
import qiime2
import plotly

import pandas as pd
import numpy as np
import gglasso

from gglasso.helper.utils import zero_replacement, log_transform
from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.problem import glasso_problem


from qiime2.plugin import (
    Str
)

gglasso.__version__

# test = pd.DataFrame({"x": [0,0,0,0,0,0,0,0,0,0,94,85,99,101,92,95,99,101,120,92],
#                      "y": [0,0,0,0,0,6,6,6,8,6,15,15,8,2,8,12,9,8,12,10]})
#
# test_new = np.log(test[test>0] / 4)
# np.nanmin(test_new.values)
#
# test + test_new
#
# geometric_mean(test, positive=True)
#
# a = norm(test)
# log_transform_0(a, transformation="mclr")
#
# X = norm(test)
# X = trans(X)


import numpy as np
from biom.table import Table
data = np.array([[0, 2], [1, 1], [2, 0], [7, 0]])
sample_ids = ['S%d' % i for i in range(2)]
observ_ids = ['O%d' % i for i in range(4)]
sample_metadata = [{'environment': 'A'}, {'environment': 'B'}]
observ_metadata = [{'taxonomy': ['Bacteria', 'Firmicutes']},
               {'taxonomy': ['Bacteria', 'Firmicutes']},
               {'taxonomy': ['Bacteria', 'Proteobacteria']},
                 {'taxonomy': ['Bacteria', 'Proteobacteria']}]
table = Table(data, observ_ids, sample_ids, observ_metadata,
             sample_metadata, table_id='Example Table')

pd.DataFrame(table)

import multiprocessing
from functools import partial
import warnings
from scipy.stats import spearmanr
import numpy as np


def pairwise_iter_wo_metadata(pairwise_iter):
    for (val_i, id_i, _), (val_j, id_j, _) in pairwise_iter:
        yield (val_i, id_i), (val_j, id_j)


def calculate_correlation(data, corr_method=np.cov):
    cov = corr_method(data)
    return cov



def calculate_correlations(table: Table, corr_method=np.cov, nprocs=1) -> pd.DataFrame:
    if nprocs > multiprocessing.cpu_count():
        warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
        nprocs = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(nprocs)
    cor = partial(calculate_correlation, corr_method=corr_method)
    results = pool.map(cor, pairwise_iter_wo_metadata(table.iter_pairwise(axis='observation')))
    index = [i[0] for i in results]
    data = [i[1] for i in results]
    pool.close()
    pool.join()
    correls = pd.DataFrame(data, index=index, columns=['cov', 'p'])
    # Turn tuple index into actual multiindex, now guaranteeing that correls index is sorted
    correls.index = pd.MultiIndex.from_tuples([sorted(i) for i in correls.index])

    return results


from scipy import stats


def calculate_covariance(X: pd.DataFrame, corr_method=np.cov, bias=True) -> pd.DataFrame:

    if corr_method == "unscaled":
        cov = np.cov(X, bias=bias)
        pval = None

    elif corr_method == "spearmanr":
        cov, pval = stats.spearmanr(X, X)

    return cov, pval

np.cov(data)

calculate_covariance(data)


data = np.array([[0, 2], [1, 1], [2, 0], [7, 0]])

genus_df = pd.read_csv('example/data/exported-genus-table/feature-table.tsv', sep='\t')
genus_df.columns


genus_df.iloc[:, 0]

from q2_gglasso._func import to_zarr


# plotly.__version__
# qiime2.__version__
# root = zarr.group()
#
# foo = root.create_group('foo')
# bar = foo.create_group('bar')
#
# z1 = bar.zeros('baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
# bar.ones("oleg", shape=(10, 10))
#
# root.tree()



###Leo's verstion of CLR

def transform_features(
        features: pd.DataFrame, transformation: Str = "clr", coef: float = 0.5
) -> pd.DataFrame:
    if transformation == "clr":
        X = features.values
        null_set = X <= 0.0
        X[null_set] = coef  # zero replacement
        X = np.log(X)
        X = (X.T - np.mean(X, axis=1)).T
        #X = (X - np.mean(X, axis=0))

        return pd.DataFrame(
            data=X, index=list(features.index), columns=list(features.columns)
        )

    else:
        raise ValueError(
            "Unknown transformation name, use clr and not %r" % transformation
        )


df = pd.DataFrame(np.random.randint(1, 10, size=(3, 3)), columns=list('ABC'))
np.log(df.values)

transform_features(pd.DataFrame([[2, 2], [4, 4]]))


# ### Fabian's version of CLR
# df = pd.DataFrame(np.random.randint(0, 10, size=(3, 3)), columns=list('ABC'))
#
def F_transform_features(
        features: pd.DataFrame, transformation: Str = "clr", pseudo: float = 0.5
) -> pd.DataFrame:
    if transformation == "clr":
        X = zero_replacement(features, c=pseudo)
        X = log_transform(X)

        return pd.DataFrame(
            data=X, index=list(features.index), columns=list(features.columns)
        )

    else:
        raise ValueError(
            "Unknown transformation name, use clr and not %r" % transformation
        )

F_transform_features(df)

from skbio.stats.composition import clr
x = np.array([.1, .3, .4, .2])
clr(df)

F_transform_features(df)
transform_features(df)

test = pd.DataFrame(np.random.randint(2, 10, size=(3, 3)), columns=list('ABC'))

transform_features(test)
F_transform_features(test)
clr(test)


#
# transform_features(df)
#
# X = np.array(df)
#
# X = (X - np.mean(X, axis=0))
# X = (X.T - np.mean(X, axis=1)).T
#
# df = pd.DataFrame(np.random.randint(0, 100, size=(10, 4)), columns=list('ABCD'))
#
# df.values


### Synthetic networks

p = 200
N = 10

Sigma, Theta = generate_precision_matrix(p=p, M=1, style='erdos', prob=0.1)

S, sample = sample_covariance_matrix(Sigma, N)

print("Shape of empirical covariance matrix: ", S.shape)
print("Shape of the sample array: ", sample.shape)


def df_to_biom(df):
    return Table(np.transpose(df.values), [str(i) for i in df.columns], [str(i) for i in df.index])
