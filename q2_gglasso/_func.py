import qiime2
import numpy as np
from biom.table import Table
import pandas as pd
import zarr
from scipy.stats import spearmanr
import multiprocessing
from functools import partial
import warnings

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

from gglasso.problem import glasso_problem
from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.helper.utils import log_transform, zero_replacement


from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

import pandas as pd


def transform_features(
    table: pd.DataFrame, transformation: str = "clr", pseudocount: float = 0.5
) -> pd.DataFrame:
    if transformation == "clr":
        X = zero_replacement(table, c=pseudocount)
        X = log_transform(X)

        return pd.DataFrame(
            data=X, index=list(table.index), columns=list(table.columns)
        )

    else:
        raise ValueError(
            "Unknown transformation name, use clr and not %r" % transformation
        )


def pairwise_iter_wo_metadata(pairwise_iter):
    for (val_i, id_i, _), (val_j, id_j, _) in pairwise_iter:
        yield ((val_i, id_i), (val_j, id_j))

def calculate_correlation(data, corr_method=spearmanr):
    (val_i, id_i), (val_j, id_j) = data
    r, p = corr_method(val_i, val_j)
    return (id_i, id_j), (r, p)


def calculate_correlations(table: Table, corr_method=spearmanr, nprocs=1) -> pd.DataFrame:
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
    correls = pd.DataFrame(data, index=index, columns=['r', 'p'])
    # Turn tuple index into actual multiindex, now guaranteeing that correls index is sorted
    correls.index = pd.MultiIndex.from_tuples([sorted(i) for i in correls.index])

    return correls


def calculate_covariance(table: Table,
                         method: str,
                         bias: bool = True,
                         ) -> pd.DataFrame:

    if method == "unscaled":

        print("Calculate {0} covariance matrices S".format(method))
        S = np.cov(table, bias=bias)

    # elif method is "correlation":
    #     print("Calculate Spearman's {0} matrices C".format(method))
    #     C = spearmanr(table)
    else:
        raise ValueError('Provided covariance calculation method is not supported.')
    return S


def to_zarr(obj, name, root, first=True):
    """
    Function for converting a python object to a zarr file, a with tree structue.
    """
    if type(obj) == dict:
        if first:
            zz = root
        else:
            zz = root.create_group(name)

        for key, value in obj.items():
            to_zarr(value, key, zz, first=False)

    elif type(obj) in [np.ndarray, pd.DataFrame]:
        root.create_dataset(name, data=obj, shape=obj.shape)

    elif type(obj) == np.float64:
        root.attrs[name] = float(obj)

    elif type(obj) == np.int64:
        root.attrs[name] = int(obj)

    elif type(obj) == list:
        if name == "tree":
            root.attrs[name] = obj
        else:
            to_zarr(np.array(obj), name, root, first=False)

    elif obj is None or type(obj) in [str, bool, float, int]:
        root.attrs[name] = obj

    else:
        to_zarr(obj.__dict__, name, root, first=first)

