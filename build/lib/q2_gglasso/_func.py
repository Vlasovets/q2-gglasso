import qiime2
import numpy as np
from biom.table import Table
import pandas as pd
import zarr
from scipy import stats
import multiprocessing
from functools import partial
import warnings

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

from gglasso.problem import glasso_problem
from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.helper.utils import log_transform
from gglasso.helper.model_selection import aic, ebic, K_single_grid

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

import pandas as pd

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


def transform_features(
        table: pd.DataFrame, transformation: str = "clr"
) -> pd.DataFrame:
    if transformation == "clr":

        X = log_transform(table)

        return pd.DataFrame(
            data=X, index=list(table.index), columns=list(table.columns)
        )

    else:
        raise ValueError(
            "Unknown transformation name, use clr and not %r" % transformation
        )


def calculate_covariance(table: pd.DataFrame,
                         method: str,
                         bias: bool = True,
                         ) -> pd.DataFrame:
    # X = table.to_dataframe()

    if method == "unscaled":
        print("Calculate {0} covariance matrices S".format(method))
        result = np.cov(table, bias=bias)

    elif method == "spearmanr":
        result, pval = stats.spearmanr(table, table)  # TODO Scaling function here

    else:
        raise ValueError('Given covariance calculation method is not supported.')

    return pd.DataFrame(result)





# def solve_problem(covariance_matrix: pd.DataFrame,
#                   lambda1: float = .05) -> pd.DataFrame:
#
#     S = covariance_matrix.values
#
#     P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1}, latent=False, do_scaling=False)
#     P.solve()
#     sol = P.solution.precision_
#
#     # if modelselect_params:
#     #
#     #     P.model_selection(modelselect_params=modelselect_params, method=method, gamma=gamma)
#
#     # else:
#     #     P.solve()
#     #     sol = P.solution.precision_
#
#     return pd.DataFrame(sol)


def solve_problem(covariance_matrix: pd.DataFrame, lambda1: float=0.05) -> pd.DataFrame:

    S = covariance_matrix.values

    P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1}, latent=False, do_scaling=False)
    P.solve()
    sol = P.solution.precision_

    return pd.DataFrame(sol)

# def solve_problem(covariance_matrix: pd.DataFrame,
#                  problem: str,
#                  lambda1: list,
#                  n_samples: int,
#                  method: str = 'eBIC',
#                  gamma: float = 0.3,
#                  latent: bool = False,
#                  use_block: bool = True
#                  ) -> pd.DataFrame:
#
#     if problem == "single":
#         print("Solve {0} Graphical Lasso problem".format(method))
#
#         covariance_matrix = np.array(covariance_matrix)
#         covariance_matrix = covariance_matrix.reshape(1, covariance_matrix.shape[0], covariance_matrix.shape[1])
#
#         est_uniform, est_indv, statistics = K_single_grid(covariance_matrix,
#                                                           lambda_range=lambda1,
#                                                           N=n_samples,
#                                                           method=method,
#                                                           gamma=gamma,
#                                                           latent=latent,
#                                                           use_block=use_block)
#
#         result = est_uniform["Theta"].reshape(covariance_matrix.shape[0], covariance_matrix.shape[1])
#
#     else:
#         raise ValueError('Given Graphical lasso problem is not supported.')
#
#     return pd.DataFrame(result)
