import qiime2
import numpy as np
from biom.table import Table
import pandas as pd
import zarr
from scipy import stats
import multiprocessing
from functools import partial
import warnings

from biom.table import Table
from biom import load_table

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

from gglasso.problem import glasso_problem
from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.helper.utils import log_transform, normalize
from gglasso.helper.basic_linalg import scale_array_by_diagonal
from gglasso.helper.model_selection import aic, ebic, K_single_grid

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

import pandas as pd


def to_zarr(obj, name, root, first=True):
    """
    Function for converting a GGLasso object to a zarr file, a with tree structue.
    """
    # name 'S' is dedicated for some internal usage in zarr notation and cannot be accessed as a key while reading
    if name == "S":
        name = 'covariance'

    if isinstance(obj, dict):
        if first:
            zz = root
        else:
            zz = root.create_group(name)

        for key, value in obj.items():
            to_zarr(value, key, zz, first=False)

    elif isinstance(obj, (np.ndarray, pd.DataFrame)):
        root.create_dataset(name, data=obj, shape=obj.shape)

    elif isinstance(obj, (str, bool, float, int)):
        to_zarr(np.array(obj), name, root, first=False)

    elif isinstance(obj, type(None)):
        pass
    else:
        to_zarr(obj.__dict__, name, root, first=first)


def transform_features(
        table: Table, transformation: str = "clr",
) -> pd.DataFrame:
    if transformation == "clr":

        X = table.to_dataframe()
        X = normalize(X)
        X = log_transform(X)

        return pd.DataFrame(X)

    else:
        raise ValueError(
            "Unknown transformation name, use clr and not %r" % transformation
        )


def calculate_covariance(table: pd.DataFrame,
                         method: str,
                         bias: bool = True,
                         ) -> pd.DataFrame:
    S = np.cov(table.values, bias=bias)

    if method == "unscaled":
        print("Calculate {0} covariance matrices S".format(method))
        result = S

    elif method == "scaled":
        print("Calculate {0} covariance (correlation) matrices S".format(method))
        result = scale_array_by_diagonal(S)

    else:
        raise ValueError('Given covariance calculation method is not supported.')

    return pd.DataFrame(result)


def solve_problem(covariance_matrix: pd.DataFrame, lambda1: list = None, latent: bool = None, mu1: list = None) \
        -> (pd.DataFrame, pd.DataFrame):
    S = covariance_matrix.values

    model_selection = True

    if mu1 is None:
        mu1 = [None]

    if (len(lambda1) == 1) and (len(mu1) == 1):
        model_selection = False
        lambda1 = np.array(lambda1).item()
        mu1 = np.array(mu1).item()

    if latent:

        if model_selection:
            modelselect_params = {'lambda1_range': lambda1, 'mu1_range': mu1}
            P = glasso_problem(S, N=1, latent=True)
            P.model_selection(modelselect_params=modelselect_params)
        else:
            P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1, "mu1": mu1}, latent=True)
            P.solve()

    else:

        if model_selection:
            modelselect_params = {'lambda1_range': lambda1}
            P = glasso_problem(S, N=1, latent=False)
            P.model_selection(modelselect_params=modelselect_params)
        else:
            P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1}, latent=False)
            P.solve()

    sol = P.solution.precision_
    L = P.solution.lowrank_

    return pd.DataFrame(sol), pd.DataFrame(L)


def solve_problem_new(covariance_matrix: pd.DataFrame, lambda1: list = None, latent: bool = None, mu1: list = None) \
        -> glasso_problem:
    S = covariance_matrix.values

    model_selection = True

    if mu1 is None:
        mu1 = [None]

    # method solve() is for solving GGLasso with particular lambda/mu value (just 1)
    if (len(lambda1) == 1) and (len(mu1) == 1):
        model_selection = False
        lambda1 = np.array(lambda1).item()
        mu1 = np.array(mu1).item()

    if latent:

        if model_selection:
            modelselect_params = {'lambda1_range': lambda1, 'mu1_range': mu1}
            P = glasso_problem(S, N=1, latent=True)
            P.model_selection(modelselect_params=modelselect_params)
        else:
            P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1, "mu1": mu1}, latent=True)
            P.solve()

    else:

        if model_selection:
            modelselect_params = {'lambda1_range': lambda1}
            P = glasso_problem(S, N=1, latent=False)
            P.model_selection(modelselect_params=modelselect_params)
        else:
            P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1}, latent=False)
            P.solve()

    return P




def PCA(X, L, inverse=True):
    sig, V = np.linalg.eigh(L)

    # sort eigenvalues in descending order
    sig = sig[::-1]
    V = V[:, ::-1]

    ind = np.argwhere(sig > 1e-9)

    if inverse:
        loadings = V[:, ind] @ np.diag(np.sqrt(1 / sig[ind]))
    else:
        loadings = V[:, ind] @ np.diag(np.sqrt(sig[ind]))

    # compute the projection
    zu = X.values @ loadings

    return zu, loadings, np.round(sig[ind].squeeze(), 3)


def remove_biom_header(file_path):
    with open(str(file_path), 'r') as fin:
        data = fin.read().splitlines(True)
    with open(str(file_path), 'w') as fout:
        fout.writelines(data[1:])
