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
from gglasso.helper.basic_linalg import scale_array_by_diagonal
from gglasso.helper.model_selection import aic, ebic, K_single_grid

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

from .utils import if_none_to_list, if_2d_array, if_no_model_selection
from .utils import normalize, single_hyperparameters, log_transform, to_zarr


def transform_features(
        table: Table, transformation: str = "clr",
) -> pd.DataFrame:
    if transformation == "clr":
        X = table.to_dataframe()
        X = normalize(X)
        X = log_transform(X, transformation=transformation)

        return pd.DataFrame(X)

    elif transformation == "mclr":
        X = table.to_dataframe()
        X = normalize(X)
        X = log_transform(X, transformation=transformation)

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

    elif method == "scaled":  # TO DO: add doc string explaining that it is the same as Pearson's correlation
        print("Calculate {0} covariance (correlation) matrices S".format(method))
        result = scale_array_by_diagonal(S)

    else:
        raise ValueError('Given covariance calculation method is not supported.')

    return pd.DataFrame(result)


def solve_problem(covariance_matrix: list, lambda1: list = None, lambda2: list = None, latent: bool = None,
                  mu1: list = None, reg: str = 'GGL') \
        -> glasso_problem:
    S = np.array(covariance_matrix)

    S = if_2d_array(S)
    mu1 = if_none_to_list(mu1)
    lambda2 = if_none_to_list(lambda2)

    if S.ndim == 2:  # 2d array => SGL
        model_selection = if_no_model_selection(lambda1=lambda1, lambda2=lambda2, mu1=mu1)
        lambda1, lambda2, mu1 = single_hyperparameters(model_selection=model_selection,
                                                       lambda1=lambda1, lambda2=lambda2, mu1=mu1)

        if latent:
            print("\n----SOLVING SINGLE GRAPHICAL LASSO PROBLEM WITH LATENT VARIABLES-----")

            if model_selection:
                print("\tDD MODEL SELECTION:")
                modelselect_params = {'lambda_range': lambda1, 'mu1_range': mu1}
                P = glasso_problem(S, N=1, latent=True)
                P.model_selection(modelselect_params=modelselect_params)
            else:
                print("\tWITH LAMBDA={0} and MU={1}".format(lambda1, mu1))
                P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1, "mu1": mu1}, latent=True)
                P.solve()

        else:
            print("----SOLVING SINGLE GRAPHICAL LASSO PROBLEM-----")

            if model_selection:
                print("\tDD MODEL SELECTION:")
                modelselect_params = {'lambda1_range': lambda1}
                P = glasso_problem(S, N=1, latent=False)
                P.model_selection(modelselect_params=modelselect_params)
            else:
                print("\tWITH LAMBDA={0}".format(lambda1))
                P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1}, latent=False)
                P.solve()

    elif S.ndim == 3:  # 3d array => MGL
        model_selection = if_no_model_selection(lambda1=lambda1, lambda2=lambda2, mu1=mu1)
        lambda1, lambda2, mu1 = single_hyperparameters(model_selection=model_selection,
                                                       lambda1=lambda1, lambda2=lambda2, mu1=mu1)

        if latent:
            print("\n----SOLVING GROUP GRAPHICAL LASSO PROBLEM WITH LATENT VARIABLES-----")

            if model_selection:
                print("\tDD MODEL SELECTION:")
                modelselect_params = {'lambda1_range': lambda1, 'lambda2_range': lambda2, 'mu_range': mu1}
                P = glasso_problem(S, N=1, latent=True, reg=reg)
                P.model_selection(modelselect_params=modelselect_params)
            else:
                print("\tWITH LAMBDA1={0}, LAMBDA2={1} and MU={2}".format(lambda1, lambda2, mu1))
                P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1, 'lambda2': lambda2, "mu1": mu1}, latent=True,
                                   reg=reg)
                P.solve()

        else:
            print("----SOLVING GROUP GRAPHICAL LASSO PROBLEM-----")

            if model_selection:
                print("\tDD MODEL SELECTION:")
                modelselect_params = {'lambda1_range': lambda1, 'lambda2_range': lambda2}
                P = glasso_problem(S, N=1, latent=False, reg=reg)
                P.model_selection(modelselect_params=modelselect_params)
            else:
                print("\tWITH LAMBDA1={0} and LAMBDA2={1}".format(lambda1, lambda2))
                P = glasso_problem(S, N=1, reg_params={'lambda1': lambda1, 'lambda2': lambda2}, latent=False, reg=reg)
                P.solve()

    return P

