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
from gglasso.helper.ext_admm_helper import create_group_array, construct_indexer, check_G
from gglasso.helper.model_selection import aic, ebic, K_single_grid

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

from .utils import if_2d_array, if_model_selection, if_all_none, list_to_array
from .utils import normalize, log_transform, to_zarr, transpose_dataframes


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


def build_groups(tables: Table, check_groups: bool = True) -> np.ndarray:

    columns_dict = dict()
    dataframes = list()
    i = 0
    for table in tables:
        df = table.to_dataframe()
        dataframes.append(df)
        columns_dict[i] = df.columns
        i += 1

    for k in range(0, len(columns_dict) - 1):
        non_conforming_problem = set(columns_dict[k].difference(columns_dict[k + 1]))
        if non_conforming_problem:
            tables_trans = transpose_dataframes(dataframes)
            # we transpose dataframes, so the shape of each dataframe becomes (n_variables, n_samples)
            ix_exist, ix_location = construct_indexer(list(tables_trans))

            G = create_group_array(ix_exist, ix_location)

            # print("Dimensions p_k: ", p_arr)
            # print("Sample sizes N_k: ", num_samples)

            print("Number of groups found: ", G.shape[1])

            return G

        else:
            print("All datasets have exactly the same number of features.")



def calculate_covariance(table: pd.DataFrame,
                         method: str,
                         bias: bool = True,
                         ) -> pd.DataFrame:
    # normalize with N => bias = True
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


def solve_SGL(S: np.ndarray, N: list, latent: bool = None, model_selection: bool = None,
              lambda1: list = None, mu1: list = None):

    if model_selection:
        print("\tDD MODEL SELECTION:")
        modelselect_params = {'lambda1_range': lambda1, 'mu1_range': mu1}
        P = glasso_problem(S, N=N, latent=latent)
        P.model_selection(modelselect_params=modelselect_params)
        print(P.__dict__["modelselect_params"])
    else:
        print("\tWITH LAMBDA={0} and MU={1}".format(lambda1, mu1))
        P = glasso_problem(S, N=N, reg_params={'lambda1': lambda1, "mu1": mu1}, latent=latent)
        P.solve()

    return P


def solve_MGL(S: np.ndarray, N: list, reg: str, latent: bool = None, model_selection: bool = None,
              lambda1: list = None, lambda2: list = None, mu1: list = None):

    if model_selection:
        print("\tDD MODEL SELECTION:")
        modelselect_params = {'lambda1_range': lambda1, 'lambda2_range': lambda2, 'mu1_range': mu1}
        P = glasso_problem(S, N=N, latent=latent, reg=reg)
        P.model_selection(modelselect_params=modelselect_params)
        print(P.__dict__["modelselect_params"])
    else:
        print("\tWITH LAMBDA1={0}, LAMBDA2={1} and MU={2}".format(lambda1, lambda2, mu1))
        P = glasso_problem(S, N=N, reg_params={'lambda1': lambda1, 'lambda2': lambda2, "mu1": mu1},
                           latent=latent, reg=reg)
        P.solve()

    return P


def solve_non_conforming(S: np.ndarray, N: list, G: list, latent: bool = None, model_selection: bool = None,
                         lambda1: list = None, lambda2: list = None, mu1: list = None):

    if model_selection:
        print("\tDD MODEL SELECTION:")
        modelselect_params = {'lambda1_range': lambda1, 'lambda2_range': lambda2, 'mu1_range': mu1}
        P = glasso_problem(S, N=N, G=G, latent=latent, reg='GGL')
        P.model_selection(modelselect_params=modelselect_params)
        print(P.__dict__["modelselect_params"])

    else:
        print("\tWITH LAMBDA1={0}, LAMBDA2={1} and MU={2}".format(lambda1, lambda2, mu1))
        P = glasso_problem(S, N=N, G=G, reg_params={'lambda1': lambda1, 'lambda2': lambda2, "mu1": mu1},
                           latent=latent, reg='GGL')
        P.solve()

    return P


def solve_problem(covariance_matrix: list, n_samples: list, latent: bool = None, non_conforming: bool = None,
                  lambda1: list = None, lambda2: list = None, mu1: list = None,  G: list = None, reg: str = 'GGL') \
        -> glasso_problem:
    S = np.array(covariance_matrix)
    S = if_2d_array(S)

    n_samples = list_to_array(n_samples)

    # set default hyperparameters if not provided by the user
    lambda1, lambda2, mu1 = if_all_none(lambda1, lambda2, mu1)

    h_params = if_model_selection(lambda1, lambda2, mu1)
    model_selection = h_params["model_selection"]
    lambda1, lambda2, mu1 = h_params["lambda1"], h_params["lambda2"], h_params["mu1"]

    # if 2d array => solve SGL
    if S.ndim == 2:

        if latent:
            print("\n----SOLVING SINGLE GRAPHICAL LASSO PROBLEM WITH LATENT VARIABLES-----")

            P = solve_SGL(S=S, N=n_samples, latent=latent, model_selection=model_selection, lambda1=lambda1, mu1=mu1)

        else:
            print("----SOLVING SINGLE GRAPHICAL LASSO PROBLEM-----")

            P = solve_SGL(S=S, N=n_samples, latent=latent, model_selection=model_selection, lambda1=lambda1, mu1=mu1)

    # if 3d array => solve MGL
    elif S.ndim == 3:

        if non_conforming:

            if latent:
                print("\n----SOLVING NON-CONFORMING PROBLEM WITH LATENT VARIABLES-----")

                P = solve_non_conforming(S=S, N=n_samples, G=G, latent=latent, model_selection=model_selection,
                                         lambda1=lambda1, lambda2=lambda2, mu1=mu1)
            else:
                print("\n----SOLVING NON-CONFORMING PROBLEM-----")

                P = solve_non_conforming(S=S, N=n_samples, G=G, latent=latent, model_selection=model_selection,
                                         lambda1=lambda1, lambda2=lambda2, mu1=mu1)

        else:
            if latent:
                print("\n----SOLVING {0} PROBLEM WITH LATENT VARIABLES-----".format(reg))

                P = solve_MGL(S=S, N=n_samples, reg=reg, latent=latent, model_selection=model_selection,
                              lambda1=lambda1, lambda2=lambda2, mu1=mu1)
            else:
                print("\n----SOLVING {0} PROBLEM-----".format(reg))

                P = solve_MGL(S=S, N=n_samples, reg=reg, latent=latent, model_selection=model_selection,
                              lambda1=lambda1, lambda2=lambda2, mu1=mu1)

    return P


#
# def solve_problem(covariance_matrix: list, n_samples: list, latent: bool = None,
#                   lambda1: list = None, lambda2: list = None, mu1: list = None, reg: str = 'GGL') \
#         -> glasso_problem:
#     S = np.array(covariance_matrix)
#     S = if_2d_array(S)
#
#     n_samples = list_to_array(n_samples)
#
#     # set default hyperparameters if not provided by the user
#     lambda1, lambda2, mu1 = if_all_none(lambda1, lambda2, mu1)
#
#     h_params = if_model_selection(lambda1, lambda2, mu1)
#     model_selection = h_params["model_selection"]
#     lambda1, lambda2, mu1 = h_params["lambda1"], h_params["lambda2"], h_params["mu1"]
#
#     # if 2d array => solve SGL
#     if S.ndim == 2:
#
#         if latent:
#             print("\n----SOLVING SINGLE GRAPHICAL LASSO PROBLEM WITH LATENT VARIABLES-----")
#
#             if model_selection:
#                 print("\tDD MODEL SELECTION:")
#                 modelselect_params = {'lambda1_range': lambda1, 'mu1_range': mu1}
#                 P = glasso_problem(S, N=n_samples, latent=True)
#                 P.model_selection(modelselect_params=modelselect_params)
#                 print(P.__dict__["modelselect_params"])
#             else:
#                 print("\tWITH LAMBDA={0} and MU={1}".format(lambda1, mu1))
#                 P = glasso_problem(S, N=n_samples, reg_params={'lambda1': lambda1, "mu1": mu1}, latent=True)
#                 P.solve()
#
#         else:
#             print("----SOLVING SINGLE GRAPHICAL LASSO PROBLEM-----")
#
#             if model_selection:
#                 print("\tDD MODEL SELECTION:")
#                 modelselect_params = {'lambda1_range': lambda1}
#                 P = glasso_problem(S, N=n_samples, latent=False)
#                 P.model_selection(modelselect_params=modelselect_params)
#                 print(P.__dict__["modelselect_params"])
#             else:
#                 print("\tWITH LAMBDA={0}".format(lambda1))
#                 P = glasso_problem(S, N=n_samples, reg_params={'lambda1': lambda1}, latent=False)
#                 P.solve()
#
#     # if 3d array => solve MGL
#     elif S.ndim == 3:
#
#         if latent:
#             print("\n----SOLVING {0} PROBLEM WITH LATENT VARIABLES-----".format(reg))
#
#             if model_selection:
#                 print("\tDD MODEL SELECTION:")
#                 modelselect_params = {'lambda1_range': lambda1, 'lambda2_range': lambda2, 'mu1_range': mu1}
#                 P = glasso_problem(S, N=n_samples, latent=True, reg=reg)
#                 P.model_selection(modelselect_params=modelselect_params)
#                 print(P.__dict__["modelselect_params"])
#             else:
#                 print("\tWITH LAMBDA1={0}, LAMBDA2={1} and MU={2}".format(lambda1, lambda2, mu1))
#                 P = glasso_problem(S, N=n_samples, reg_params={'lambda1': lambda1, 'lambda2': lambda2, "mu1": mu1},
#                                    latent=True, reg=reg)
#                 P.solve()
#
#         else:
#             print("\n----SOLVING {0} PROBLEM-----".format(reg))
#
#             if model_selection:
#                 print("\tDD MODEL SELECTION:")
#                 modelselect_params = {'lambda1_range': lambda1, 'lambda2_range': lambda2}
#                 P = glasso_problem(S, N=n_samples, latent=False, reg=reg)
#                 P.model_selection(modelselect_params=modelselect_params)
#                 print(P.__dict__["modelselect_params"])
#             else:
#                 print("\tWITH LAMBDA1={0} and LAMBDA2={1}".format(lambda1, lambda2))
#                 P = glasso_problem(S, N=n_samples, reg_params={'lambda1': lambda1, 'lambda2': lambda2},
#                                    latent=False, reg=reg)
#                 P.solve()
#
#     return P
