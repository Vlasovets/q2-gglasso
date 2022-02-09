import qiime2
import numpy as np
import pandas as pd
import zarr
from scipy.stats import spearmanr

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


def calculate_covariance(table: pd.DataFrame,
                         method: str,
                         bias: bool = True,
                         ) -> np.ndarray:

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

