import qiime2
import numpy as np
import pandas as pd
import zarr

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

from gglasso.problem import glasso_problem
from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.helper.utils import normalize, log_transform


from qiime2.plugin import (
    SemanticType,
    Plugin,
    Int,
    Float,
    Range,
    Metadata,
    Str,
    Bool,
    Choices,
    MetadataColumn,
    Categorical,
    List,
    Citations,
    TypeMatch,
    Numeric,
)

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData

import pandas as pd

def transform_features(
    features: pd.DataFrame, transformation: Str = "clr", coef: float = 0.5
) -> pd.DataFrame:
    if transformation == "clr":
        X = features.values
        null_set = X <= 0.0
        X[null_set] = coef
        X = np.log(X)
        X = (X.T - np.mean(X, axis=1)).T

        return pd.DataFrame(
            data=X, index=list(features.index), columns=list(features.columns)
        )

    else:
        raise ValueError(
            "Unknown transformation name, use clr and not %r" % transformation
        )


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