"""QIIME 2 plugin for General Graphical Lasso problems."""

from q2_gglasso._func import (
    transform_features,  # noqa: F401
    build_groups,        # noqa: F401
    calculate_covariance,  # noqa: F401
    solve_problem,       # noqa: F401
)

from q2_gglasso._dict import glasso_parameters  # noqa: F401

from q2_gglasso._format import (  # noqa: F401
    ZarrProblemFormat,  # noqa: F401
    GGLassoDataFormat,  # noqa: F401
    TensorDataFormat,  # noqa: F401
    PairwiseFeatureDataDirectoryFormat,  # noqa: F401
    GGLassoProblemDirectoryFormat,  # noqa: F401
    TensorDataDirectoryFormat,  # noqa: F401
)

from q2_gglasso._type import (  # noqa: F401
    PairwiseFeatureData,  # noqa: F401
    GGLassoProblem,  # noqa: F401
    TensorData,  # noqa: F401
)

from ._pca import pca  # noqa: F401
from ._summarize import summarize  # noqa: F401

from .utils import (  # noqa: F401
    flatten_array,  # noqa: F401
    if_2d_array,  # noqa: F401
    remove_biom_header,  # noqa: F401
    to_zarr,  # noqa: F401
    PCA,  # noqa: F401
    correlated_PC,  # noqa: F401
    calculate_seq_depth,  # noqa: F401
    get_hyperparameters,  # noqa: F401
)

from gglasso.problem import glasso_problem  # noqa: F401
from . import _version  # noqa: F401


__version__ = _version.get_versions()["version"]

__all__ = [
    "pca",
    "transform_features",
    "build_groups",
    "calculate_covariance",
    "solve_problem",
    "glasso_parameters",
    "ZarrProblemFormat",
    "GGLassoDataFormat",
    "TensorDataFormat",
    "PairwiseFeatureDataDirectoryFormat",
    "GGLassoProblemDirectoryFormat",
    "TensorDataDirectoryFormat",
    "PairwiseFeatureData",
    "GGLassoProblem",
    "TensorData",
    "summarize",
    "flatten_array",
    "if_2d_array",
    "remove_biom_header",
    "to_zarr",
    "PCA",
    "correlated_PC",
    "calculate_seq_depth",
    "get_hyperparameters",
    "glasso_problem",
    "__version__",
]
