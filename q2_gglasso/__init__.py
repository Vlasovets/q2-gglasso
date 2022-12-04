from q2_gglasso._func import (
    transform_features,
    build_groups,
    calculate_covariance,
    solve_problem
)

from q2_gglasso._dict import (
    glasso_parameters
)

from q2_gglasso._format import (
    ZarrProblemFormat,
    GGLassoDataFormat,
    TensorDataFormat,
    PairwiseFeatureDataDirectoryFormat,
    GGLassoProblemDirectoryFormat,
    TensorDataDirectoryFormat
)

from q2_gglasso._type import (
    PairwiseFeatureData,
    GGLassoProblem,
    TensorData
)

from ._heatmap import (
    heatmap,
    heatmap_choices
)

from ._pca import (
    pca
)

from ._summarize import (
    summarize
)

from .utils import (
    flatten_array,
    if_2d_array,
    remove_biom_header,
    to_zarr,
    PCA,
    correlated_PC,
    calculate_seq_depth,
    get_hyperparameters
)

from gglasso.problem import glasso_problem
from . import _version

__version__ = _version.get_versions()['version']
