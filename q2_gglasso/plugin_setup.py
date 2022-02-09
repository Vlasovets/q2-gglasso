import sys
from os.path import dirname

import qiime2
import q2_gglasso
from q2_types.feature_table import FeatureTable, Composition, Frequency
from q2_gglasso._type import PairwiseFeatureData
from qiime2.plugin import (Plugin, Float, Str, Bool)

# make sure you are in the correct directory
# q2_gglasso_dir = dirname(os.getcwd())
q2_gglasso_dir = dirname('/opt/project/')
sys.path.append(q2_gglasso_dir)


version = qiime2.__version__

plugin = Plugin(
    name="gglasso",
    version="0.0.0.dev0",
    website="https://github.com/Vlasovets/q2-gglasso",
    package="q2-gglasso",
    short_description=(
        "Package for solving General Graphical Lasso problems"
    ),
    description=(
        "This package contains algorithms for solving General Graphical Lasso (GGLasso) problems"
        "including single, multiple, as well as latent Graphical Lasso problems."
    ),
)

# features_clr
plugin.methods.register_function(
    function=q2_gglasso.transform_features,
    inputs={"table": FeatureTable[Composition | Frequency]},
    parameters={"transformation": Str, "pseudocount": Float},
    outputs=[("transformed_table", FeatureTable[Composition])],
    input_descriptions={
        "table": (
            "Matrix representing the compositional "
            "data of the problem, in order to clr transform it"
        )
    },
    parameter_descriptions={
        "transformation": (
            "String representing the name of the "
            "transformation we will use "
        ),
        "pseudocount": (
            "Value that should be put instead of zeros"
            "in the feature table. Default value is 0.5"
        ),
    },
    output_descriptions={"transformed_table": "Matrix representing the data of the problem"},
    name="transform-features",
    description=(
        "Perform transformation, "
        "from FeatureTable[Frequency]"
        " prior to network analysis"
        " default transformation is centered log ratio"
    ),
)

plugin.methods.register_function(
    function=q2_gglasso.calculate_covariance,
    inputs={"table": FeatureTable[Composition | Frequency]},
    parameters={"method": Str, "bias": Bool},
    outputs=[("covariance_matrix", PairwiseFeatureData)],
    input_descriptions={
        "table": (
            "Matrix representing the microbiome data:"
            "p x n matrix where OTUs - p rows, samples - n columns"
        )
    },
    parameter_descriptions={
        "method": (
            "String if 'unscaled' calculates covariance"
            "for more details check np.cov() documentaion"
        ),
        "bias": (
            "Default value is True"
            "If you derive the log likelihood of inverse covariance "
            "you get out the empirical covariance matrix with normalization N"
        ),
    },
    output_descriptions={"covariance_matrix": "p x p matrix with covariance entries"},
    name="calculate_covariance",
    description=(
        "Perform empirical covariance estimation given the data p x N, "
        "from FeatureTable[Composition | Frequency]"
        "prior to network analysis"
        "default transformation is centered log ratio"
    ),
)
