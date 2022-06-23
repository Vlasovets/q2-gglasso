import sys
from os.path import dirname

# make sure you are in the correct directory
# q2_gglasso_dir = dirname(os.getcwd())
q2_gglasso_dir = dirname('/opt/project/')
sys.path.append(q2_gglasso_dir)

print(sys.path)

import importlib
import q2_gglasso as q2g

from q2_types.feature_table import FeatureTable, Composition, Frequency
from qiime2.plugin import (Plugin, Float, Str, Bool, List, Int, Choices)

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

plugin.register_semantic_types(
    q2g.PairwiseFeatureData,
    q2g.GGLassoProblem
)

plugin.register_formats(
    q2g.GGLassoDataFormat,
    q2g.PairwiseFeatureDataDirectoryFormat,
    q2g.ZarrProblemFormat,
    q2g.GGLassoProblemDirectoryFormat,

)

plugin.register_semantic_type_to_format(
    q2g.PairwiseFeatureData, artifact_format=q2g.PairwiseFeatureDataDirectoryFormat,
)
plugin.register_semantic_type_to_format(
    q2g.GGLassoProblem, artifact_format=q2g.GGLassoProblemDirectoryFormat
)

# features_clr
plugin.methods.register_function(
    function=q2g.transform_features,
    inputs={"table": FeatureTable[Composition | Frequency]},
    parameters={"transformation": Str},
    outputs=[("transformed_table", FeatureTable[Composition])],
    input_descriptions={
        "table": (
            "Matrix representing the compositional "
            "data of the problem, in order to clr transform it"
        ),
    },
    parameter_descriptions={
        "transformation": (
            "String representing the name of the "
            "transformation we will use "
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
    function=q2g.calculate_covariance,
    inputs={"table": FeatureTable[Composition]},
    parameters={"method": Str, "bias": Bool},
    outputs=[("covariance_matrix", q2g.PairwiseFeatureData)],
    input_descriptions={
        "table": (
            "Matrix representing the microbiome data:"
            "p x n matrix where OTUs - p rows, samples - n columns"
        )
    },
    parameter_descriptions={
        "method": (
            "String if 'unscaled' calculates covariance"
            "if 'scaled' calculates correlation by scaling the entries with 1/sqrt(d) where d is a matrix diagonal"
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

plugin.methods.register_function(
    function=q2g.solve_problem,
    inputs={
        "covariance_matrix": List[q2g.PairwiseFeatureData]
    },
    parameters=q2g.glasso_parameters,
    outputs=[("solution", q2g.GGLassoProblem)],
    input_descriptions={
        "covariance_matrix": (
            "p x p semi-positive definite covariance matrix."
        )
    },
    parameter_descriptions={
        "lambda1": (
            "List of regularization hyperparameters lambda1."
            "Note, sort lambda list in descending order."
        ),
        "lambda2": (
            "List of regularization hyperparameters lambda2 for MGL."
            "Note, sort lambda list in descending order."
        ),
        "latent": ("Specify whether latent variables should be modeled."
                   "The default is False."),
        "mu1": ("Low-rank regularization parameter."
                "Only needs to be specified if latent=True."),
        "reg": ("Type of regularization for MGL problems."
                "'FGL' = Fused Graphical Lasso, 'GGL' = Group Graphical Lasso."
                "The default is 'GGL'."),
    },
    output_descriptions={"solution": "dictionary containing the solution and "
                                     "hyper-/parameters of GGLasso problem"},
    name="solve_problem",
    description=(
        "Method for doing model selection for K single Graphical Lasso problems."
        "Use grid search and AIC/eBIC."
    ),
)

plugin.visualizers.register_function(
    function=q2g.heatmap,
    inputs={
        "solution": q2g.GGLassoProblem,
    },
    name='Generate a heatmap',
    description='Generate a heatmap representation of a symmetric matrix',
    input_descriptions={
        "solution": (
            "p x p semi-positive definite covariance matrix."
        ),
    },
    parameters={'color_scheme': Str % Choices(q2g.heatmap_choices['color_scheme'])},
    parameter_descriptions={
        'color_scheme': 'The matplotlib colorscheme to generate the heatmap '
                        'with.',
    },
)


plugin.visualizers.register_function(
    function=q2g.summarize,
    inputs={
        "solution": q2g.GGLassoProblem,
    },
    name='Summary table',
    description='Summary table with sparsity level, lambda, mu path and rank of the solution',
    input_descriptions={
        "solution": (
            "p x p semi-positive definite covariance matrix."
        ),
    },
)



importlib.import_module('q2_gglasso._transformer')
