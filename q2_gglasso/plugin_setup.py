import sys
from os.path import dirname

# make sure you are in the correct directory
# q2_gglasso_dir = dirname(os.getcwd())
q2_gglasso_dir = dirname('/opt/project/')
sys.path.append(q2_gglasso_dir)

print(sys.path)

import importlib
import q2_gglasso as q2g
import qiime2

from q2_types.feature_table import FeatureTable, Composition, Frequency
from q2_types.feature_data import FeatureData, Taxonomy
from qiime2.plugin import Plugin, Float, Str, Bool, List, Int, Metadata

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
    q2g.GGLassoProblem,
    q2g.TensorData
)

plugin.register_formats(
    q2g.GGLassoDataFormat,
    q2g.PairwiseFeatureDataDirectoryFormat,
    q2g.ZarrProblemFormat,
    q2g.GGLassoProblemDirectoryFormat,
    q2g.TensorDataFormat,
    q2g.TensorDataDirectoryFormat

)

plugin.register_semantic_type_to_format(
    q2g.PairwiseFeatureData, artifact_format=q2g.PairwiseFeatureDataDirectoryFormat,
)
plugin.register_semantic_type_to_format(
    q2g.GGLassoProblem, artifact_format=q2g.GGLassoProblemDirectoryFormat
)
plugin.register_semantic_type_to_format(
    q2g.TensorData, artifact_format=q2g.TensorDataDirectoryFormat
)

plugin.methods.register_function(
    function=q2g.transform_features,
    inputs={"table": FeatureTable[Composition | Frequency]},
    parameters={"transformation": Str, "pseudo_count": Int},
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
        "pseudo_count": (
            "Add pseudo count, only necessary for clr-transformation."
        ),
    },
    output_descriptions={"transformed_table": "Matrix representing the data of the problem"},
    name="transform-features",
    description=(
        "Perform transformation, "
        "from FeatureTable[Frequency]"
        "prior to network analysis"
        "default transformation is centered log ratio (CLR)"
    ),
)

plugin.methods.register_function(
    function=q2g.build_groups,
    inputs={"tables": List[FeatureTable[Composition]]},
    parameters={"check_groups": Bool},
    outputs=[("group_array", q2g.TensorData)],
    input_descriptions={
        "tables": (
            "Dataframes containing the transformed data "
        ),
    },
    parameter_descriptions={
        "check_groups": (
            "Check built groups of overlapping variables"
        ),
    },
    output_descriptions={"group_array": " (2,L,K)-shape array which contains the indices of a precision matrix entry "
                                        "for every group of overlapping features (L) and every instance (K)"
                         },
    name="build-groups",
    description=(
        "G can be seen as a bookeeping array between instances having different number of features"
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
        "n_samples": (
            "List of number of samples for each instance k=1,..,K."
        ),
        "lambda1_min": (
            "List of regularization hyperparameters lambda1."
            "Note, sort lambda list in descending order."
        ),
        "lambda2_min": (
            "List of regularization hyperparameters lambda2 for MGL."
            "Note, sort lambda list in descending order."
        ),
        "lambda1_max": (
            "List of regularization hyperparameters lambda1."
            "Note, sort lambda list in descending order."
        ),
        "lambda2_max": (
            "List of regularization hyperparameters lambda2 for MGL."
            "Note, sort lambda list in descending order."
        ),
        "n_lambda1": (
            "List of regularization hyperparameters lambda1."
            "Note, sort lambda list in descending order."
        ),
        "n_lambda2": (
            "List of regularization hyperparameters lambda2 for MGL."
            "Note, sort lambda list in descending order."
        ),
        "latent": (
            "Specify whether latent variables should be modeled."
            "The default is False."
        ),
        "mu1": (
            "Low-rank regularization parameter."
            "Only needs to be specified if latent=True."
        ),
        "lambda1_mask": (
            "Array (p,p), non-negative, symmetric."
            "The lambda1 parameter is multiplied element-wise with this array, thus lambda1 has to be provided."
        ),
        "reg": (
            "Type of regularization for MGL problems."
            "'FGL' = Fused Graphical Lasso, 'GGL' = Group Graphical Lasso."
            "The default is 'GGL'."
        ),
        "non_conforming": (
            "Non-conforming MGL problems."
        ),
        "group_array": (
            "Bookeeping array"
        ),
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
    function=q2g.pca,
    inputs={
        "table": FeatureTable[Composition],
        "solution": q2g.GGLassoProblem,
    },
    name='Principal component analysis (PCA)',
    description='Generate a scatter plot for PCA',
    input_descriptions={
        "table": (
            "Matrix representing the microbiome data:"
            "p x n matrix where OTUs - p rows, samples - n columns"
        ),
        "solution": (
            "Solution artifact of Graphical Lasso problem with latent variables."
        ),
    },
    parameters={'sample_metadata': Metadata, "n_components": Int, "color_by": Str},
    parameter_descriptions={
        'sample_metadata': "Metadata of the study.",
        'n_components': "Number of PCs to be printed",
        'color_by': "Color components by selected covariate from metadata"
    },
)

plugin.visualizers.register_function(
    function=q2g.summarize,
    inputs={
        "solution": q2g.GGLassoProblem,
        "transformed_table": FeatureTable[Composition],
        "taxonomy": FeatureData[Taxonomy],
    },
    name='Summary table',
    description='Summary table with sparsity level, lambda, mu path and rank of the solution',
    input_descriptions={
        "solution": (
            "p x p semi-positive definite covariance matrix."
        ),
    },
    parameters={"width": Int, "height": Int, "label_size": Str},
    parameter_descriptions={
        'width': "The width you would like your plots to be, by default 1500.",
        'height': 'The height you would like your plots to be, by default 1500',
        'label_size': 'The font size of labels ticks in, by default "5pt".',
    },
)

importlib.import_module('q2_gglasso._transformer')
