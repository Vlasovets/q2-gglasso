import numpy as np
import pandas as pd
import zarr
import qiime2
import biom
import skbio
import sys
import os
from os.path import join, dirname

from qiime2.plugin import (Plugin, Int, Float, Range, Metadata, Str, Bool, Choices, MetadataColumn, Categorical, List,
     Citations, TypeMatch, Numeric, SemanticType)

from q2_types.feature_table import FeatureTable, Composition, BIOMV210Format, BIOMV210DirFmt, Frequency, Design
from q2_types.feature_data import TSVTaxonomyFormat, FeatureData, Taxonomy


q2_gglasso_dir = dirname(os.getcwd())
sys.path.append(q2_gglasso_dir)
import q2_gglasso



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
    inputs={"features": FeatureTable[Composition | Frequency | Design]},
    parameters={"transformation": Str, "coef": Float},
    outputs=[("x", FeatureTable[Design])],
    input_descriptions={
        "features": (
            "Matrix representing the compositional "
            "data of the problem, in order to clr transform it"
        )
    },
    parameter_descriptions={
        "transformation": (
            "String representing the name of the "
            "transformation we will use "
        ),
        "coef": (
            "Value that should be put instead of zeros"
            "in the feature table. Default value is 0.5"
        ),
    },
    output_descriptions={"x": "Matrix representing the data of the problem"},
    name="transform-features",
    description=(
        "Perform transformation, "
        "from FeatureTable[Composition/Frequency]"
        " prior to regress or classify,"
        " default transformation is centered log ratio"
    ),
)