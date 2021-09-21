import numpy as np
import pandas as pd
import zarr
import qiime2


from qiime2.plugin import (Plugin, Int, Float, Range, Metadata, Str, Bool, Choices, MetadataColumn, Categorical, List,
     Citations, TypeMatch, Numeric, SemanticType)

from q2_types.feature_table import FeatureTable, Composition, BIOMV210Format, BIOMV210DirFmt, Frequency, Design
from q2_types.feature_data import TSVTaxonomyFormat, FeatureData, Taxonomy

from . import _func, _dict, _formats


version = qiime2.__version__


# generate_data
plugin.methods.register_function(
           function=test,
           inputs={'taxa': Int},
           parameters={'n': Int},
           outputs=[('x', np.array())],
           input_descriptions={'taxa': 'Taxonomy of the data. If it is given, it will generate random data associated to this'},
           parameter_descriptions={'n': 'number of sample'},
           output_descriptions={'x': 'Matrix representing the data of the problem'},
           name='test',
           description=("Function that build random data")
           )