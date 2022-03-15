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


from qiime2.plugin import SemanticType
from q2_types.feature_data import FeatureData


glasso_parameters = {
    "lambda1": Float
}

# glasso_parameters = {
#     "method": Str,
#     "lambda1": List[Float],
#     "n_samples": Int,
#     "gamma": Float,
#     "latent": Bool,
#     "use_block": Bool
# }