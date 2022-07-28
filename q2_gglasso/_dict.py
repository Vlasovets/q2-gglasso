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
    "n_samples": List[Int],
    "lambda1": List[Float],
    "lambda2": List[Float],
    "latent": Bool,
    "non_conforming": Bool,
    "G": List[Int],
    "mu1": List[Float],
    "reg": Str
}