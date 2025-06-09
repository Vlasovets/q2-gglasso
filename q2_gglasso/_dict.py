from qiime2.plugin import (
    Int,
    Float,
    Str,
    Bool,
    List,
)

glasso_parameters = {
    "n_samples": List[Int],
    "lambda1_min": List[Float],
    "lambda1_max": List[Float],
    "lambda2_min": List[Float],
    "lambda2_max": List[Float],
    "mu1_min": List[Float],
    "mu1_max": List[Float],
    "n_lambda1": Int,
    "n_lambda2": Int,
    "n_mu1": Int,
    "weights": List[Str],
    "latent": Bool,
    "non_conforming": Bool,
    "group_array": List[Int],
    "reg": Str,
    "gamma": Float,
}
