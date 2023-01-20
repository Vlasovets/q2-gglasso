import numpy as np
import pandas as pd
import warnings
import qiime2

from biom.table import Table

from gglasso.problem import glasso_problem
from gglasso.helper.basic_linalg import scale_array_by_diagonal
from gglasso.helper.ext_admm_helper import create_group_array, construct_indexer, check_G

from .utils import if_2d_array, get_hyperparameters, list_to_array, remove_biom_header
from .utils import normalize, log_transform, zero_imputation, check_lambda_path, get_lambda_mask
from sklearn import preprocessing


def transform_features(table: Table, taxonomy: pd.Series, sample_metadata: qiime2.Metadata = None,
                       transformation: str = "clr", pseudo_count: int = 1,
                       scale_metadata: bool = True, add_metadata: bool = False) -> pd.DataFrame:
    """
    Project compositional data to Euclidean space.

    Parameters
    ----------
    pseudo_count: int, optional
        Add pseudo count, only necessary for transformation = "clr".
    table: biom.Table
        A table with count microbiome data.
    transformation: str
        If 'clr' the data is transformed with center log-ratio method by Aitchison (1982).
        If 'mclr' the data is transformed with modified center log-ratio method by Yoon et al. (2019).
    add_metadata: bool
        Add metadata associated with count table, default False.
    scale_metadata: bool
        Scaling metadata with mu=0, std=1, default 'True'.
    sample_metadata: qiime2.Metadata, optional
        Metadata associated with count table.
    taxonomy: pd.Series, optional
        Taxonomic assignments, i.e. bacteria names for feature IDs.

    Returns
    -------
    X: pd.Dataframe
        Count data projected to Euclidean space.

    """
    X = table.to_dataframe()
    X = X.sparse.to_dense()

    if transformation == "clr":
        X = zero_imputation(X, pseudo_count=pseudo_count)
        X = normalize(X)
        X = log_transform(X, transformation=transformation)

    elif transformation == "mclr":
        X = normalize(X)
        X = log_transform(X, transformation=transformation)

    else:
        raise ValueError(
            "Unknown transformation name, use clr and not %r" % transformation
        )

    X = pd.DataFrame(X, columns=X.columns, index=X.index)

    taxa = pd.DataFrame(taxonomy.view(pd.Series))
    ASV_names = X.index.values

    for i in ASV_names:
        if i in taxa.index:
            name = taxa.loc[i, "Taxon"]
            if name not in X.index:  # avoid duplicated names
                X = X.rename(index={i: name})

    X = X.T  # p, N

    if add_metadata:
        numeric_md_cols = sample_metadata.filter_columns(column_type='numeric')
        md = numeric_md_cols.to_dataframe()

        if md.isnull().values.any():
            warnings.warn("Missing values are imputed with 0!")
            md = md.fillna(0)

        if scale_metadata:
            scaler = preprocessing.StandardScaler().fit(md)
            md_scaled = scaler.transform(md)
            md = pd.DataFrame(md_scaled, index=md.index, columns=md.columns)

        result = X.join(md)
    else:
        result = X

    result = result.T  # N, p

    return result


def calculate_covariance(table: pd.DataFrame, method: str = "scaled", bias: bool = True) -> pd.DataFrame:
    """
    A function calculating covariance matrix.

    Parameters
    ----------
    table: pd.Dataframe
        A dataframe with transformed microbiome data.
        See 'transform_features()' method for transformation options.
    method: str
        If 'unscaled', calculates covariance with "np.cov()", see Numpy documentation for details.
        If 'scaled', scales covariance dataframe with the square root of its diagonal, i.e. X_ij/sqrt(d_i*d_j),
        see (2.4) in https://fan.princeton.edu/papers/09/Covariance.pdf.
    bias: boolean, optional
        If 'True', normalize covariance estimation by the number of samples N,
        if "False" - by N-1, see Numpy documentation for details.

    Returns
    -------
    covariance matrix: pd.Dataframe
        A squared positive semi-definite matrix required for solving Graphical Lasso problem.

    """
    S = np.cov(table.values, bias=bias)

    if method == "unscaled":
        print("Calculate {0} covariance matrices S".format(method))
        result = S

    elif method == "scaled":
        print("Calculate {0} covariance (correlation) matrices S".format(method))
        result = scale_array_by_diagonal(S)

    else:
        raise ValueError('Given covariance calculation method is not supported.')

    result = pd.DataFrame(result, index=table.index, columns=table.index)

    return result


def build_groups(tables: Table, check_groups: bool = True) -> np.ndarray:
    """
    Building groups for solving the Group Graphical Lasso problem
    where not all instances have the same number of dimensions,
    i.e. some variables are present in some instances and not in others.

    Parameters
    ----------
    tables: list(biom.Table)
        A list with tables of count microbiome data.
    check_groups: bool
        GGLasso function to check a bookkeeping group penalty matrix G.

    Returns
    -------
    G: ndarray
    A bookkeeping group penalty matrix G, see Non-conforming case in GGLasso docs.

    """
    columns_dict = dict()
    dataframes_p_N = list()
    p_arr = list()
    num_samples = list()

    i = 0
    for table in tables:
        df = table.to_dataframe()

        dataframes_p_N.append(df)  # (p_variables, N_samples) required shape of dataframe
        p_arr.append(df.shape[0])  # number of variables
        num_samples.append(df.shape[1])  # number of samples

        columns_dict[i] = df.columns.values.tolist()
        i += 1

    all_names = set()
    for columns in columns_dict.values():
        for name in columns:
            all_names.add(name)

    non_conforming_problem = False

    for k in range(0, len(columns_dict)):
        diff = all_names.difference(columns_dict[k])
        if len(diff) > 0:
            non_conforming_problem = True

    if non_conforming_problem:

        ix_exist, ix_location = construct_indexer(dataframes_p_N)
        G = create_group_array(ix_exist, ix_location)

        if check_groups:
            check_G(G, p_arr)
            print("Dimensions p_k: ", p_arr)
            print("Sample sizes N_k: ", num_samples)
            print("Number of groups found: ", G.shape[1])

        return G

    else:
        print("All datasets have exactly the same number of features.")


def solve_SGL(S: np.ndarray, N: list, latent: bool = None, model_selection: bool = None,
              lambda1: list = None, mu1: list = None, lambda1_mask: list = None):
    """
    Solve Single Graphical Lasso (SGL) problem, see Friedman et al. (2007).

    Parameters
    ----------
    S: np.ndarray
        A squared positive semi-definite matrix.
    N: list
        A number of data samples.
    latent: boolean, optional
        If 'True', solve SGL accounting for latent variables, see Chandrasekaran et al. (2012).
    model_selection: boolean, optional
        If 'True', run a model selection procedure over a grid of hyperparameters, i.e., lambda, mu.
        eBIC is used for selecting the best performing model, see Foygel and Drton (2010).
    lambda1: list
        A list of non-negative regularization hyperparameters 'lambda1',
        'lambda1' accounts for sparsity level in SGL solution.
    mu1: list
        A list of non-negative regularization hyperparameters 'mu1',
        'mu1' accounts for L component in SGL solution to be a low-rank.
    lambda1_mask: list
        A non-negative, symmetric matrix, 'lambda1' is multiplied element-wise with this matrix.

    Returns
    -------
    solution: glasso_problem object
        Contains the solution, i.e. Omega, Theta, X (and L if ``latent=True``) after termination.
        All arrays are of shape (p,p).

    """
    if model_selection:
        print("\tDD MODEL SELECTION:")
        modelselect_params = {'lambda1_range': lambda1, 'mu1_range': mu1, 'lambda1_mask': lambda1_mask}
        P = glasso_problem(S, N=N, latent=latent)
        P.model_selection(modelselect_params=modelselect_params)

        boundary_lambdas = check_lambda_path(P)
        if boundary_lambdas:
            warnings.warn("lambda is on the edge of the interval, the solution might have not reached global minimum!")
    else:
        print("\tWITH LAMBDA={0} and MU={1}".format(lambda1, mu1))
        P = glasso_problem(S, N=N, reg_params={'lambda1': lambda1, "mu1": mu1, 'lambda1_mask': lambda1_mask},
                           latent=latent)
        P.solve()

    return P


def solve_MGL(S: np.ndarray, N: list, reg: str, latent: bool = None, model_selection: bool = None,
              lambda1: list = None, lambda2: list = None, mu1: list = None):
    """
    Solve Multiple  Graphical Lasso (MGL) problem, see Danaher et al. (2013).

    Parameters
    ----------
    S: np.ndarray
        Array of K squared positive semi-definite matrices.
    N: list
        A number of data samples.
    reg: str
        Choose either ’GGL’: Group Graphical Lasso or ’FGL’: Fused Graphical Lasso.
    latent: boolean, optional
        If 'True', solve MGL accounting for latent variables, see Tomasi et al. (2018).
    model_selection: boolean, optional
        If 'True', run a model selection procedure over a grid of hyperparameters, i.e., lambda, mu.
        eBIC is used for selecting the best performing model, see Foygel and Drton (2010).
    lambda1: list
        A list of non-negative regularization hyperparameters 'lambda1',
        'lambda1' accounts for sparsity level in within the groups.
    lambda2: list
        A list of non-negative regularization hyperparameters 'lambda2',
        'lambda2' accounts for sparsity level in across the groups.
    mu1: list
        A list of non-negative low-rank regularization hyperparameters 'mu1',
        Only needs to be specified if 'latent=True'.

    Returns
    -------
    solution: glasso_problem object
        Contains the solution, i.e. Omega, Theta, X (and L if ``latent=True``) after termination.
        All arrays are of shape (K, p,p).

    """
    if model_selection:
        print("\tDD MODEL SELECTION:")
        modelselect_params = {'lambda1_range': lambda1, 'lambda2_range': lambda2, 'mu1_range': mu1}
        P = glasso_problem(S, N=N, latent=latent, reg=reg)
        P.model_selection(modelselect_params=modelselect_params)

        boundary_lambdas = check_lambda_path(P, mgl_problem=True)
        if boundary_lambdas:
            warnings.warn("The solution might have not reached global minimum!")
    else:
        print("\tWITH LAMBDA1={0}, LAMBDA2={1} and MU={2}".format(lambda1, lambda2, mu1))
        P = glasso_problem(S, N=N, reg_params={'lambda1': lambda1, 'lambda2': lambda2, "mu1": mu1},
                           latent=latent, reg=reg)
        P.solve()

    return P


def solve_non_conforming(S: np.ndarray, N: list, G: list, latent: bool = None, model_selection: bool = None,
                         lambda1: list = None, lambda2: list = None, mu1: list = None):
    """
    Solve the Group Graphical Lasso problem where not all instances have the same number of dimensions,
    i.e. some variables are present in some instances and not in others.
    A group sparsity penalty is applied to all pairs of variables present in multiple instances.

    Parameters
    ----------
    S: np.ndarray
        Array of K squared positive semi-definite matrices.
    N: list
        A number of data samples.
    G: list
        Bookkeeping array containing information where the respective entries for each group can be found.
    latent: boolean, optional
        If 'True', solve MGL accounting for latent variables, see Tomasi et al. (2018).
    model_selection: boolean, optional
        If 'True', run a model selection procedure over a grid of hyperparameters, i.e., lambda, mu.
        eBIC is used for selecting the best performing model, see Foygel and Drton (2010).
    lambda1: list
        A list of non-negative regularization hyperparameters 'lambda1',
        'lambda1' accounts for sparsity level in within the groups.
    lambda2: list
        A list of non-negative regularization hyperparameters 'lambda2',
        'lambda2' accounts for sparsity level in across the groups.
    mu1: list
        A list of non-negative low-rank regularization hyperparameters 'mu1',
        Only needs to be specified if 'latent=True'.

    Returns
    -------
    solution: glasso_problem object
        Contains the solution, i.e. Omega, Theta, X (and L if ``latent=True``) after termination.
        All elements are dictionaries with keys 1,...,K and (p_k,p_k)-arrays as values.

    """
    if model_selection:
        print("\tDD MODEL SELECTION:")
        modelselect_params = {'lambda1_range': lambda1, 'lambda2_range': lambda2, 'mu1_range': mu1}
        P = glasso_problem(S, N=N, G=G, latent=latent, reg='GGL')
        P.model_selection(modelselect_params=modelselect_params)

        boundary_lambdas = check_lambda_path(P, mgl_problem=True)
        if boundary_lambdas:
            warnings.warn("lambda is on the edge of the interval, the solution might have not reached global minimum!")
    else:
        print("\tWITH LAMBDA1={0}, LAMBDA2={1} and MU={2}".format(lambda1, lambda2, mu1))
        P = glasso_problem(S, N=N, G=G, reg_params={'lambda1': lambda1, 'lambda2': lambda2, "mu1": mu1},
                           latent=latent, reg='GGL')
        P.solve()

    return P


def solve_problem(covariance_matrix: pd.DataFrame, n_samples: list, latent: bool = None, non_conforming: bool = None,
                  lambda1_min: float = None, lambda1_max: float = None, n_lambda1: int = 1,
                  lambda2_min: float = None, lambda2_max: float = None, n_lambda2: int = 1,
                  mu1_min: float = None, mu1_max: float = None, n_mu1: int = 1, adapt_lambda1: list = None,
                  group_array: list = None, reg: str = 'GGL') -> glasso_problem:
    """
    Solve Graphical Lasso problem.

    Parameters
    ----------

    covariance_matrix: list
        Array of K covariance matrices.
    n_samples: list
        A number of data samples.
    latent: boolean, optional
        If 'True', solve MGL accounting for latent variables, see Tomasi et al. (2018).
    non_conforming: boolean, optional
        Solve the Group Graphical Lasso problem where not all instances have the same number of dimensions,
        i.e. some variables are present in some instances and not in others.
    lambda2_min: float
        A minimal non-negative regularization hyperparameter 'lambda2';
        'lambda2' accounts for sparsity level in across the groups.
    lambda2_max: float
         A maximal of non-negative regularization hyperparameters 'lambda2';
        'lambda2' accounts for sparsity level in across the groups.
    n_lambda2: int
        A range of 'lambda1'.
    lambda1_min: float
        A minimal non-negative regularization hyperparameter 'lambda1';
        'lambda1' accounts for sparsity level in within the groups.
    lambda1_max: float
        A maximal non-negative regularization hyperparameter 'lambda1';
        'lambda1' accounts for sparsity level in within the groups.
    n_lambda1: int
        A range of 'lambda1'.
    mu1_min: float
        A minimal of non-negative low-rank regularization hyperparameters 'mu1';
        Only needs to be specified if 'latent=True'.
    mu1_max: float
        A maximal of non-negative low-rank regularization hyperparameters 'mu1';
        Only needs to be specified if 'latent=True'.
    n_mu1: int
        A range of 'mu1'.
    adapt_lambda1: list, optional
        List of elements and associated weights which will be multiplied by 'lambda1' during regularization;
        The 'lambda1' parameter is multiplied element-wise with list. Only available for SGL;
        List must be provided in the format: (entry, weight), where entry is a string, and weight is a float;
        For example, ph 0.01, es 0.5, Ammoniphilus 0.7 Solibacteriales 1.
        Note, in the example below we set the weight of "Solibacteriales" to 1, AFTER setting "es" to 0.5,
        otherwise, both entries would have weight 0.5.
    group_array: list, optional
        Bookkeeping array containing information where the respective entries for each group can be found.
    reg: str
        Choose either ’GGL’: Group Graphical Lasso or ’FGL’: Fused Graphical Lasso.

    Returns
    -------
    solution: glasso_problem object
        Contains the solution, i.e. Omega, Theta, X (and L if ``latent=True``) after termination.
        All elements are dictionaries with keys 1,...,K and (p_k,p_k)-arrays as values.

    """
    S = np.array(covariance_matrix)
    S = if_2d_array(S)

    n_samples = list_to_array(n_samples)

    h_params = get_hyperparameters(lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                                   lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                                   mu1_min=mu1_min, mu1_max=mu1_max, n_mu1=n_mu1)

    model_selection = h_params["model_selection"]
    lambda1, lambda2, mu1 = h_params["lambda1"], h_params["lambda2"], h_params["mu1"]

    if adapt_lambda1 is not None:
        lambda1_mask = get_lambda_mask(adapt_lambda1=adapt_lambda1, covariance_matrix=covariance_matrix)

    # if 2d array => solve SGL
    if S.ndim == 2:

        if latent:
            print("\n----SOLVING SINGLE GRAPHICAL LASSO PROBLEM WITH LATENT VARIABLES-----")

            P = solve_SGL(S=S, N=n_samples, latent=latent, model_selection=model_selection, lambda1=lambda1, mu1=mu1,
                          lambda1_mask=lambda1_mask)

        else:
            print("----SOLVING SINGLE GRAPHICAL LASSO PROBLEM-----")

            P = solve_SGL(S=S, N=n_samples, latent=latent, model_selection=model_selection, lambda1=lambda1, mu1=mu1,
                          lambda1_mask=lambda1_mask)

    # if 3d array => solve MGL
    elif S.ndim == 3:

        if non_conforming:

            if latent:
                print("\n----SOLVING NON-CONFORMING PROBLEM WITH LATENT VARIABLES-----")

                P = solve_non_conforming(S=S, N=n_samples, G=group_array, latent=latent,
                                         model_selection=model_selection,
                                         lambda1=lambda1, lambda2=lambda2, mu1=mu1)
            else:
                print("\n----SOLVING NON-CONFORMING PROBLEM-----")

                P = solve_non_conforming(S=S, N=n_samples, G=group_array, latent=latent,
                                         model_selection=model_selection,
                                         lambda1=lambda1, lambda2=lambda2, mu1=mu1)

        else:
            if latent:
                print("\n----SOLVING {0} PROBLEM WITH LATENT VARIABLES-----".format(reg))

                P = solve_MGL(S=S, N=n_samples, reg=reg, latent=latent, model_selection=model_selection,
                              lambda1=lambda1, lambda2=lambda2, mu1=mu1)
            else:
                print("\n----SOLVING {0} PROBLEM-----".format(reg))

                P = solve_MGL(S=S, N=n_samples, reg=reg, latent=latent, model_selection=model_selection,
                              lambda1=lambda1, lambda2=lambda2, mu1=mu1)

    labels = list(covariance_matrix.columns)
    labels_range = range(len(labels))
    P.__dict__["labels"] = dict(zip(labels_range, labels))

    return P
