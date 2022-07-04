import numpy as np
import zarr
import pandas as pd

from gglasso.helper.utils import normalize as norm
from gglasso.helper.utils import log_transform as trans


def flatten_array(x):
    x = np.array(x)
    x = x.flatten()
    return x


def if_none_to_list(x):
    if x is None:
        x = [x]
    return x


def list_to_array(x=list):
    if isinstance(x, list):
        x = np.array(x)
    return x


def if_equal_dict(a, b):
    x = True
    for key in a.keys():
        if a[key].all() == b[key].all():
            continue
        else:
            x = False
    return x


def if_2d_array(x=np.ndarray):
    #  if 3d array of shape (1,p,p),
    #  make it 2d array of shape (p,p).
    if x.shape[0] == 1:
        x = x[0, :]
    return x


def if_all_none(lambda1, lambda2, mu1):
    if lambda1 is None and lambda2 is None and mu1 is None:
        lambda1 = np.logspace(0, -3, 10)
        lambda2 = np.logspace(-1, -4, 5)
        mu1 = np.logspace(2, -1, 10)

        print("Setting default hyperparameters:")
        print('\tlambda1 range: [%s]' % ', '.join(map(str, lambda1)))
        print('\tlambda2 range: [%s]' % ', '.join(map(str, lambda2)))
        print('\tmu1 range: [%s]\n' % ', '.join(map(str, mu1)))

    return lambda1, lambda2, mu1


def if_model_selection(lambda1, lambda2, mu1):
    lambda1 = if_none_to_list(lambda1)
    lambda2 = if_none_to_list(lambda2)
    mu1 = if_none_to_list(mu1)

    model_selection = True
    if (len(lambda1) == 1) and len(lambda2) == 1 and (len(mu1) == 1):
        model_selection = False

    if model_selection:
        if None in lambda1:
            lambda1 = np.logspace(0, -3, 10)
        if None in lambda2:
            lambda2 = np.logspace(-1, -4, 5)
        if None in mu1:
            mu1 = np.logspace(2, -1, 10)
    else:
        lambda1 = np.array(lambda1).item()
        lambda2 = np.array(lambda2).item()
        mu1 = np.array(mu1).item()

    h_params = {"model_selection": model_selection,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "mu1": mu1}

    return h_params


def normalize(X):
    """
    transforms to the simplex
    X should be of a pd.DataFrame of form (p,N)
    """
    return X / X.sum(axis=0)


def geometric_mean(x, positive=False):
    """
    calculates the geometric mean of a vector
    """
    assert not np.all(x == 0)

    if positive:
        x = x[x > 0]
    a = np.log(x)
    g = np.exp(a.sum() / len(a))
    return g


def log_transform(X, transformation=str, eps=0.1):
    """
    log transform, scaled with geometric mean
    X should be a pd.DataFrame of form (p,N)
    """
    if transformation == "clr":
        assert not np.any(X.values == 0), "Add pseudo count before using clr"
        g = X.apply(geometric_mean)
        Z = np.log(X / g)
    elif transformation == "mclr":
        # g = X.apply(geometric_mean, positive=True)
        # Z = np.log(X / g)
        # Z = Z + abs(np.amin(Z.values)) + eps
        g = X.apply(geometric_mean, positive=True)
        X_pos = X[X > 0]
        Z = np.log(X_pos / g)
        Z = Z + abs(np.nanmin(Z.values)) + eps
        Z = Z.fillna(0)
    return Z


def remove_biom_header(file_path):
    with open(str(file_path), 'r') as fin:
        data = fin.read().splitlines(True)
    with open(str(file_path), 'w') as fout:
        fout.writelines(data[1:])


def if_no_model_selection(lambda1, lambda2=None, mu1=None):
    mu1 = if_none_to_list(mu1)
    lambda2 = if_none_to_list(lambda2)

    model_selection = True
    if (len(lambda1) == 1) and len(lambda2) == 1 and (len(mu1) == 1):
        model_selection = False

    return model_selection


def single_hyperparameters(model_selection, lambda1, lambda2=None, mu1=None):
    if model_selection is False:
        lambda1 = np.array(lambda1).item()
        lambda2 = np.array(lambda2).item()
        mu1 = np.array(mu1).item()
    return lambda1, lambda2, mu1


def to_zarr(obj, name, root, first=True):
    """
    Function for converting a GGLasso object to a zarr file, a with tree structue.
    """
    # name 'S' is dedicated for some internal usage in zarr notation and cannot be accessed as a key while reading
    if name == "S":
        name = 'covariance'

    if isinstance(obj, dict):
        if first:
            zz = root
        else:
            zz = root.create_group(name)

        for key, value in obj.items():
            to_zarr(value, key, zz, first=False)

    elif isinstance(obj, (list, set)):
        root.create_dataset(name, data=obj, shape=len(obj))

    elif isinstance(obj, (np.ndarray, pd.DataFrame)):
        root.create_dataset(name, data=obj, shape=obj.shape)

    elif isinstance(obj, (str, bool, float, int)):
        to_zarr(np.array(obj), name, root, first=False)

    elif isinstance(obj, (np.str_, np.bool_, np.int64, np.float64)):
        to_zarr(np.array(obj), name, root, first=False)

    elif isinstance(obj, type(None)):
        pass
    else:
        to_zarr(obj.__dict__, name, root, first=first)


def PCA(X, L, inverse=True):
    sig, V = np.linalg.eigh(L)

    # sort eigenvalues in descending order
    sig = sig[::-1]
    V = V[:, ::-1]

    ind = np.argwhere(sig > 1e-9)

    if inverse:
        loadings = V[:, ind] @ np.diag(np.sqrt(1 / sig[ind]))
    else:
        loadings = V[:, ind] @ np.diag(np.sqrt(sig[ind]))

    # compute the projection
    zu = X.values @ loadings

    return zu, loadings, np.round(sig[ind].squeeze(), 3)
