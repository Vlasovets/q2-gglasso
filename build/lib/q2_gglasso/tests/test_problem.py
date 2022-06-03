import unittest
import zarr
import q2_gglasso as q2g
import pandas as pd
import numpy as np
from gglasso.problem import glasso_problem

try:
    from q2_gglasso._func import solve_problem

except ImportError:
    raise ImportWarning('Qiime2 not installed.')


table = pd.DataFrame([[1, 1, 7, 3],
                      [2, 6, 2, 4],
                      [5, 5, 3, 3],
                      [3, 2, 8, 1]],
                     index=['s1', 's2', 's3', 's4'],
                     columns=['o1', 'o2', 'o3', 'o4'])

S = np.cov(table.values)

P = glasso_problem(S, N=1, reg_params={'lambda1': [0.5, 0.01], "mu1": [0.5, 0.1]}, latent=True)
P.model_selection()

P.__dict__

zipfile = str("problem.zip")
store = zarr.ZipStore(zipfile, mode="w")
root = zarr.open(store=store)
q2g.to_zarr(P.__dict__, "problem", root)
store.close()




# def test_zarr_format():
