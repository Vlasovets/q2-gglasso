import unittest
import pandas as pd
import numpy as np

try:
    from q2_gglasso._func import calculate_covariance

except ImportError:
    raise ImportWarning('Qiime2 not installed.')


class TestUtil(unittest.TestCase):
    def test_semi_positive_definite(self):
        table = pd.DataFrame([[1, 1, 7, 3],
                              [2, 6, 2, 4],
                              [5, 5, 3, 3],
                              [3, 2, 8, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['o1', 'o2', 'o3', 'o4'])

        tol_up = 10e-6
        tol_low = 10e-14

        for method in ["scaled", "unscaled"]:
            S = calculate_covariance(table, method=method)  # covariance matrix S

            I = np.eye(S.shape[0])  # identity matrix of the same shape as S
            eigenvalues = np.linalg.eigvals(S)
            min_positive = np.select(eigenvalues > 0, eigenvalues)  # select the minimum positive eigenvalue

            if min_positive > tol_up:
                min_positive = tol_up
            elif min_positive < tol_low:
                min_positive = tol_low

            S = S + I * min_positive  # scale the diagonal preserving the variance

            try:
                np.linalg.cholesky(S)
            except self.assertRaises(np.linalg.LinAlgError):
                np.linalg.cholesky(S)


if __name__ == '__main__':
    unittest.main()
