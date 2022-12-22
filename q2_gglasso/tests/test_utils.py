import unittest
import warnings
from gglasso.helper.data_generation import generate_precision_matrix, sample_covariance_matrix

from gglasso.problem import glasso_problem

try:
    from q2_gglasso._func import solve_problem

except ImportError:
    raise ImportWarning('Qiime2 not installed.')

class TestUtil(unittest.TestCase):

    def test_assertWarns(self):
        with self.assertWarnsRegex(Warning, r'lambda is on the edge of the interval'):
            p = 32
            N = 10
            K = 3

            Sigma, Theta = generate_precision_matrix(p=p, M=1, style='erdos', prob=0.1, seed=1234)
            S, sample = sample_covariance_matrix(Sigma, N)

            S_SGL = S

            lambda1 = 0.05
            P_org = glasso_problem(S=S_SGL, reg_params={'lambda1': lambda1}, N=N, latent=False)
            P_org.solve()
            #print("hi")

if __name__ == '__main__':
    unittest.main()



### test heatmap
### test biom editing

