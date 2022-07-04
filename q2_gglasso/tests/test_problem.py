import unittest
import numpy as np
from q2_gglasso.utils import if_equal_dict, if_non_conforming
from gglasso.problem import glasso_problem
from gglasso.problem import GGLassoEstimator
from gglasso.helper.data_generation import generate_precision_matrix, sample_covariance_matrix
from gglasso.helper.ext_admm_helper import create_group_array, construct_indexer

try:
    from q2_gglasso._func import solve_problem

except ImportError:
    raise ImportWarning('Qiime2 not installed.')


class TestUtil(unittest.TestCase):
    p = 32
    N = 10
    K = 3
    M = 4
    B = int(p / M)

    Sigma, Theta = generate_precision_matrix(p=p, M=1, style='erdos', prob=0.1, seed=1234)
    S, sample = sample_covariance_matrix(Sigma, N)

    # nonconforming case
    non_Sigma, non_Theta = generate_precision_matrix(p=p, M=M, style='powerlaw', gamma=2.8, prob=0.1, seed=3456)
    all_obs, non_S = if_non_conforming(Sigma=non_Sigma, N=N, p=p, K=K, B=B)
    ix_exist, ix_location = construct_indexer(list(all_obs.values()))
    G = create_group_array(ix_exist, ix_location, min_inst=K - 1)

    S_SGL = S
    S_MGL = np.array(K * [S])

    # for single solution
    lambda1 = 0.05
    lambda2 = 0.1
    mu1 = 0.15

    # for model selection
    lambda1_range = np.logspace(0, -2, 5)
    lambda2_range = np.logspace(0, -2, 5)
    mu1_range = np.logspace(0, -1, 2)

    ### test for single hyperparameters
    def test_SGL(self, S=S_SGL, N=N, lambda1=lambda1):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1}, N=N, latent=False)
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1=[lambda1])
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_SGL_low(self, S=S_SGL, N=N, lambda1=lambda1, mu1=mu1):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, "mu1": mu1}, N=N, latent=True)
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1=[lambda1], mu1=[mu1], latent=True)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_GGL(self, S=S_MGL, N=N, lambda1=lambda1, lambda2=lambda2):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, 'lambda2': lambda2}, N=N, latent=False, reg='GGL')
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1=[lambda1], lambda2=[lambda2])
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_GGL_low(self, S=S_MGL, N=N, lambda1=lambda1, lambda2=lambda2, mu1=mu1):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, 'lambda2': lambda2, 'mu1': mu1},
                               N=N, latent=True, reg='GGL')
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True,
                             lambda1=[lambda1], lambda2=[lambda2], mu1=[mu1])
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_FGL(self, S=S_MGL, N=N, lambda1=lambda1, lambda2=lambda2):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, 'lambda2': lambda2}, N=N, latent=False, reg='FGL')
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1=[lambda1], lambda2=[lambda2], reg='FGL')
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_FGL_low(self, S=S_MGL, N=N, lambda1=lambda1, lambda2=lambda2, mu1=mu1):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, 'lambda2': lambda2, 'mu1': mu1},
                               N=N, latent=True, reg='FGL')
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True, reg='FGL',
                             lambda1=[lambda1], lambda2=[lambda2], mu1=[mu1])
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    ### test for model selection
    def test_modelselect_SGL(self, S=S_SGL, N=N, lambda1_range=lambda1_range, lambda2_range=lambda2_range):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range}
        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=False)
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1=lambda1_range, lambda2=lambda2_range)
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_SGL_low(self, S=S_SGL, N=N, lambda1_range=lambda1_range, lambda2_range=lambda2_range,
                                 mu1_range=mu1_range):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range, 'mu1_range': mu1_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=True, reg='GGL')
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True,
                             lambda1=lambda1_range, lambda2=lambda2_range, mu1=mu1_range)
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_GGL(self, S=S_MGL, N=N, lambda1_range=lambda1_range, lambda2_range=lambda2_range):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=False, reg='GGL')
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1=lambda1_range, lambda2=lambda2_range, reg='GGL')
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_GGL_low(self, S=S_MGL, N=N,
                                 lambda1_range=lambda1_range, lambda2_range=lambda2_range, mu1_range=mu1_range):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range, 'mu1_range': mu1_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=True, reg='GGL')
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True, reg='GGL',
                             lambda1=lambda1_range, lambda2=lambda2_range, mu1=mu1_range)
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_FGL(self, S=S_MGL, N=N, lambda1_range=lambda1_range, lambda2_range=lambda2_range):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=False, reg='FGL')
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1=lambda1_range, lambda2=lambda2_range, reg='FGL')
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_FGL_low(self, S=S_MGL, N=N,
                                 lambda1_range=lambda1_range, lambda2_range=lambda2_range, mu1_range=mu1_range):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range, 'mu1_range': mu1_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=True, reg='FGL')
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True, reg='FGL',
                             lambda1=lambda1_range, lambda2=lambda2_range, mu1=mu1_range)
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    # def test_non_conform_modelselect_GGL(self, S=non_S, N=N, G=G,
    #                                      lambda1_range=lambda1_range, lambda2_range=lambda2_range):
    #     modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range}
    #
    #     P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, G=G, latent=False, reg='GGL')
    #     P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
    #     ebic_org = P_org.modelselect_stats["BIC"]
    #
    #     P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1=lambda1_range, lambda2=lambda2_range, reg='GGL')
    #     ebic_q2 = P_q2.modelselect_stats["BIC"]
    #
    #     x = if_equal_dict(ebic_org, ebic_q2)
    #
    #     self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")


if __name__ == '__main__':
    unittest.main()
