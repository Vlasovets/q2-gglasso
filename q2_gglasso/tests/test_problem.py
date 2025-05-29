import unittest
import numpy as np
import pandas as pd

from q2_gglasso.utils import if_equal_dict
from gglasso.problem import glasso_problem
from gglasso.problem import GGLassoEstimator
from gglasso.helper.data_generation import generate_precision_matrix, sample_covariance_matrix
from gglasso.helper.ext_admm_helper import create_group_array, construct_indexer, check_G

try:
    from q2_gglasso._func import solve_problem

except ImportError:
    raise ImportWarning('Qiime2 not installed.')


class TestUtil(unittest.TestCase):
    p = 32
    N = 10
    K = 3

    Sigma, Theta = generate_precision_matrix(p=p, M=1, style='erdos', prob=0.1, seed=1234)
    S, sample = sample_covariance_matrix(Sigma, N)

    S_SGL = S
    S_MGL = np.array(K * [S])

    # non_conforming
    M = 8
    B = int(p / 8)

    Sigma, Theta = generate_precision_matrix(p=p, M=M, style='powerlaw', gamma=2.8, prob=0.1, seed=3456)
    all_obs = dict()
    S_non_conforming = list()
    for k in np.arange(K):
        _, obs = sample_covariance_matrix(Sigma, N, seed=456)

        # drop the k-th block starting from the end
        all_obs[k] = pd.DataFrame(obs).drop(np.arange(p - (k + 1) * B, p - k * B), axis=0)
        cov = np.cov(all_obs[k], bias=True)
        S_non_conforming.append(cov)

    ix_exist, ix_location = construct_indexer(list(all_obs.values()))
    G = create_group_array(ix_exist, ix_location)
    check_G(G, p)

    # for single solution
    lambda1 = 0.05
    lambda1_min = 0.05
    lambda1_max = 1
    lambda2_min = 0.1
    lambda2_max = 1
    n_lambda1 = 3
    n_lambda2 = 2
    mu1 = 0.15
    lambda1_mask = abs(S_SGL)

    # for model selection
    lambda1_range = np.linspace(lambda1_min, lambda1_max, n_lambda1)
    lambda2_range = np.linspace(lambda2_min, lambda2_max, n_lambda2)
    # lambda1_range = np.logspace(0, -2, 5)
    # lambda2_range = np.logspace(0, -2, 5)
    mu1_range = np.logspace(0, -1, 2)

    #  test for single hyperparameters
    def test_SGL(self, S=S_SGL, N=N, lambda1=lambda1_min, equal=False):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1}, N=N, latent=False)
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1_min=lambda1, lambda1_max=lambda1, n_lambda1=1)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_SGL_low(self, S=S_SGL, N=N, lambda1=lambda1_min, mu1=mu1, equal=False, equal_low=False):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, "mu1": mu1}, N=N, latent=True)
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1_min=lambda1, lambda1_max=lambda1, n_lambda1=1,
                             mu1=mu1, latent=True)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        if (P_org.solution.lowrank_ == P_q2.solution.lowrank_).all():
            equal_low = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical.")
        self.assertTrue(equal_low, msg="Low-rank solutions from GGLasso and q2-gglasso are not identical.")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver.")

    def test_GGL(self, S=S_MGL, N=N, lambda1=lambda1_min, lambda2=lambda2_min, equal=False):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, 'lambda2': lambda2}, N=N, latent=False, reg='GGL')
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1_min=lambda1, lambda1_max=lambda1, n_lambda1=1,
                             lambda2_min=lambda2, lambda2_max=lambda2, n_lambda2=1)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical.")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_GGL_low(self, S=S_MGL, N=N, lambda1=lambda1_min, lambda2=lambda2_min, mu1=mu1, equal=False, equal_low=False):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, 'lambda2': lambda2, 'mu1': mu1},
                               N=N, latent=True, reg='GGL')
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True,
                             lambda1_min=lambda1, lambda1_max=lambda1, n_lambda1=1,
                             lambda2_min=lambda2, lambda2_max=lambda2, n_lambda2=1, mu1=mu1)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        if (P_org.solution.lowrank_ == P_q2.solution.lowrank_).all():
            equal_low = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical.")
        self.assertTrue(equal_low, msg="Low-rank solutions from GGLasso and q2-gglasso are not identical.")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_FGL(self, S=S_MGL, N=N, lambda1=lambda1_min, lambda2=lambda2_min, equal=False):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, 'lambda2': lambda2}, N=N, latent=False, reg='FGL')
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1_min=lambda1, lambda1_max=lambda1, n_lambda1=1,
                             lambda2_min=lambda2, lambda2_max=lambda2, n_lambda2=1, reg='FGL')
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical.")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_FGL_low(self, S=S_MGL, N=N, lambda1=lambda1_min, lambda2=lambda2_min, mu1=mu1, equal=False, equal_low=False):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, 'lambda2': lambda2, 'mu1': mu1},
                               N=N, latent=True, reg='FGL')
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True, reg='FGL',
                             lambda1_min=lambda1, lambda1_max=lambda1, n_lambda1=1,
                             lambda2_min=lambda2, lambda2_max=lambda2, n_lambda2=1, mu1=mu1)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        if (P_org.solution.lowrank_ == P_q2.solution.lowrank_).all():
            equal_low = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical.")
        self.assertTrue(equal_low, msg="Low-rank solutions from GGLasso and q2-gglasso are not identical.")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_non_conforming(self, S=np.array(S_non_conforming), N=N, lambda1=lambda1_min, lambda2=lambda2_min, G=G, equal=True):
        P_org = glasso_problem(S=S, N=N, G=G, reg_params={'lambda1': lambda1, 'lambda2': lambda2}, latent=False)
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, group_array=G,
                             lambda1_min=lambda1, lambda1_max=lambda1, n_lambda1=1,
                             lambda2_min=lambda2, lambda2_max=lambda2, n_lambda2=1,
                             non_conforming=True)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical.")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_non_conforming_low(self, S=np.array(S_non_conforming), N=N, lambda1=lambda1_min, lambda2=lambda2_min, mu1=mu1, G=G,
                                equal=False, equal_low=False):
        P_org = glasso_problem(S=S, N=N, G=G, reg_params={'lambda1': lambda1, 'lambda2': lambda2, 'mu1': mu1},
                               latent=True)
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, group_array=G,
                             lambda1_min=lambda1, lambda1_max=lambda1, n_lambda1=1,
                             lambda2_min=lambda2, lambda2_max=lambda2, n_lambda2=1,
                             mu1=mu1, non_conforming=True, latent=True)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        if (P_org.solution.lowrank_ == P_q2.solution.lowrank_).all():
            equal_low = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical.")
        self.assertTrue(equal_low, msg="Low-rank solutions from GGLasso and q2-gglasso are not identical.")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_SGL_mask(self, S=S_SGL, N=N, lambda1=lambda1_min, lambda1_mask=lambda1_mask, equal=False):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, 'lambda1_mask': lambda1_mask}, N=N, latent=False)
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1_min=lambda1, lambda1_mask=lambda1_mask)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_SGL_mask_low(self, S=S_SGL, N=N, lambda1=lambda1_min, mu1=mu1, lambda1_mask=lambda1_mask,
                          equal=False, equal_low=False):
        P_org = glasso_problem(S=S, reg_params={'lambda1': lambda1, "mu1": mu1, "lambda1_mask": lambda1_mask},
                               N=N, latent=True)
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1_min=lambda1, mu1=mu1,
                             lambda1_mask=lambda1_mask, latent=True)
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        if (P_org.solution.lowrank_ == P_q2.solution.lowrank_).all():
            equal_low = True

        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical.")
        self.assertTrue(equal_low, msg="Low-rank solutions from GGLasso and q2-gglasso are not identical.")
        self.assertEqual(ebic_org, ebic_q2, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver.")

    #  test for model selection
    def test_modelselect_SGL(self, S=S_SGL, N=N, lambda1_range=lambda1_range, equal=False, best_lambda=False,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1):

        modelselect_params = {'lambda1_range': lambda1_range}
        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=False)
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1_min=lambda1_min, lambda1_max=lambda1_max,
                             n_lambda1=n_lambda1)
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        if best_lambda_org == best_lambda_q2:
            best_lambda = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda, msg="Best chosen lambda  from GGLasso and q2-gglasso are not the same")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_SGL_low(self, S=S_SGL, N=N, lambda1_range=lambda1_range, mu1_range=mu1_range,
                                 lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                                 equal=False, best_lambda=False, best_mu=False):
        modelselect_params = {'lambda1_range': lambda1_range, 'mu1_range': mu1_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=True)
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max,
                             n_lambda1=n_lambda1, mu1=mu1_range)
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        best_mu_org = P_org.modelselect_stats["BEST"]['mu1']
        best_mu_q2 = P_q2.modelselect_stats["BEST"]['mu1']

        if best_mu_org == best_mu_q2:
            best_mu = True

        if best_lambda_org == best_lambda_q2:
            best_lambda = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda, msg="Best chosen lambda  from GGLasso and q2-gglasso are not the same")
        self.assertTrue(best_mu, msg="Best chosen mu  from GGLasso and q2-gglasso are not the same")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_GGL(self, S=S_MGL, N=N, lambda1_range=lambda1_range, lambda2_range=lambda2_range,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                             lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                             equal=False, best_lambda1=False, best_lambda2=False):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=False, reg='GGL')
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, reg='GGL',
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                             lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                             )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda1_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda1_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        best_lambda2_org = P_org.modelselect_stats["BEST"]['lambda2']
        best_lambda2_q2 = P_q2.modelselect_stats["BEST"]['lambda2']

        if best_lambda1_org == best_lambda1_q2:
            best_lambda1 = True

        if best_lambda2_org == best_lambda2_q2:
            best_lambda2 = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda1, msg="Best chosen lambda1 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(best_lambda2, msg="Best chosen lambda2 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_GGL_low(self, S=S_MGL, N=N, lambda1_range=lambda1_range, lambda2_range=lambda2_range,
                                 lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                                 lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                                 mu1_range=mu1_range, equal=False, best_lambda1=False, best_lambda2=False, mu_equal=False):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range, 'mu1_range': mu1_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=True, reg='GGL')
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True, reg='GGL', mu1=mu1_range,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                             lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                             )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda1_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda1_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        best_lambda2_org = P_org.modelselect_stats["BEST"]['lambda2']
        best_lambda2_q2 = P_q2.modelselect_stats["BEST"]['lambda2']

        # eBIC depends on lambda1 and lambda2, but not on of mu
        mu_trace_org = P_org.reg_params["mu1"]
        mu_trace_q2 = P_q2.reg_params["mu1"]

        if best_lambda1_org == best_lambda1_q2:
            best_lambda1 = True

        if best_lambda2_org == best_lambda2_q2:
            best_lambda2 = True

        if (mu_trace_org == mu_trace_q2).all():
            mu_equal = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda1, msg="Best chosen lambda1 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(best_lambda2, msg="Best chosen lambda2 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(mu_equal, msg="mu1 trace from GGLasso and q2-gglasso are not identical")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_FGL(self, S=S_MGL, N=N, lambda1_range=lambda1_range, lambda2_range=lambda2_range,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                             lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                             equal=False, best_lambda1=False, best_lambda2=False):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=False, reg='FGL')
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, reg='FGL',
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                             lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                             )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda1_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda1_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        best_lambda2_org = P_org.modelselect_stats["BEST"]['lambda2']
        best_lambda2_q2 = P_q2.modelselect_stats["BEST"]['lambda2']

        if best_lambda1_org == best_lambda1_q2:
            best_lambda1 = True

        if best_lambda2_org == best_lambda2_q2:
            best_lambda2 = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda1, msg="Best chosen lambda1 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(best_lambda2, msg="Best chosen lambda2 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_FGL_low(self, S=S_MGL, N=N, lambda1_range=lambda1_range, lambda2_range=lambda2_range,
                                 lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                                 lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                                 mu1_range=mu1_range, equal=False, best_lambda1=False, best_lambda2=False, mu_equal=False):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range, 'mu1_range': mu1_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=True, reg='FGL')
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True, reg='FGL', mu1=mu1_range,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                             lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                             )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda1_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda1_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        best_lambda2_org = P_org.modelselect_stats["BEST"]['lambda2']
        best_lambda2_q2 = P_q2.modelselect_stats["BEST"]['lambda2']

        # eBIC depends on lambda1 and lambda2, but not on of mu
        mu_trace_org = P_org.reg_params["mu1"]
        mu_trace_q2 = P_q2.reg_params["mu1"]

        if best_lambda1_org == best_lambda1_q2:
            best_lambda1 = True

        if best_lambda2_org == best_lambda2_q2:
            best_lambda2 = True

        if (mu_trace_org == mu_trace_q2).all():
            mu_equal = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda1, msg="Best chosen lambda1 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(best_lambda2, msg="Best chosen lambda2 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(mu_equal, msg="mu1 trace from GGLasso and q2-gglasso are not identical")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_non_conforming(self, S=np.array(S_non_conforming), N=N, G=G,
                                        lambda1_range=lambda1_range, lambda2_range=lambda2_range,
                                        lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                                        lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                                        equal=False, best_lambda1=False, best_lambda2=False):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, G=G, latent=False)
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, group_array=G, non_conforming=True,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                             lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                             )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda1_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda1_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        best_lambda2_org = P_org.modelselect_stats["BEST"]['lambda2']
        best_lambda2_q2 = P_q2.modelselect_stats["BEST"]['lambda2']

        if best_lambda1_org == best_lambda1_q2:
            best_lambda1 = True

        if best_lambda2_org == best_lambda2_q2:
            best_lambda2 = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda1, msg="Best chosen lambda1 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(best_lambda2, msg="Best chosen lambda2 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_non_conforming_low(
            self, S=np.array(S_non_conforming), N=N, G=G,
            lambda1_range=lambda1_range, lambda2_range=lambda2_range, mu1_range=mu1_range,
            lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
            lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
            equal=False, best_lambda1=False, best_lambda2=False, mu_equal=False):
        modelselect_params = {'lambda1_range': lambda1_range, 'lambda2_range': lambda2_range, 'mu1_range': mu1_range}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, G=G, latent=True)
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True, group_array=G, non_conforming=True, mu1=mu1_range,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                             lambda2_min=lambda2_min, lambda2_max=lambda2_max, n_lambda2=n_lambda2,
                             )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda1_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda1_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        best_lambda2_org = P_org.modelselect_stats["BEST"]['lambda2']
        best_lambda2_q2 = P_q2.modelselect_stats["BEST"]['lambda2']

        # eBIC depends on lambda1 and lambda2, but not on of mu
        mu_trace_org = P_org.reg_params["mu1"]
        mu_trace_q2 = P_q2.reg_params["mu1"]

        if best_lambda1_org == best_lambda1_q2:
            best_lambda1 = True

        if best_lambda2_org == best_lambda2_q2:
            best_lambda2 = True

        if (mu_trace_org == mu_trace_q2).all():
            mu_equal = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda1, msg="Best chosen lambda1 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(best_lambda2, msg="Best chosen lambda2 from GGLasso and q2-gglasso are not the same")
        self.assertTrue(mu_equal, msg="mu1 trace from GGLasso and q2-gglasso are not identical")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_SGL_mask(self, S=S_SGL, N=N, lambda1_range=lambda1_range, lambda1_mask=lambda1_mask,
                                  lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                                  equal=False, best_lambda=False):
        modelselect_params = {'lambda1_range': lambda1_range, "lambda1_mask": lambda1_mask}
        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=False)
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, lambda1_mask=lambda1_mask,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1
                             )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        if best_lambda_org == best_lambda_q2:
            best_lambda = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda, msg="Best chosen lambda  from GGLasso and q2-gglasso are not the same")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")

    def test_modelselect_SGL_mask_low(self, S=S_SGL, N=N, lambda1_range=lambda1_range, mu1_range=mu1_range,
                                      lambda1_mask=lambda1_mask, equal=False, best_lambda=False, best_mu=False,
                                      lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1,
                                      ):
        modelselect_params = {'lambda1_range': lambda1_range, 'mu1_range': mu1_range, "lambda1_mask": lambda1_mask}

        P_org = glasso_problem(S=S, reg_params=modelselect_params, N=N, latent=True)
        P_org.model_selection(modelselect_params=modelselect_params, method='eBIC')
        ebic_org = P_org.modelselect_stats["BIC"]

        P_q2 = solve_problem(covariance_matrix=S, n_samples=N, latent=True, mu1=mu1_range, lambda1_mask=lambda1_mask,
                             lambda1_min=lambda1_min, lambda1_max=lambda1_max, n_lambda1=n_lambda1
                             )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        x = if_equal_dict(ebic_org, ebic_q2)

        best_lambda_org = P_org.modelselect_stats["BEST"]['lambda1']
        best_lambda_q2 = P_q2.modelselect_stats["BEST"]['lambda1']

        best_mu_org = P_org.modelselect_stats["BEST"]['mu1']
        best_mu_q2 = P_q2.modelselect_stats["BEST"]['mu1']

        if best_mu_org == best_mu_q2:
            best_mu = True

        if best_lambda_org == best_lambda_q2:
            best_lambda = True

        if (P_org.solution.precision_ == P_q2.solution.precision_).all():
            equal = True

        self.assertTrue(best_lambda, msg="Best chosen lambda  from GGLasso and q2-gglasso are not the same")
        self.assertTrue(best_mu, msg="Best chosen mu  from GGLasso and q2-gglasso are not the same")
        self.assertTrue(equal, msg="Solutions from GGLasso and q2-gglasso are not identical")
        self.assertTrue(x, msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver")


if __name__ == '__main__':
    unittest.main()
