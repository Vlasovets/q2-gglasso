"""Tests for q2_gglasso plugin problem solving functionality.

This module tests the various problem solving methods and their integration
with QIIME2.
"""

import unittest
import numpy as np
import pandas as pd

from q2_gglasso.utils import if_equal_dict
from gglasso.problem import glasso_problem
from gglasso.problem import GGLassoEstimator
from gglasso.helper.data_generation import (
    generate_precision_matrix,
    sample_covariance_matrix,
)
from gglasso.helper.ext_admm_helper import (
    create_group_array,
    construct_indexer,
    check_G,
)

try:
    from q2_gglasso._func import solve_problem

except ImportError:
    raise ImportWarning("Qiime2 not installed.")


class TestUtil(unittest.TestCase):
    """Test utilities for problem solving in q2-gglasso."""

    p = 32
    N = 10
    K = 3
    rtol = 1e-2
    atol = 1e-2
    ebic_diff = 5

    Sigma, Theta = generate_precision_matrix(
        p=p, M=1, style="erdos", prob=0.1, seed=1234
    )
    S, sample = sample_covariance_matrix(Sigma, N)

    S_SGL = S
    S_MGL = np.array(K * [S])

    # non_conforming
    M = 8
    B = int(p / 8)

    Sigma, Theta = generate_precision_matrix(
        p=p, M=M, style="powerlaw", gamma=2.8, prob=0.1, seed=3456
    )
    all_obs = dict()
    S_non_conforming = list()
    for k in np.arange(K):
        _, obs = sample_covariance_matrix(Sigma, N, seed=456)

        # drop the k-th block starting from the end
        drop_range = np.arange(p - (k + 1) * B, p - k * B)
        all_obs[k] = pd.DataFrame(obs).drop(drop_range, axis=0)
        cov = np.cov(all_obs[k], bias=True)
        S_non_conforming.append(cov)

    ix_exist, ix_location = construct_indexer(list(all_obs.values()))
    G = create_group_array(ix_exist, ix_location)
    check_G(G, p)

    # for single solution
    lambda1_min = 0.01
    lambda1_max = 1
    lambda2_min = 0.1
    lambda2_max = 1
    n_lambda1 = 3
    n_lambda2 = 2
    mu1_min = 0.1
    mu1_max = 3
    n_mu1 = 1

    # create random weights for SGL
    labels = [f"taxon{i+1}" for i in range(p)]
    weights_df = pd.DataFrame(abs(S_SGL), index=labels, columns=labels)
    diag = np.abs(np.diag(weights_df.values))
    weights = []
    for label, value in zip(labels, diag):
        weights.extend([label, float(value)])

    #  test for single hyperparameters

    def test_SGL(
        self,
        S=S_SGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        n_lambda1=1,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        equal=False,
    ):
        """Test Sparse Graphical Lasso implementation."""
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            reg_params={"lambda1": P_q2.reg_params["lambda1"]},
            N=N,
            latent=False,
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal, msg="Solutions from GGLasso and q2-gglasso are not identical"
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    def test_SGL_low(
        self,
        S=S_SGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_mu1=1,
        n_lambda1=1,
        equal=False,
        equal_low=False,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
            latent=True,
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            reg_params={
                "lambda1": P_q2.reg_params["lambda1"],
                "mu1": P_q2.reg_params["mu1"],
            },
            N=N,
            latent=True,
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        equal_low = np.allclose(
            P_org.solution.lowrank_,
            P_q2.solution.lowrank_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal,
            msg="Solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            equal_low,
            msg="Low-rank solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    def test_GGL(
        self,
        S=S_MGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=2,
        n_lambda2=2,
        equal=False,
    ):
        """Test Group Graphical Lasso implementation."""
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            reg_params={
                "lambda1": P_q2.reg_params["lambda1"],
                "lambda2": P_q2.reg_params["lambda2"],
            },
            N=N,
            latent=False,
            reg="GGL",
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal, msg="Solutions from GGLasso and q2-gglasso are not identical"
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    def test_GGL_low(
        self,
        S=S_MGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_mu1=2,
        n_lambda1=2,
        n_lambda2=2,
        equal=False,
        equal_low=False,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            latent=True,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            reg_params={
                "lambda1": P_q2.reg_params["lambda1"],
                "lambda2": P_q2.reg_params["lambda2"],
                "mu1": P_q2.reg_params["mu1"],
            },
            N=N,
            latent=True,
            reg="GGL",
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        equal_low = np.allclose(
            P_org.solution.lowrank_,
            P_q2.solution.lowrank_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal,
            msg="Solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            equal_low,
            msg="Low-rank solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    def test_FGL(
        self,
        S=S_MGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=1,
        n_lambda2=1,
        equal=False,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
            reg="FGL",
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            reg_params={
                "lambda1": P_q2.reg_params["lambda1"],
                "lambda2": P_q2.reg_params["lambda2"],
            },
            N=N,
            latent=False,
            reg="FGL",
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal, msg="Solutions from GGLasso and q2-gglasso are not identical"
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    def test_FGL_low(
        self,
        S=S_MGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_mu1=1,
        n_lambda1=1,
        n_lambda2=1,
        equal=False,
        equal_low=False,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            latent=True,
            reg="FGL",
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            reg_params={
                "lambda1": P_q2.reg_params["lambda1"],
                "lambda2": P_q2.reg_params["lambda2"],
                "mu1": P_q2.reg_params["mu1"],
            },
            N=N,
            latent=True,
            reg="FGL",
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        equal_low = np.allclose(
            P_org.solution.lowrank_,
            P_q2.solution.lowrank_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal,
            msg="Solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            equal_low,
            msg="Low-rank solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    def test_non_conforming(
        self,
        S=np.array(S_non_conforming),
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=1,
        n_lambda2=1,
        G=G,
        equal=True,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            group_array=G,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
            non_conforming=True,
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            N=N,
            G=G,
            reg_params={
                "lambda1": P_q2.reg_params["lambda1"],
                "lambda2": P_q2.reg_params["lambda2"],
            },
            latent=False,
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal, msg="Solutions from GGLasso and q2-gglasso are not identical"
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    def test_non_conforming_low(
        self,
        S=np.array(S_non_conforming),
        N=N,
        G=G,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_mu1=1,
        n_lambda1=1,
        n_lambda2=1,
        equal=False,
        equal_low=False,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            group_array=G,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
            non_conforming=True,
            latent=True,
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            N=N,
            G=G,
            reg_params={
                "lambda1": P_q2.reg_params["lambda1"],
                "lambda2": P_q2.reg_params["lambda2"],
                "mu1": P_q2.reg_params["mu1"],
            },
            latent=True,
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        equal_low = np.allclose(
            P_org.solution.lowrank_,
            P_q2.solution.lowrank_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal,
            msg="Solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            equal_low,
            msg="Low-rank solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    def test_SGL_mask(
        self,
        S=S_SGL,
        N=N,
        weights=weights,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=1,
        equal=False,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            weights=weights,
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            reg_params={
                "lambda1": P_q2.reg_params["lambda1"],
                "lambda1_mask": P_q2.modelselect_params["lambda1_mask"],
            },
            N=N,
            latent=False,
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal, msg="Solutions from GGLasso and q2-gglasso are not identical"
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    def test_SGL_mask_low(
        self,
        S=S_SGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        weights=weights,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=1,
        n_mu1=1,
        equal=False,
        equal_low=False,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
            weights=weights,
            latent=True,
        )
        ebic_q2 = GGLassoEstimator.calc_ebic(P_q2.solution)

        P_org = glasso_problem(
            S=S,
            reg_params={
                "lambda1": P_q2.reg_params["lambda1"],
                "mu1": P_q2.reg_params["mu1"],
                "lambda1_mask": P_q2.modelselect_params["lambda1_mask"],
            },
            N=N,
            latent=True,
        )
        P_org.solve()
        ebic_org = GGLassoEstimator.calc_ebic(P_org.solution)

        equal = np.allclose(
            P_org.solution.precision_,
            P_q2.solution.precision_,
            rtol=rtol,
            atol=atol,
        )

        equal_low = np.allclose(
            P_org.solution.lowrank_,
            P_q2.solution.lowrank_,
            rtol=rtol,
            atol=atol,
        )

        self.assertTrue(
            equal,
            msg="Solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            equal_low,
            msg="Low-rank solutions from GGLasso and q2-gglasso are not identical within tolerance.",
        )
        self.assertTrue(
            abs(ebic_org - ebic_q2) <= ebic_diff,
            msg=f"eBIC values differ beyond tolerance of ±2: {ebic_org} vs {ebic_q2}",
        )

    #  test for model selection

    def test_modelselect_SGL(
        self,
        S=S_SGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        n_lambda1=n_lambda1,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
    ):
        # Solve using q2-gglasso
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"]
        }
        P_org = glasso_problem(
            S=S, reg_params=modelselect_params, N=N, latent=False
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        # Compare eBIC dictionaries
        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg=f"eBIC values differ between solvers:\n{ebic_org}\nvs\n{ebic_q2}",
        )

        # best_lambda_org = P_org.modelselect_stats["BEST"]["lambda1"]
        # best_lambda_q2 = P_q2.modelselect_stats["BEST"]["lambda1"]

        # self.assertEqual(
        #     best_lambda_org,
        #     best_lambda_q2,
        #     msg=f"Best lambda differs:\nOriginal: {best_lambda_org}\nQ2: {best_lambda_q2}",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg="Precision matrices are not sufficiently close between solvers.",
        # )

    def test_modelselect_SGL_low(
        self,
        S=S_SGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        n_lambda1=1,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        n_mu1=1,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            latent=True,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        # Run original solver
        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"],
            "mu1_range": P_q2.modelselect_params["mu1_range"],
        }
        P_org = glasso_problem(
            S=S, reg_params=modelselect_params, N=N, latent=True
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg=f"eBIC dictionaries differ:\nOriginal: {ebic_org}\nQ2: {ebic_q2}",
        )

        # lambda_org = P_org.modelselect_stats["BEST"]["lambda1"]
        # lambda_q2 = P_q2.modelselect_stats["BEST"]["lambda1"]
        # self.assertEqual(
        #     lambda_org,
        #     lambda_q2,
        #     msg=f"Best lambda mismatch:\nOriginal: {lambda_org}\nQ2: {lambda_q2}",
        # )

        # mu_org = P_org.modelselect_stats["BEST"]["mu1"]
        # mu_q2 = P_q2.modelselect_stats["BEST"]["mu1"]

        # self.assertEqual(
        #     mu_org,
        #     mu_q2,
        #     msg=f"Best mu mismatch:\nOriginal: {mu_org}\nQ2: {mu_q2}",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg=(
        #         f"Precision matrices are not numerically close.\n"
        #         f"Max abs diff: {np.max(np.abs(P_org.solution.precision_ - P_q2.solution.precision_))}"
        #     ),
        # )

    def test_modelselect_GGL(
        self,
        S=S_MGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=2,
        n_lambda2=2,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            reg="GGL",
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"],
            "lambda2_range": P_q2.modelselect_params["lambda2_range"],
        }
        P_org = glasso_problem(
            S=S, reg_params=modelselect_params, N=N, latent=False, reg="GGL"
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg=f"eBIC dictionaries differ:\nOriginal: {ebic_org}\nQ2: {ebic_q2}",
        )

        # lambda1_org = P_org.modelselect_stats["BEST"]["lambda1"]
        # lambda1_q2 = P_q2.modelselect_stats["BEST"]["lambda1"]

        # self.assertEqual(
        #     lambda1_org,
        #     lambda1_q2,
        #     msg=f"Best lambda1 mismatch:\nOriginal: {lambda1_org}\nQ2: {lambda1_q2}",
        # )

        # lambda2_org = P_org.modelselect_stats["BEST"]["lambda2"]
        # lambda2_q2 = P_q2.modelselect_stats["BEST"]["lambda2"]

        # self.assertEqual(
        #     lambda2_org,
        #     lambda2_q2,
        #     msg=f"Best lambda2 mismatch:\nOriginal: {lambda2_org}\nQ2: {lambda2_q2}",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg=(
        #         f"Precision matrices are not numerically close.\n"
        #         f"Max abs diff: {np.max(np.abs(P_org.solution.precision_ - P_q2.solution.precision_))}"
        #     ),
        # )

    def test_modelselect_GGL_low(
        self,
        S=S_MGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        n_lambda1=2,
        n_lambda2=2,
        n_mu1=2,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            latent=True,
            reg="GGL",
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"],
            "lambda2_range": P_q2.modelselect_params["lambda2_range"],
            "mu1_range": P_q2.modelselect_params["mu1_range"],
        }
        P_org = glasso_problem(
            S=S, reg_params=modelselect_params, N=N, latent=True, reg="GGL"
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg=f"eBIC mismatch:\nOriginal: {ebic_org}\nQ2: {ebic_q2}",
        )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda1"],
        #     P_q2.modelselect_stats["BEST"]["lambda1"],
        #     msg="Best lambda1 from original and QIIME2 solver differ",
        # )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda2"],
        #     P_q2.modelselect_stats["BEST"]["lambda2"],
        #     msg="Best lambda2 from original and QIIME2 solver differ",
        # )

        # mu_trace_org = np.array(P_org.reg_params["mu1"])
        # mu_trace_q2 = np.array(P_q2.reg_params["mu1"])
        # self.assertTrue(
        #     np.allclose(mu_trace_org, mu_trace_q2, rtol=rtol, atol=atol),
        #     msg=f"mu1 trace mismatch:\nOriginal: {mu_trace_org}\nQ2: {mu_trace_q2}",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg="Precision matrices differ between original and QIIME2 solvers",
        # )

    def test_modelselect_FGL(
        self,
        S=S_MGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        n_lambda1=2,
        n_lambda2=2,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            reg="FGL",
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"],
            "lambda2_range": P_q2.modelselect_params["lambda2_range"],
        }

        P_org = glasso_problem(
            S=S, reg_params=modelselect_params, N=N, latent=False, reg="FGL"
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg=f"eBIC mismatch:\nOriginal: {ebic_org}\nQ2: {ebic_q2}",
        )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda1"],
        #     P_q2.modelselect_stats["BEST"]["lambda1"],
        #     msg="Best lambda1 from original and Q2 solver differ",
        # )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda2"],
        #     P_q2.modelselect_stats["BEST"]["lambda2"],
        #     msg="Best lambda2 from original and Q2 solver differ",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg="Precision matrices differ between original and Q2 solvers",
        # )

    def test_modelselect_FGL_low(
        self,
        S=S_MGL,
        N=N,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        n_lambda1=2,
        n_lambda2=2,
        n_mu1=2,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
            latent=True,
            reg="FGL",
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"],
            "lambda2_range": P_q2.modelselect_params["lambda2_range"],
            "mu1_range": P_q2.modelselect_params["mu1_range"],
        }

        P_org = glasso_problem(
            S=S,
            reg_params=modelselect_params,
            N=N,
            latent=True,
            reg="FGL",
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg=f"eBIC mismatch:\nOriginal: {ebic_org}\nQ2: {ebic_q2}",
        )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda1"],
        #     P_q2.modelselect_stats["BEST"]["lambda1"],
        #     msg="Best lambda1 differs between GGLasso and q2-gglasso",
        # )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda2"],
        #     P_q2.modelselect_stats["BEST"]["lambda2"],
        #     msg="Best lambda2 differs between GGLasso and q2-gglasso",
        # )

        # self.assertTrue(
        #     np.array_equal(P_org.reg_params["mu1"], P_q2.reg_params["mu1"]),
        #     msg="mu1 trace differs between GGLasso and q2-gglasso",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg="Precision matrices differ between GGLasso and q2-gglasso",
        # )

    def test_modelselect_non_conforming(
        self,
        S=S_non_conforming,
        N=N,
        G=G,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=2,
        n_lambda2=2,
    ):
        P_q2 = solve_problem(
            covariance_matrix=np.array(S),
            n_samples=N,
            group_array=G,
            non_conforming=True,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"],
            "lambda2_range": P_q2.modelselect_params["lambda2_range"],
        }

        P_org = glasso_problem(
            S=np.array(S),
            reg_params=modelselect_params,
            N=N,
            G=G,
            latent=False,
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver",
        )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda1"],
        #     P_q2.modelselect_stats["BEST"]["lambda1"],
        #     msg="Best chosen lambda1 differs between GGLasso and q2-gglasso",
        # )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda2"],
        #     P_q2.modelselect_stats["BEST"]["lambda2"],
        #     msg="Best chosen lambda2 differs between GGLasso and q2-gglasso",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg="Precision matrices differ between GGLasso and q2-gglasso",
        # )

    def test_modelselect_non_conforming_low(
        self,
        S=S_non_conforming,
        N=N,
        G=G,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        lambda2_min=lambda2_min,
        lambda2_max=lambda2_max,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=2,
        n_lambda2=2,
        n_mu1=2,
    ):
        P_q2 = solve_problem(
            covariance_matrix=np.array(S),
            n_samples=N,
            latent=True,
            group_array=G,
            non_conforming=True,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
            lambda2_min=lambda2_min,
            lambda2_max=lambda2_max,
            n_lambda2=n_lambda2,
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]
        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"],
            "lambda2_range": P_q2.modelselect_params["lambda2_range"],
            "mu1_range": P_q2.modelselect_params["mu1_range"],
        }

        P_org = glasso_problem(
            S=np.array(S),
            reg_params=modelselect_params,
            N=N,
            G=G,
            latent=True,
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver",
        )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda1"],
        #     P_q2.modelselect_stats["BEST"]["lambda1"],
        #     msg="Best chosen lambda1 from GGLasso and q2-gglasso are not the same",
        # )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda2"],
        #     P_q2.modelselect_stats["BEST"]["lambda2"],
        #     msg="Best chosen lambda2 from GGLasso and q2-gglasso are not the same",
        # )

        # self.assertTrue(
        #     np.array_equal(P_org.reg_params["mu1"], P_q2.reg_params["mu1"]),
        #     msg="mu1 trace from GGLasso and q2-gglasso are not identical",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg="Solutions from GGLasso and q2-gglasso are not identical",
        # )

    def test_modelselect_SGL_mask(
        self,
        S=S_SGL,
        N=N,
        weights=weights,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=2,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            weights=weights,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"],
            "lambda1_mask": P_q2.modelselect_params["lambda1_mask"],
        }

        P_org = glasso_problem(
            S=S, reg_params=modelselect_params, N=N, latent=False
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver",
        )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda1"],
        #     P_q2.modelselect_stats["BEST"]["lambda1"],
        #     msg="Best chosen lambda from GGLasso and q2-gglasso are not the same",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg="Solutions from GGLasso and q2-gglasso are not identical",
        # )

    def test_modelselect_SGL_mask_low(
        self,
        S=S_SGL,
        N=N,
        weights=weights,
        lambda1_min=lambda1_min,
        lambda1_max=lambda1_max,
        mu1_min=mu1_min,
        mu1_max=mu1_max,
        rtol=rtol,
        atol=atol,
        ebic_diff=ebic_diff,
        n_lambda1=2,
        n_mu1=2,
    ):
        P_q2 = solve_problem(
            covariance_matrix=S,
            n_samples=N,
            latent=True,
            weights=weights,
            mu1_min=mu1_min,
            mu1_max=mu1_max,
            n_mu1=n_mu1,
            lambda1_min=lambda1_min,
            lambda1_max=lambda1_max,
            n_lambda1=n_lambda1,
        )
        ebic_q2 = P_q2.modelselect_stats["BIC"]

        modelselect_params = {
            "lambda1_range": P_q2.modelselect_params["lambda1_range"],
            "mu1_range": P_q2.modelselect_params["mu1_range"],
            "lambda1_mask": P_q2.modelselect_params["lambda1_mask"],
        }

        P_org = glasso_problem(
            S=S, reg_params=modelselect_params, N=N, latent=True
        )
        P_org.model_selection(
            modelselect_params=modelselect_params, method="eBIC"
        )
        ebic_org = P_org.modelselect_stats["BIC"]

        self.assertTrue(
            if_equal_dict(ebic_org, ebic_q2),
            msg="eBIC of QIIME2 solver is different from eBIC of GGLasso solver",
        )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["lambda1"],
        #     P_q2.modelselect_stats["BEST"]["lambda1"],
        #     msg="Best chosen lambda from GGLasso and q2-gglasso are not the same",
        # )

        # self.assertEqual(
        #     P_org.modelselect_stats["BEST"]["mu1"],
        #     P_q2.modelselect_stats["BEST"]["mu1"],
        #     msg="Best chosen mu from GGLasso and q2-gglasso are not the same",
        # )

        # self.assertTrue(
        #     np.allclose(
        #         P_org.solution.precision_,
        #         P_q2.solution.precision_,
        #         rtol=rtol,
        #         atol=atol,
        #     ),
        #     msg="Solutions from GGLasso and q2-gglasso are not identical",
        # )


if __name__ == "__main__":
    unittest.main()
