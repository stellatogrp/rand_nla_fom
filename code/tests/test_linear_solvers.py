"""Regression tests for both inner linear-system formulations."""

import unittest

import numpy as np

from rand_nla_fom import primal_dual_drs, quadratic_prox
from tests._fixtures import equality_qp


class LinearSolverTests(unittest.TestCase):
    def test_each_solver_recovers_the_qp_solution(self):
        matrix, rhs, hessian, linear_term, expected = equality_qp()
        for solver in (
            "schur_cg",
            "schur_block_kaczmarz",
            "schur_randomized_kaczmarz",
            "coupled_gmres",
        ):
            with self.subTest(solver=solver):
                result = primal_dual_drs(
                    quadratic_prox(hessian, linear_term),
                    matrix,
                    rhs,
                    gamma_x=0.15,
                    gamma_lambda=0.4,
                    sigma=0.2,
                    theta=1.2,
                    linear_solver=solver,
                    block_size=1,
                    store_iterates=solver != "schur_block_kaczmarz",
                    tolerance=1e-9,
                    max_iterations=5_000,
                )
                self.assertTrue(result.converged)
                np.testing.assert_allclose(
                    result.primal, expected, atol=3e-8, rtol=3e-8
                )
                self.assertTrue(np.all(result.relative_error_ratios <= 0.2 + 1e-12))
                if solver == "schur_block_kaczmarz":
                    self.assertEqual(
                        result.primal_iterates.shape,
                        (0, expected.size),
                    )


if __name__ == "__main__":
    unittest.main()
