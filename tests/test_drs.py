"""Tests for the public Douglas--Rachford API."""

import unittest

import numpy as np

from randomized_sketch_descent import primal_dual_drs, quadratic_prox

from tests._fixtures import equality_qp


class PrimalDualDRSTests(unittest.TestCase):
    def test_invalid_parameters(self):
        identity = np.eye(2)
        zero = np.zeros(2)
        invalid = (
            ("theta", 0.0),
            ("theta", 2.0),
            ("sigma", -0.1),
            ("sigma", 0.5),
            ("gamma_x", 0.0),
            ("gamma_lambda", 0.0),
        )
        for name, value in invalid:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                primal_dual_drs(
                    lambda vector, _: vector,
                    identity,
                    zero,
                    **{name: value},
                )

        for block_size in (0, 3):
            with self.subTest(block_size=block_size), self.assertRaises(ValueError):
                primal_dual_drs(
                    lambda vector, _: vector,
                    identity,
                    zero,
                    block_size=block_size,
                )

    def test_adaptive_parameters_remain_admissible(self):
        matrix, rhs, hessian, linear_term, _ = equality_qp()
        result = primal_dual_drs(
            quadratic_prox(hessian, linear_term),
            matrix,
            rhs,
            gamma_x=0.15,
            gamma_lambda=0.4,
            sigma=0.2,
            theta=1.0,
            adaptive=True,
            adaptation_interval=5,
            tolerance=1e-8,
            max_iterations=1_000,
        )
        self.assertTrue(result.converged)
        self.assertTrue(np.all((0 < result.theta_history) & (result.theta_history < 2)))
        self.assertTrue(np.all(result.sigma_history >= 0))
        self.assertTrue(
            np.all(result.sigma_history < (2 - result.theta_history) / 2)
        )

    def test_quadratic_prox_rejects_indefinite_hessian(self):
        with self.assertRaises(ValueError):
            quadratic_prox(np.diag([1.0, -1.0]), np.zeros(2))


if __name__ == "__main__":
    unittest.main()
