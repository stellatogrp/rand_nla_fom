"""Small integration tests for shared experiment constructions."""

import unittest

import numpy as np

from randomized_sketch_descent import primal_dual_drs
from scripts.problems import inequality_qp_optimum, slack_quadratic_prox
from scripts.standard import make_lp, solve
from scripts.theta_sweep import POLICIES, THETAS, sigma_for


class ExperimentSmokeTests(unittest.TestCase):
    def test_standard_lp(self):
        problem = make_lp(seed=7)
        result = solve(problem, 0.4, 0.8, 0.3, 0.8, "schur_cg")
        self.assertTrue(result.converged)
        self.assertLess(abs(problem.objective(result.primal) - problem.optimum), 3e-6)
        self.assertLess(np.linalg.norm(problem.matrix @ result.primal - problem.rhs), 3e-7)

    def test_inequality_qp_slack_formulation(self):
        rng = np.random.default_rng(18)
        constraints, variables = 3, 8
        matrix = rng.standard_normal((constraints, variables))
        factor = rng.standard_normal((variables, variables))
        hessian = factor.T @ factor + 0.5 * np.eye(variables)
        feasible = rng.standard_normal(variables)
        rhs = matrix @ feasible + rng.uniform(0.1, 0.3, constraints)
        slack_matrix = np.hstack((matrix, np.eye(constraints)))
        optimum = inequality_qp_optimum(hessian, matrix, rhs)
        result = primal_dual_drs(
            slack_quadratic_prox(hessian),
            slack_matrix,
            rhs,
            gamma_x=0.15,
            gamma_lambda=0.8,
            sigma=0.2,
            theta=1.3,
            tolerance=1e-9,
            max_iterations=5_000,
        )
        primal, slack = result.primal[:variables], result.primal[variables:]
        self.assertTrue(result.converged)
        self.assertGreaterEqual(slack.min(), -1e-12)
        self.assertLess(np.linalg.norm(matrix @ primal + slack - rhs), 2e-8)
        self.assertLess(abs(float(primal @ hessian @ primal) - optimum), 5e-6)


    def test_theta_sweep_tolerances_stay_admissible(self):
        # Theorem 5 couples the two parameters, so every (theta, sigma) the
        # sweep generates must satisfy sigma < (2 - theta) / 2.
        for theta in THETAS:
            for policy in POLICIES:
                sigma = sigma_for(policy, float(theta))
                with self.subTest(theta=theta, policy=policy):
                    self.assertGreaterEqual(sigma, 0.0)
                    self.assertLess(sigma, (2 - float(theta)) / 2)


if __name__ == "__main__":
    unittest.main()
