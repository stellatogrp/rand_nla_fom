"""Small integration tests for shared experiment constructions."""

import unittest

import numpy as np

from experiments.lasso import make_lasso
from experiments.problems import (
    equality_qp_optimum,
    hybrid_objective,
    inequality_qp_optimum,
    lp_optimum,
    slack_quadratic_prox,
)
from experiments.standard import make_lp, solve
from experiments.sweep_problems import (
    make_equality_qp,
    make_large_equality_qp,
    make_large_inequality_qp,
    make_large_lp,
)
from experiments.theta_sweep import POLICIES, THETAS, sigma_for
from rand_nla_fom import primal_dual_drs


def _planted(name, maker, seed):
    """Rebuild a planted instance and its minimizer's objective value.

    The makers draw the minimizer first, so replaying their random stream
    recovers it: the matrix draws, then the solution and multiplier.
    """
    problem = maker("ill-conditioned", 20, 60, seed=seed)
    rng = np.random.default_rng(seed)
    matrix = problem.matrix[:, : problem.dimension]
    if name == "LP":
        rng.standard_normal((20, 40))
        rng.standard_normal((20, 20))
        solution = np.zeros(60)
        solution[:20] = rng.uniform(0.2, 1.0, 20)
        solution /= solution.sum()
        return name, problem, problem.objective(solution)
    rng.standard_normal((60, 20))
    rng.standard_normal((20, 20))
    if name == "equality QP":
        solution = rng.standard_normal(60) / np.sqrt(60)
        return name, problem, problem.objective(solution)
    diagonal = np.geomspace(0.3, 3.0, 60)
    multiplier = rng.uniform(0.2, 1.0, 20) / np.sqrt(20)
    solution = -(matrix.T @ multiplier) / (2 * diagonal)
    return name, problem, problem.objective(np.r_[solution, np.zeros(20)])


class ExperimentSmokeTests(unittest.TestCase):
    def test_standard_lp(self):
        problem = make_lp(seed=7)
        result = solve(problem, 0.4, 0.8, 0.3, 0.8, "schur_cg")
        self.assertTrue(result.converged)
        self.assertLess(abs(problem.objective(result.primal) - problem.optimum), 3e-6)
        residual = problem.matrix @ result.primal - problem.rhs
        self.assertLess(np.linalg.norm(residual), 3e-7)

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

    def test_lasso_split_form(self):
        problem = make_lasso("correlated", samples=30, features=90, nonzeros=5, seed=3)
        # The split objective at a feasible point is the LASSO objective, so
        # the certified reference must agree with a direct evaluation.
        columns = problem.matrix[:, : problem.features]
        result = primal_dual_drs(
            problem.prox,
            problem.matrix,
            problem.rhs,
            gamma_x=1.0,
            gamma_lambda=1.0,
            sigma=0.2,
            theta=1.0,
            objective=problem.objective,
            tolerance=1e-10,
            max_iterations=20_000,
        )
        self.assertTrue(result.converged)
        primal = result.primal[: problem.features]
        value = hybrid_objective(columns, problem.rhs, problem.coefficients)(primal)
        self.assertLess(abs(value - problem.optimum), 1e-7)
        self.assertLess(abs(problem.objective(result.primal) - problem.optimum), 1e-7)
        self.assertGreaterEqual(problem.matrix_condition, 1.0)
        self.assertGreater(problem.design_condition, problem.matrix_condition)

    def test_cvxpy_references_match_exact_values(self):
        # The equality QP has a closed-form optimum through its KKT system,
        # and the planted instances know their optimum by construction; the
        # CVXPY reference must reproduce both far below the target accuracy.
        rng = np.random.default_rng(9)
        constraints, variables = 5, 12
        matrix = rng.standard_normal((constraints, variables))
        factor = rng.standard_normal((variables, variables))
        hessian = factor.T @ factor + np.eye(variables)
        linear_term = rng.standard_normal(variables)
        rhs = rng.standard_normal(constraints)
        kkt = np.block(
            [[hessian, matrix.T], [matrix, np.zeros((constraints, constraints))]]
        )
        solution = np.linalg.solve(kkt, np.r_[-linear_term, rhs])[:variables]
        exact = 0.5 * solution @ hessian @ solution + linear_term @ solution
        self.assertLess(
            abs(equality_qp_optimum(hessian, linear_term, matrix, rhs) - exact),
            1e-9 * max(1.0, abs(exact)),
        )
        # The planted generators know their minimizer; the CVXPY value must
        # agree with the objective there.
        for name, problem, planted_value in (
            _planted("equality QP", make_large_equality_qp, 41),
            _planted("LP", make_large_lp, 42),
            _planted("inequality QP", make_large_inequality_qp, 43),
        ):
            with self.subTest(problem=name):
                self.assertLess(
                    abs(problem.optimum - planted_value),
                    1e-9 * max(1.0, abs(planted_value)),
                )
        # A standard-form LP against its planted primal--dual pair.
        rng = np.random.default_rng(12)
        matrix = rng.standard_normal((4, 10))
        solution = np.zeros(10)
        solution[:4] = rng.uniform(0.5, 1.0, 4)
        multiplier = rng.standard_normal(4)
        cost = -matrix.T @ multiplier
        cost[4:] += rng.uniform(0.2, 1.0, 6)
        self.assertLess(
            abs(lp_optimum(cost, matrix, matrix @ solution) - cost @ solution), 1e-9
        )

    def test_sweep_matrices_have_the_stated_spectrum_and_couple_rows(self):
        # The sweeps prescribe kappa(A); the Schur complement must still couple
        # the rows, or the Kaczmarz methods solve it in one sweep.
        for name, problem in (
            ("equality QP", make_equality_qp("ill-conditioned")),
            ("large equality QP", make_large_equality_qp("ill-conditioned", 20, 60)),
            ("large LP", make_large_lp("ill-conditioned", 20, 60)),
            (
                "large inequality QP",
                make_large_inequality_qp("ill-conditioned", 20, 60),
            ),
        ):
            with self.subTest(problem=name):
                matrix = problem.matrix[:, : problem.dimension]
                singular = np.linalg.svd(matrix, compute_uv=False)
                self.assertAlmostEqual(singular[0] / singular[-1], 100.0, places=6)
                schur = np.eye(matrix.shape[0]) + matrix @ matrix.T
                off_diagonal = schur - np.diag(np.diag(schur))
                self.assertGreater(
                    np.linalg.norm(off_diagonal) / np.linalg.norm(schur), 0.1
                )

    def test_hybrid_penalty_reference_is_certified(self):
        problem = make_lasso(
            "independent",
            samples=30,
            features=90,
            nonzeros=5,
            seed=5,
            penalty=(0.5, 0.05, 1.0, 0.5),
            variant="all four terms",
        )
        self.assertGreaterEqual(problem.solution_nonzeros, 0)
        columns = problem.matrix[:, : problem.features]
        # The method with an inexact solver must reach the certified optimum.
        result = primal_dual_drs(
            problem.prox,
            problem.matrix,
            problem.rhs,
            sigma=0.2,
            objective=problem.objective,
            tolerance=1e-10,
            max_iterations=20_000,
        )
        self.assertTrue(result.converged)
        value = hybrid_objective(columns, problem.rhs, problem.coefficients)(
            result.primal[: problem.features]
        )
        self.assertLess(abs(value - problem.optimum), 1e-7)

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
