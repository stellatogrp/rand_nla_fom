"""Checks on the solvers, the theory constants, and the CSV pipeline at small size."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from numerical_experiments import (
    admm,
    aggregate,
    plots,
    ppm,
    problems,
    study,
    theory,
    trajectory_2d,
)

DELTA = 1e-9


class TheoryTests(unittest.TestCase):
    def test_optimal_sigma_is_the_minimizer_and_below_its_bound(self):
        for ratio in (0.1, 1.0, 10.0, 100.0):
            for b in (1, 5):
                alpha, c0 = theory.inner_constants(ratio)
                for deterministic in (True, False):
                    root = theory.optimal_sigma(alpha, c0, b, deterministic)
                    c = theory.check_constant(alpha, b, deterministic)
                    grid = np.linspace(1e-4, 0.999, 100_000)
                    values = [theory.g_coefficient(s, c, c0) for s in grid]
                    self.assertLess(abs(root - grid[int(np.argmin(values))]), 2e-4)
                    self.assertLess(
                        root, theory.sigma_upper_bound(alpha, c0, b, deterministic)
                    )
                    self.assertTrue(0.01 < root < 0.09)

    def test_bounds_are_monotone_in_their_arguments(self):
        alpha, c0 = theory.inner_constants(10.0)
        self.assertEqual(theory.per_step_bound(0.5, alpha, 3), 0)
        self.assertEqual(theory.per_step_bound(1e6, alpha, 3) % 3, 0)
        self.assertLessEqual(
            theory.total_bound_deterministic(10.0, 5, 0.1, c0, alpha, 1, 1.0, DELTA),
            theory.total_bound_deterministic(10.0, 6, 0.1, c0, alpha, 1, 1.0, DELTA),
        )
        b = theory.best_integer_b(2.4, lambda b: (b - 2.4) ** 2)
        self.assertEqual(b, 2)

    def test_exact_b_cost_is_below_the_relaxed_one_and_minimized(self):
        alpha, c0 = theory.inner_constants(10.0)
        for ratio in (0.05, 1.0, 4.0, 20.0):
            c_in, c_test = 1.0, ratio
            costs = [
                theory.b_cost_per_outer(b, c_test, c_in, 0.1, c0, alpha, relaxed=False)
                for b in range(1, 200)
            ]
            for b, exact in enumerate(costs, start=1):
                relaxed = theory.b_cost_per_outer(b, c_test, c_in, 0.1, c0, alpha)
                self.assertLessEqual(exact, relaxed + 1e-12)
            best = theory.optimal_b_exact(c_test, c_in, 0.1, c0, alpha, b_max=199)
            self.assertEqual(best, 1 + int(np.argmin(costs)))
        self.assertEqual(theory.optimal_b_exact(0.0, 1.0, 0.1, c0, alpha), 1)


class QPTests(unittest.TestCase):
    def setUp(self):
        self.qp = problems.make_qp(seed=3, n=50, m=15, kappa=50.0)
        self.truth = self.qp.ground_truth()
        self.L = self.qp.smoothness()
        self.c = self.L / 10.0
        self.projector = problems.AffineProjector(self.qp.A, self.qp.b)
        self.factor = admm.factor_subproblem(self.qp, self.c)

    def test_ground_truth_matches_the_planted_solution(self):
        self.assertLess(self.truth.planted_error, 1e-8)
        self.assertLess(self.truth.stationarity, 1e-8)
        self.assertLess(abs(self.L - np.linalg.eigvalsh(self.qp.Q).max()), 1e-8)

    def test_exact_admm_converges(self):
        trace = admm.exact_admm(self.qp, self.c, self.factor, self.projector, DELTA)
        self.assertTrue(trace.converged)
        self.assertLess(np.linalg.norm(trace.iterates[-1] - self.truth.x), 1e-6)
        self.assertLessEqual(trace.z_norm[-1], DELTA)

    def test_inexact_admm_converges_and_satisfies_the_bounds(self):
        alpha, c0 = theory.inner_constants(self.L / self.c)
        r0 = admm.exact_residual(
            self.qp, self.c, self.factor, self.projector, np.zeros(self.qp.n)
        )
        w_star = self.truth.x + self.qp.gradient(self.truth.x) / self.c
        for sigma, b in ((0.05, 1), (0.5, 3)):
            trace = admm.inexact_admm(
                self.qp, self.c, self.L, self.projector, sigma, b, DELTA
            )
            self.assertTrue(trace.converged)
            self.assertLess(np.linalg.norm(trace.iterates[-1] - self.truth.x), 1e-6)
            inner = np.array(trace.inner)
            # Per-step bound of Theorem 4.3 with the exact fixed-point residual.
            for k, w in enumerate(trace.w):
                residual = admm.exact_residual(
                    self.qp, self.c, self.factor, self.projector, w
                )
                beta = theory.beta(sigma, c0, trace.e0_norm[k], residual, DELTA)
                self.assertLessEqual(inner[k], theory.per_step_bound(beta, alpha, b))
            # Total bound of Theorem 4.3 and the telescoping bound of Theorem 3.1.
            beta0 = theory.beta(sigma, c0, trace.e0_norm[0], r0, DELTA)
            total = theory.total_bound_deterministic(
                beta0, trace.k_delta, sigma, c0, alpha, b, r0, DELTA
            )
            self.assertLessEqual(inner.sum(), total)
            dist0 = np.linalg.norm(w_star)
            self.assertLessEqual(
                np.sum(np.array(trace.z_norm[:-1]) ** 2),
                (1 + sigma) / (1 - sigma) * dist0**2,
            )


class HuberTests(unittest.TestCase):
    def setUp(self):
        self.fit = problems.make_huber(seed=5, n=40, m=60)
        self.truth = self.fit.ground_truth()
        self.L = self.fit.smoothness()
        self.gamma = 10.0 / self.L

    def test_objective_and_gradient_are_consistent(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(self.fit.n)
        direction = rng.standard_normal(self.fit.n)
        step = 1e-6
        numerical = (
            self.fit.objective(x + step * direction)
            - self.fit.objective(x - step * direction)
        ) / (2 * step)
        self.assertLess(abs(numerical - self.fit.gradient(x) @ direction), 1e-5)
        self.assertLess(
            abs(self.L - np.linalg.svd(self.fit.A, compute_uv=False)[0] ** 2), 1e-8
        )

    def test_vectorized_objective_matches_the_scalar_one(self):
        rng = np.random.default_rng(1)
        points = rng.standard_normal((5, self.fit.n))
        expected = [self.fit.objective(x) for x in points]
        np.testing.assert_allclose(self.fit.objectives(points), expected, rtol=1e-12)

    def test_gradient_descent_converges_and_records_a_thinned_history(self):
        trace = ppm.gradient_descent(self.fit, self.L, DELTA, dense=10, spacing=1.5)
        self.assertTrue(trace.converged)
        self.assertLessEqual(
            np.linalg.norm(self.fit.gradient(trace.iterates[-1])), DELTA
        )
        self.assertLess(
            abs(self.fit.objective(trace.iterates[-1]) - self.truth.value), 1e-7
        )
        iteration = np.array(trace.iteration)
        self.assertTrue(np.all(np.diff(iteration) > 0))
        self.assertEqual(iteration[-1], trace.k_delta)
        self.assertEqual(trace.k_delta + 1, trace.gradient_evaluations)
        self.assertEqual(int(np.sum(trace.inner)), trace.k_delta)
        self.assertEqual(int(np.sum(trace.checks)), trace.k_delta + 1)
        self.assertTrue(np.array_equal(iteration[:10], np.arange(10)))
        self.assertLess(len(iteration), trace.k_delta + 1)
        self.assertAlmostEqual(
            trace.z_norm[-1],
            float(np.linalg.norm(self.fit.gradient(trace.iterates[-1]))),
        )

    def test_inexact_ppm_converges_and_satisfies_the_bounds(self):
        alpha, c0 = theory.inner_constants(self.gamma * self.L)
        _, r0 = ppm.accurate_resolvent(
            self.fit, self.gamma, self.L, np.zeros(self.fit.n)
        )
        for sigma, b in ((0.05, 1), (0.5, 4)):
            trace = ppm.inexact_ppm(self.fit, self.gamma, self.L, sigma, b, DELTA)
            self.assertTrue(trace.converged)
            self.assertLessEqual(
                np.linalg.norm(self.fit.gradient(trace.iterates[-1])), DELTA
            )
            self.assertLess(
                abs(self.fit.objective(trace.iterates[-1]) - self.truth.value), 1e-7
            )
            inner = np.array(trace.inner)
            for k in range(trace.k_delta + 1):
                _, residual = ppm.accurate_resolvent(
                    self.fit, self.gamma, self.L, trace.w[k]
                )
                beta = theory.beta(sigma, c0, trace.e0_norm[k], residual, DELTA)
                self.assertLessEqual(inner[k], theory.per_step_bound(beta, alpha, b))
            beta0 = theory.beta(sigma, c0, trace.e0_norm[0], r0, DELTA)
            total = theory.total_bound_deterministic(
                beta0, trace.k_delta, sigma, c0, alpha, b, r0, DELTA
            )
            self.assertLessEqual(inner.sum(), total)
            gamma_z = self.gamma * np.array(trace.z_norm[:-1])
            self.assertLessEqual(
                np.sum(gamma_z**2),
                (1 + sigma) / (1 - sigma) * np.linalg.norm(self.truth.x) ** 2,
            )


class TrajectoryTests(unittest.TestCase):
    def test_two_dimensional_paths_start_at_the_origin_and_end_at_the_optimum(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = trajectory_2d.build_parser().parse_args(
                ["--m", "12", "--grid", "21", "--results-dir", temporary]
            )
            directory = trajectory_2d.run(args)
            instance = aggregate.read_rows(directory / "instance.csv")[0]
            x_star = np.array([instance["x_star_1"], instance["x_star_2"]])
            runs = aggregate.read_rows(directory / "runs.csv")
            self.assertEqual(len(runs), 1 + len(study.SIGMAS) + 1 + 1)
            self.assertTrue(all(r["converged"] for r in runs))
            rows = aggregate.read_rows(directory / "trajectories.csv")
            for run in runs:
                key = aggregate.config_key(run)
                path = sorted(
                    (r for r in rows if aggregate.config_key(r) == key),
                    key=lambda r: r["iteration"],
                )
                self.assertEqual((path[0]["x1"], path[0]["x2"]), (0.0, 0.0))
                # The proximal runs record the start and then every accepted
                # iterate; gradient descent records the start as iteration 0.
                extra = 1 if run["method"] == "gd" else 2
                self.assertEqual(len(path), run["k_delta"] + extra)
                self.assertEqual(path[-1]["cumulative_inner"], run["total_inner"])
                last = np.array([path[-1]["x1"], path[-1]["x2"]])
                self.assertLess(np.linalg.norm(last - x_star), 1e-6)
                # The exact subproblem solutions: the near-exact iterates match
                # theirs, and an inexact iterate is within sigma ||x - w|| of its.
                if run["method"] != "gd":
                    self.assertTrue(all(np.isnan(r["exact1"]) for r in path[:1]))
                    for r in path[1:-1]:
                        gap = np.hypot(r["x1"] - r["w1"], r["x2"] - r["w2"])
                        if run["method"] == "near_exact":
                            allowed = 1e-9
                        else:
                            allowed = gap * run["sigma"]
                        self.assertLessEqual(r["inexactness"], allowed + 1e-9)
            contours = aggregate.read_rows(directory / "contours.csv")
            self.assertEqual(len(contours), 2 * 21 * 21)
            written = plots.trajectory_2d(directory)
            written += plots.trajectory_2d_exact(directory)
            self.assertEqual(len(written), 4)
            self.assertTrue(all(path.exists() for path in written))


class PipelineTests(unittest.TestCase):
    def test_study_driver_aggregation_and_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary)
            for name, extra in (
                ("admm_qp", dict(m=12, kappa=20.0)),
                ("ppm_huber", dict(m=45, threshold=0.5)),
            ):
                parser = study.build_parser(name)
                args = parser.parse_args(
                    [
                        "--instances",
                        "2",
                        "--n",
                        "30",
                        "--workers",
                        "1",
                        "--sigmas",
                        "0.05",
                        "0.5",
                        "--bs",
                        "1",
                        "4",
                        "--results-dir",
                        str(results_dir),
                    ]
                )
                for key, value in extra.items():
                    setattr(args, key, value)
                directory = study.run_study(name, args)
                results = aggregate.read_rows(directory / "results.csv")
                self.assertTrue(all(row["converged"] for row in results))
                self.assertTrue(all("b_exact_is_optimal" in row for row in results))
                self.assertTrue(any(row["b_exact_is_optimal"] for row in results))
                self.assertTrue((directory / "curves.csv").exists())
                summary = aggregate.read_rows(directory / "summary.csv")
                inexact = [r for r in summary if r["method"] == "inexact"]
                self.assertTrue(all(r["thm31_ratio_max"] <= 1.0 for r in inexact))
                self.assertTrue(all(r["thm43_ratio_max"] <= 1.0 for r in inexact))
                self.assertTrue(
                    all(r["per_step_violations_total"] == 0 for r in inexact)
                )
                baseline = [r for r in summary if r["method"] == "gd"]
                curves = aggregate.read_rows(directory / "curves.csv")
                if name == "ppm_huber":
                    self.assertEqual(len(baseline), 1)
                    self.assertEqual(baseline[0]["n_converged"], baseline[0]["n_runs"])
                    self.assertTrue(
                        any(
                            r["method"] == "gd" and r["axis"] == "inner" for r in curves
                        )
                    )
                else:
                    self.assertEqual(baseline, [])
                written = plots.make(name, results_dir)
                self.assertEqual(len(written), 2 * len(plots.FIGURES))
                self.assertTrue(all(path.exists() for path in written))


if __name__ == "__main__":
    unittest.main()
