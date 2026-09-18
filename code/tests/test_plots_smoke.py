"""Every figure module draws from CSVs alone.

The tables below are synthetic fixtures with the right columns, not measured
results: their job is to pin the contract between what the experiment scripts
write and what the plotting code reads, without running a sweep.
"""

import tempfile
import unittest
from pathlib import Path

from experiments._paths import HISTORIES, RESULTS, study_dir
from experiments._results import write_rows
from plots import (
    gamma_sweep,
    inequality_qp_tuning,
    lp_tuning,
    sigma_sweep,
    solver_sweep,
    standard,
    theta_sweep,
)
from plots._data import MissingResults
from plots.make_plots import FIGURES
from plots.make_plots import main as make_plots_main

CONDITIONS = ("moderately-conditioned", "ill-conditioned")
SIZE = dict(n=200, m=50, solver_n=200)


def history(keys: dict, iterations: int = 4) -> list[dict]:
    return [
        dict(
            **keys,
            iteration=index + 1,
            inner_iterations=3,
            cumulative_inner_iterations=3 * (index + 1),
            objective_error=10.0 ** -(index + 1),
            feasibility=10.0 ** -(index + 2),
            accuracy=10.0 ** -(index + 1),
            theta_k=1.0,
            sigma_k=0.2,
        )
        for index in range(iterations)
    ]


class PlotSmokeTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.results = Path(self._directory.name)

    def write(self, study: str, name: str, rows: list[dict]) -> None:
        write_rows(study_dir(study, self.results) / name, rows)

    def assert_pdfs(self, written):
        self.assertTrue(written)
        for path in written:
            self.assertTrue(path.exists(), path)
            self.assertEqual(path.suffix, ".pdf")

    def test_standard(self):
        rows = []
        for problem in ("QP", "LP"):
            for group in ("M", "sigma", "theta"):
                for label in ("a", "b"):
                    rows += history(dict(problem=problem, group=group, label=label))
            for solver in ("schur_cg", "coupled_gmres"):
                rows += history(dict(problem=problem, group="solver", label=solver))
        self.write("standard", HISTORIES, rows)
        self.assert_pdfs(standard.make(self.results))

    def test_lp_tuning(self):
        rows = []
        for problem, stage in (
            ("proxy", "preconditioner"),
            ("proxy", "sigma_theta"),
            ("large", "transfer"),
        ):
            for label in ("a", "b"):
                rows += history(dict(problem=problem, stage=stage, label=label))
        self.write("lp_tuning", HISTORIES, rows)
        self.assert_pdfs(lp_tuning.make(self.results))

    def test_inequality_qp_tuning(self):
        rows = []
        for problem, stage in (
            ("proxy", "preconditioner"),
            ("proxy", "sigma_theta"),
            ("large", "transfer"),
        ):
            for label in ("a", "b"):
                rows += history(
                    dict(
                        problem=problem,
                        stage=stage,
                        label=label,
                        adaptive=stage == "transfer" and label == "b",
                    )
                )
        self.write("inequality_qp_tuning", HISTORIES, rows)
        written = inequality_qp_tuning.make(self.results)
        self.assert_pdfs(written)
        self.assertIn("adaptive_parameters.pdf", {path.name for path in written})

    def test_gamma_sweep(self):
        rows = [
            dict(
                problem=kind,
                matrix=condition,
                gamma=gamma,
                sigma=0.2,
                theta=1.0,
                inner_solver="schur_cg",
                matrix_condition=10.0,
                converged=gamma < 10.0,
                outer_iterations=100.0 / gamma,
                inner_iterations=400.0 / gamma,
                mean_inner_iterations=4.0,
                objective_error=1e-8,
                feasibility=1e-9,
                error="",
                **SIZE,
            )
            for kind in ("equality QP", "LP", "inequality QP")
            for condition in CONDITIONS
            for gamma in (0.1, 1.0, 10.0)
        ]
        self.write("gamma_sweep", RESULTS, rows)
        self.write(
            "gamma_sweep",
            "inequality_qp_results.csv",
            [row for row in rows if row["problem"] == "inequality QP"],
        )
        self.write(
            "gamma_sweep",
            HISTORIES,
            [
                row
                for condition in CONDITIONS
                for row in history(
                    dict(
                        problem="inequality QP",
                        matrix=condition,
                        gamma=1.0,
                        label=condition,
                    )
                )
            ],
        )
        self.assert_pdfs(gamma_sweep.make(self.results))

    def test_sigma_sweep(self):
        rows = [
            dict(
                problem=kind,
                matrix=condition,
                sigma=sigma,
                theta=1.0,
                gamma_x=1.0,
                gamma_lambda=1.0,
                target_accuracy=1e-7,
                matrix_condition=10.0,
                inner_solver="schur_block_kaczmarz",
                block_size=25,
                block_fraction=0.05,
                random_seed=0,
                reached_accuracy=True,
                outer_iterations=50.0,
                inner_iterations=2_000.0,
                achieved_accuracy=1e-8,
                solver_iterations=60.0,
                runtime_seconds=1.5,
                error="",
                **SIZE,
            )
            for kind in ("equality QP", "LP", "inequality QP")
            for condition in CONDITIONS
            for sigma in (0.0, 0.2, 0.4)
        ]
        self.write("sigma_sweep", RESULTS, rows)
        self.assert_pdfs(sigma_sweep.make(self.results))

    def test_theta_sweep(self):
        rows = [
            dict(
                problem=kind,
                matrix=condition,
                policy=policy,
                trial=trial,
                theta=theta,
                sigma=0.04,
                gamma_x=1.0,
                gamma_lambda=1.0,
                target_accuracy=1e-7,
                matrix_condition=10.0,
                inner_solver="schur_randomized_kaczmarz",
                random_seed=trial,
                max_outer_iterations=6_000,
                reached_accuracy=True,
                outer_iterations=100.0 + trial,
                inner_iterations=5_000.0 + trial,
                achieved_accuracy=1e-8,
                solver_iterations=120.0,
                runtime_seconds=2.0 + trial,
                error="",
                **SIZE,
            )
            for kind in ("equality QP", "LP", "inequality QP")
            for condition in ("moderately-conditioned", "mildly-ill-conditioned")
            for policy in ("fixed", "frontier")
            for trial in (0, 1)
            for theta in (0.5, 1.0, 1.5)
        ]
        self.write("theta_sweep", RESULTS, rows)
        self.assert_pdfs(theta_sweep.make(self.results))

    def _solver_rows(self, randomize_spectrum: bool) -> list[dict]:
        return [
            dict(
                problem=kind,
                matrix=condition,
                inner_solver=solver,
                instance_seed=1_000 + instance,
                sigma=0.2,
                theta=1.0,
                gamma_x=1.0,
                gamma_lambda=1.0,
                target_accuracy=1e-7,
                matrix_condition=10.0,
                block_size=5,
                block_fraction=0.05,
                random_seed=0,
                reached_accuracy=True,
                outer_iterations=80.0 + instance,
                inner_iterations=3_000.0 + instance,
                achieved_accuracy=1e-8,
                solver_iterations=90.0,
                runtime_seconds=1.0 + instance,
                error="",
                randomize_spectrum=randomize_spectrum,
                **SIZE,
            )
            for kind in ("equality QP", "LP", "inequality QP")
            for condition in CONDITIONS
            for solver in (
                "schur_cg",
                "schur_block_kaczmarz",
                "schur_randomized_kaczmarz",
                "coupled_gmres",
            )
            for instance in range(3)
        ]

    def test_solver_sweep(self):
        self.write("solver_sweep", RESULTS, self._solver_rows(False))
        self.assert_pdfs(solver_sweep.make(self.results))

    def test_solver_sweep_randomized(self):
        self.write(
            "solver_sweep_randomized_spectrum",
            RESULTS,
            self._solver_rows(True),
        )
        self.assert_pdfs(solver_sweep.make_randomized(self.results))

    def test_missing_results_are_reported_not_invented(self):
        for name in FIGURES:
            with self.subTest(figure=name):
                with self.assertRaises(MissingResults):
                    from plots.make_plots import draw

                    draw(name, self.results)

    def test_make_plots_skips_missing_studies(self):
        # An empty results tree is the normal state before anything has run.
        make_plots_main(["--results-dir", str(self.results)])


if __name__ == "__main__":
    unittest.main()
