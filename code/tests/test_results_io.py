"""The CSV contract between the experiment scripts and the plotting code."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments._metrics import accuracy_history, history_rows, target_index
from experiments._paths import HISTORIES, RESULTS, study_dir
from experiments._results import merge_rows, read_rows, select, write_rows
from rand_nla_fom import primal_dual_drs, quadratic_prox
from tests._fixtures import equality_qp


class ResultTableTests(unittest.TestCase):
    def test_types_survive_a_round_trip(self):
        rows = [
            dict(
                problem="LP",
                converged=True,
                outer_iterations=42,
                sigma=0.2,
                error="",
            ),
            dict(
                problem="LP",
                converged=False,
                outer_iterations=float("nan"),
                sigma=0.45,
                error="target accuracy not reached",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / RESULTS
            write_rows(path, rows)
            recovered = read_rows(path)

        self.assertEqual(recovered[0]["problem"], "LP")
        self.assertIs(recovered[0]["converged"], True)
        self.assertIs(recovered[1]["converged"], False)
        self.assertEqual(recovered[0]["outer_iterations"], 42)
        self.assertTrue(np.isnan(recovered[1]["outer_iterations"]))
        self.assertEqual(recovered[0]["sigma"], 0.2)
        # An empty field is a missing measurement, not the string "".
        self.assertTrue(np.isnan(recovered[0]["error"]))
        self.assertEqual(recovered[1]["error"], "target accuracy not reached")

    def test_reading_a_missing_table_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(read_rows(Path(directory) / RESULTS), [])

    def test_write_rejects_an_empty_table(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_rows(Path(directory) / RESULTS, [])

    def test_merge_replaces_matching_keys_and_reorders(self):
        rows = [dict(sigma=0.1, work=5), dict(sigma=0.3, work=9)]
        merged = merge_rows(
            rows,
            [dict(sigma=0.3, work=2), dict(sigma=0.2, work=7)],
            key=lambda row: row["sigma"],
            order=lambda row: row["sigma"],
        )
        self.assertEqual([row["sigma"] for row in merged], [0.1, 0.2, 0.3])
        self.assertEqual([row["work"] for row in merged], [5, 7, 2])

    def test_select_matches_every_field(self):
        rows = [
            dict(problem="LP", matrix="ill-conditioned"),
            dict(problem="LP", matrix="moderately-conditioned"),
            dict(problem="QP", matrix="ill-conditioned"),
        ]
        self.assertEqual(select(rows, problem="LP", matrix="ill-conditioned"), rows[:1])


class HistoryTests(unittest.TestCase):
    def setUp(self):
        matrix, rhs, hessian, linear_term, solution = equality_qp()
        self.optimum = float(
            0.5 * solution @ hessian @ solution + linear_term @ solution
        )
        self.result = primal_dual_drs(
            quadratic_prox(hessian, linear_term),
            matrix,
            rhs,
            objective=lambda x: float(0.5 * x @ hessian @ x + linear_term @ x),
            tolerance=1e-9,
            max_iterations=500,
        )

    def test_history_has_one_row_per_outer_iteration(self):
        rows = history_rows(
            dict(problem="QP", label="baseline"), self.result, self.optimum
        )
        self.assertEqual(len(rows), self.result.iterations)
        self.assertEqual([row["iteration"] for row in rows][:3], [1, 2, 3])
        self.assertEqual(rows[0]["problem"], "QP")
        self.assertEqual(
            rows[-1]["cumulative_inner_iterations"],
            int(self.result.inner_iterations.sum()),
        )

    def test_history_accuracy_matches_the_target_test(self):
        rows = history_rows(dict(label="baseline"), self.result, self.optimum)
        accuracy = accuracy_history(self.optimum, self.result)
        index = target_index(accuracy, 1e-6)
        self.assertIsNotNone(index)
        # The trace on disk is what the sweeps threshold in memory, up to the
        # floor that keeps a log axis drawable.
        self.assertAlmostEqual(
            rows[index]["accuracy"], float(accuracy[index]), places=12
        )

    def test_history_round_trips_through_csv(self):
        rows = history_rows(
            dict(problem="QP", label="baseline"), self.result, self.optimum
        )
        with tempfile.TemporaryDirectory() as directory:
            path = study_dir("smoke", directory) / HISTORIES
            write_rows(path, rows)
            recovered = read_rows(path)
        self.assertEqual(len(recovered), len(rows))
        self.assertEqual(recovered[0]["iteration"], 1)
        self.assertAlmostEqual(
            recovered[-1]["accuracy"], rows[-1]["accuracy"], places=12
        )


if __name__ == "__main__":
    unittest.main()
