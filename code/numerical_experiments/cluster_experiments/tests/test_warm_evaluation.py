"""Check arithmetic averaging and equal seed weighting for warm-only figures."""

import numpy as np
import pytest

from plots.warm_evaluation import mean_curve, theory_means, timing_components


def test_timing_phases_preserve_setup_and_sum_to_worker_wall_time():
    row = dict(
        clean_attempt_seconds=20,
        solve_seconds=8,
        solver_setup_seconds=5,
        generation_seconds=2,
        tuning_seconds=3,
    )
    components = timing_components(row)
    assert components.tolist() == [8, 5, 2, 3, 2, 20]
    assert sum(components[:5]) == components[5]
    with pytest.raises(ValueError, match="overlapping"):
        timing_components(dict(row, generation_seconds=10))


def test_curve_averages_residuals_arithmetically_without_extrapolation():
    series = [
        (np.array([0, 10]), np.array([1.0, 1.0])),
        (np.array([2, 20]), np.array([10.0, 10.0])),
        (np.array([1, 15]), np.array([100.0, 100.0])),
    ]
    grid, mean = mean_curve(series)
    assert min(grid) >= 2 and max(grid) <= 10
    assert np.allclose(mean, 37.0)


def test_theory_means_weight_seeds_equally_and_prefer_true_audits():
    rows = []
    for run_id, values in [("seed3", [1, 3]), ("seed4", [9]), ("seed5", [10])]:
        for outer, value in enumerate(values):
            rows.append(
                dict(
                    method="cg",
                    n=64000,
                    local_inner=1,
                    run_id=run_id,
                    outer=outer,
                    measured_squared_ratio=value,
                    outer_boundary=False,
                )
            )
    # Replace one recurrence value, retaining exactly one observation per solve.
    rows.append(dict(rows[0], measured_squared_ratio=5, outer_boundary=True))
    result = theory_means(rows)
    assert len(result) == 1
    assert result[0]["mean_squared_ratio"] == pytest.approx((4 + 9 + 10) / 3)
    assert theory_means([r for r in rows if r["run_id"] != "seed5"]) == []
