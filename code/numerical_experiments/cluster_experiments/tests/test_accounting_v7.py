import numpy as np
import pytest

from releases.accounting_v7.experiments import solvers
from releases.accounting_v7.experiments.problems import make_qp


@pytest.mark.parametrize("outer_cap", [1, 2])
def test_small_block_work_budget_cannot_admit_one_extra_update(monkeypatch, outer_cap):
    # Keep the actual 64,000-variable problem/checks, but make row updates cheap
    # and stationary so the test reaches the budget rather than convergence.
    p = make_qp(0, 40)

    class StationaryInner:
        def __init__(self, problem, c, method, rng, *args, **kwargs):
            self.block_size = 1
            self.products = self.row_updates = self.block_factorizations = 0
            self.setup_seconds = 0.0

        def start(self, x, rhs):
            self.x = x.copy()

        def step(self, q_x=None):
            self.row_updates += 1

    monkeypatch.setattr(solvers, "Inner", StationaryInner)
    row, _ = solvers.solve(
        p,
        dict(
            inner="block_kaczmarz",
            seed=0,
            ratio=10.0,
            sigma=0.0,
            stride=64000,
            block_size=1,
            budget_unit="work",
            max_inner=outer_cap,
            max_inner_per_outer=1,
            max_outer=10,
            time_limit=5,
            inner_time_limit=5,
        ),
    )
    assert row["total_inner"] == 64000
    assert row["row_updates"] == 64000
    assert row["work_units"] == 1.0
    assert row["status"] == ("max_inner" if outer_cap == 1 else "inner_iteration_limit")
    assert np.isfinite(row["stationarity"])
