import json
import subprocess

import pytest

from experiments import run, solvers
from experiments.io import identity, read_csv, write_csv
from experiments.limits import budgets
from experiments.problems import make_qp
from tests.test_experiments import config


@pytest.mark.parametrize("name", list(budgets({})))
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_budgets_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        budgets({name: value})


def test_small_block_kaczmarz_stops_inside_stride():
    result, _ = solvers.solve(
        make_qp(side=4),
        config(inner="block_kaczmarz", block_size=1, stride=20, max_inner_per_outer=3),
    )
    assert result["status"] == "inner_iteration_limit"
    assert result["total_inner"] == 3
    assert not result["converged"]


def test_inner_wall_budget(monkeypatch):
    ticks = iter(range(10000))
    monkeypatch.setattr(solvers, "perf_counter", lambda: next(ticks))
    result, _ = solvers.solve(
        make_qp(side=4), config(inner_time_limit=2, time_limit=1000)
    )
    assert result["status"] == "inner_time_limit"
    assert not result["converged"]


def test_parent_kills_unresponsive_worker_and_records_timeout(tmp_path, monkeypatch):
    cfg = config(phase="smoke")
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, [dict(config=json.dumps(cfg), run_id=identity(cfg))])
    events = []

    class HungWorker:
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            events.append("reaped")

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(run.subprocess, "Popen", lambda command: HungWorker())
    monkeypatch.setattr(run.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        run.sys,
        "argv",
        [
            "run",
            "--manifest",
            str(manifest),
            "--task-id",
            "0",
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )
    run.main()
    assert events == ["terminate", "kill", "reaped"]
    row = read_csv(tmp_path / "results" / identity(cfg) / "result.csv")[0]
    assert row["status"] == "worker_timeout"
    assert row["converged"] == "False"
