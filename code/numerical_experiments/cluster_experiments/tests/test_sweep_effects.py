"""Check sweep comparability, replicate accounting, and censor handling."""

import json
import math

import pytest

from experiments.io import identity, write_csv
from plots.sweep_effects import load, summarize


def records():
    return [
        dict(
            sweep="stride",
            n=64000,
            method="gd",
            initialization="warm",
            requested_value="optimal",
            choice_kind="model",
            seed=seed,
            resolved_value=stride,
            converged=True,
            status="converged",
            worker_wall_seconds=seconds,
            inner_updates=100 + seed,
            outer_solves=20 + seed,
        )
        for seed, stride, seconds in [(0, 1, 1), (1, 3, 9), (2, 2, 3)]
    ]


def test_sweep_points_use_arithmetic_means_and_resolved_model_strides():
    row = summarize(records())[0]
    assert row["worker_wall_seconds_mean"] == pytest.approx(13 / 3)
    assert row["resolved_value_mean"] == 2
    assert not any(k.endswith(("_median", "_min", "_max")) for k in row)
    assert row["converged_runs"] == row["runs"] == 3


def test_censored_runs_are_not_treated_as_converged_timings():
    rows = records()
    rows[1].update(converged=False, status="time_limit")
    row = summarize(rows)[0]
    assert row["runs"] == 3 and row["converged_runs"] == 2
    assert math.isnan(row["worker_wall_seconds_mean"])
    assert "time_limit" in row["statuses"]


def test_duplicate_or_missing_seeds_cannot_produce_sweep_means():
    with pytest.raises(ValueError, match="duplicate seed"):
        summarize([*records(), records()[0]])
    with pytest.raises(ValueError, match="seeds 0, 1, and 2"):
        summarize(records()[:2])


def test_sweep_rejects_changes_to_an_additional_solver_parameter(tmp_path):
    tasks = []
    for sigma, factor in [(0.2, 1), (0.5, 0.5)]:
        config = dict(
            family="qp",
            phase="tune",
            sweep="sigma",
            inner="gd",
            inner_initialization="warm",
            sigma=sigma,
            stride=1,
            ratio=10,
            seed=0,
            gd_factor=factor,
        )
        key = identity(config)
        task = dict(run_id=key, config=json.dumps(config))
        tasks.append(task)
        write_csv(
            tmp_path / key / "result.csv",
            [
                dict(
                    task,
                    n=64000,
                    converged=True,
                    status="converged",
                    sigma=sigma,
                    stride=1,
                    c=0.1,
                    total_inner=100,
                    outer=10,
                    clean_attempt_seconds=2,
                    setup_policy="independent",
                )
            ],
        )
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, tasks)
    with pytest.raises(ValueError, match="additional configuration"):
        load(tmp_path, manifest)
