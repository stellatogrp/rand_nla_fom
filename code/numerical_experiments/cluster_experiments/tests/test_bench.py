"""The one-node benchmark: plan, seal, interleaved runs, resumption, summary."""

import json
import math

import pytest

from bench import one_node
from studies import campaign
from studies._io import read_rows


def tiny_campaign(tmp_path):
    """Three solvers of a 64-variable QP, three instances, run to the end."""
    directory = tmp_path / "campaign"
    directory.mkdir()
    small = ["--sigmas", "0.5", "1.0", "--bs", "1", "2", "--instances", "3"]
    spec = dict(
        campaign="tiny",
        results_dir=str(tmp_path / "results"),
        common=["--calibration-repetitions", "3", *small],
        lanes=dict(qp=dict(throttle=2, mem="1G", time="00:05:00")),
        studies=[
            dict(module="admm_qp", lane="qp", flags=["--side", "4"]),
            dict(module="admm_qp", lane="qp", flags=["--side", "4", "--inner", "cg"]),
            dict(module="admm_qp", lane="qp", flags=["--side", "4", "--inner", "agd"]),
        ],
    )
    (directory / "campaign.json").write_text(json.dumps(spec))
    campaign.prepare(directory / "campaign.json")
    info = json.loads((directory / "seal.json").read_text())["lanes"]["qp"]
    manifests = [info["calibrate"], *(p["manifest"] for p in info["runs"])]
    for manifest in [*manifests, info["collect"]]:
        for row in read_rows(campaign.ROOT / manifest):
            campaign.task(campaign.ROOT / manifest, int(row["task"]))
    return directory


@pytest.fixture
def benchmark(tmp_path):
    directory = tiny_campaign(tmp_path)
    bench = tmp_path / "bench"
    bench.mkdir()
    plan = one_node.make_plan(directory, "tiny_bench", {"qp_side4": 1}, 2)
    (bench / "plan.json").write_text(json.dumps(plan, indent=1))
    one_node.seal(bench)
    return bench, plan, tmp_path / "bench_results"


def test_the_plan_times_every_inexact_configuration(benchmark, tmp_path):
    _, plan, _ = benchmark
    (problem,) = plan["problems"]
    assert problem["slug"] == "qp_side4" and problem["seeds"] == [0, 1, 2]
    assert problem["near_exact_seeds"] == [0]
    assert problem["exact_study"] == "admm_qp_side4_cg"
    for study in problem["studies"]:
        listed = read_rows(tmp_path / "results" / study["name"] / "configurations.csv")
        inexact = [row for row in listed if row["method"] == "inexact"]
        assert len(study["configurations"]) == len(inexact)
        marks = [c["kinds"] for c in study["configurations"]]
        assert sum("fixed" in m for m in marks) == 1
        assert sum("campaign_best" in m for m in marks) == 1
        # sigma* exists only for the solvers with contraction constants.
        assert sum("sigma_star" in m for m in marks) == (study["solver"] != "agd")
        assert study["near_exact"]["method"] == "near_exact"
        assert study["exact"]["method"] == "exact"


def test_the_plan_can_take_the_first_instances(tmp_path):
    directory = tiny_campaign(tmp_path)
    plan = one_node.make_plan(directory, "tiny_bench", {}, 5, instances=2)
    (problem,) = plan["problems"]
    assert problem["seeds"] == [0, 1] and problem["near_exact_seeds"] == [0, 1]


def test_every_planned_run_is_timed_once_and_repeats_the_campaign(benchmark):
    bench, plan, results = benchmark
    (problem,) = plan["problems"]
    expected = sum(len(one_node.tasks_for(problem, s, {0})) for s in problem["seeds"])
    assert one_node.run(bench / "plan.json", results, ["qp_side4"]) == expected
    # A resubmission finds every run done.
    assert one_node.run(bench / "plan.json", results, ["qp_side4"]) == 0
    rows = read_rows(results / "qp_side4" / "measurements.csv")
    assert len(rows) == expected
    for seed in problem["seeds"]:
        positions = sorted(int(r["position"]) for r in rows if r["seed"] == seed)
        assert positions == list(range(len(positions)))
    first = [r for r in rows if r["seed"] == 0]
    assert {r["method"] for r in first} == {"inexact", "near_exact", "exact"}
    assert all(r["own_setup_seconds"] >= 0 for r in rows)
    table, report = one_node.summarize(bench / "plan.json", results)
    assert f"equal to the campaign's in {expected}, different in 0" in report
    assert "unmatched 0" in report
    fastest = [r for r in table if r["fastest_here"]]
    assert {r["solver"] for r in fastest} == {"gmres", "cg", "agd"}
    assert min(r["ratio_to_fastest_median"] for r in fastest) >= 1.0
    at_fixed = [r for r in table if not math.isnan(r["ratio_at_fixed_median"])]
    assert {r["solver"] for r in at_fixed} == {"gmres", "cg", "agd", "exact ADMM"}
    assert "every solver at sigma 0.1, s 1" in report


def test_the_order_is_seeded_and_interleaves_the_solvers(benchmark):
    _, plan, _ = benchmark
    (problem,) = plan["problems"]
    tasks = one_node.tasks_for(problem, 1, set())
    first = one_node.order(plan, problem, 0, 1, len(tasks))
    assert first == one_node.order(plan, problem, 0, 1, len(tasks))
    assert first != one_node.order(plan, problem, 0, 2, len(tasks))
    assert sorted(first) == list(range(len(tasks)))


def test_the_seal_freezes_the_plan(benchmark):
    bench, plan, results = benchmark
    with pytest.raises(SystemExit, match="new benchmark name"):
        one_node.seal(bench)
    plan["problems"][0]["seeds"] = [0]
    (bench / "plan.json").write_text(json.dumps(plan, indent=1))
    with pytest.raises(SystemExit, match="changed since"):
        one_node.run(bench / "plan.json", results, ["qp_side4"])


def test_problem_slugs():
    assert one_node.slug("qp side 100") == "qp_side100"
    assert one_node.slug("lasso random_normal") == "lasso_random_normal"
