"""Campaigns of original-format studies: one Slurm task per run, lanes apart."""

import json
import math

import pytest

from studies import campaign, study
from studies._io import read_rows


def specification(directory, results):
    tiny = ["--sigmas", "0.5", "1.0", "--bs", "1", "2", "--instances", "2"]
    return dict(
        campaign="test_campaign",
        results_dir=str(results),
        common=["--calibration-repetitions", "3", *tiny],
        lanes=dict(
            qp=dict(throttle=3, mem="1G", time="00:05:00"),
            regression=dict(
                throttle=2, mem="1G", time="00:05:00", flags=["--rows", "60"]
            ),
        ),
        studies=[
            dict(module="admm_qp", lane="qp", flags=["--side", "4"]),
            dict(module="admm_qp", lane="qp", flags=["--side", "4", "--inner", "agd"]),
            dict(module="ppm_logistic", lane="regression", flags=["--cols", "30"]),
        ],
    )


def prepared(tmp_path):
    directory = tmp_path / "campaign"
    directory.mkdir()
    spec = directory / "campaign.json"
    spec.write_text(json.dumps(specification(directory, tmp_path / "results")))
    campaign.prepare(spec)
    return directory, json.loads((directory / "seal.json").read_text())


def rows(manifest):
    return read_rows(campaign.ROOT / manifest)


def test_every_lane_runs_its_tasks_and_collects_what_finished(tmp_path):
    directory, seal = prepared(tmp_path)
    for lane in ("qp", "regression"):
        info = seal["lanes"][lane]
        for row in rows(info["calibrate"]):
            campaign.task(campaign.ROOT / info["calibrate"], int(row["task"]))
        for part in info["runs"]:
            for row in rows(part["manifest"]):
                campaign.task(campaign.ROOT / part["manifest"], int(row["task"]))
        for row in rows(info["collect"]):
            campaign.task(campaign.ROOT / info["collect"], int(row["task"]))
    progress = campaign.status(directory)
    assert {r["study"] for r in progress} == {
        "admm_qp_side4_gmres",
        "admm_qp_side4_agd",
        "ppm_logistic_random_normal_gd",
    }
    for lane in ("qp", "regression"):
        # One task per study and configuration, every instance within it.
        tasks = sum(part["tasks"] for part in seal["lanes"][lane]["runs"])
        assert tasks == sum(s["reserved"] for s in seal["studies"] if s["lane"] == lane)
    for entry in progress:
        assert entry["calibrated"] and entry["finished"] == entry["planned"] > 0
        assert entry["instances_complete"] == entry["instances"] == 2
        results = read_rows(tmp_path / "results" / entry["study"] / "results.csv")
        assert len(results) == entry["planned"]
        assert not (tmp_path / "results" / entry["study"] / "missing.csv").exists()
    # The unconstrained solver reserves no room for recommended runs.
    reserved = {s["name"]: s["reserved"] for s in seal["studies"]}
    assert reserved["admm_qp_side4_gmres"] == 2 + 2 + 2 + 3
    assert reserved["admm_qp_side4_agd"] == 2 + 2 + 2


def test_runs_are_configuration_major_and_lanes_have_their_own_manifests(tmp_path):
    _, seal = prepared(tmp_path)
    qp = rows(seal["lanes"]["qp"]["runs"][0]["manifest"])
    assert [int(r["task"]) for r in qp] == list(range(len(qp)))
    first = [r for r in qp if r["configuration"] == 0]
    assert qp[: len(first)] == first
    assert {r["study"] for r in first} == {"admm_qp_side4_gmres", "admm_qp_side4_agd"}
    assert all(r["lane"] == "qp" for r in qp)
    regression = rows(seal["lanes"]["regression"]["runs"][0]["manifest"])
    assert {r["study"] for r in regression} == {"ppm_logistic_random_normal_gd"}


def test_a_long_lane_is_split_under_the_array_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "MAX_ARRAY", 9)
    _, seal = prepared(tmp_path)
    info = seal["lanes"]["qp"]
    sizes = [part["tasks"] for part in info["runs"]]
    assert len(sizes) == math.ceil(info["tasks"] / 10) > 1 and max(sizes) <= 10
    assert sum(sizes) == info["tasks"]
    # The parts share the lane's cap of three side by side.
    assert info["run_throttle"] == 3 // len(sizes) and not info["run_parts_chained"]


def test_the_seal_freezes_the_inputs_and_the_campaign(tmp_path):
    directory, seal = prepared(tmp_path)
    with pytest.raises(SystemExit, match="new campaign name"):
        campaign.prepare(directory / "campaign.json")
    args = campaign.ROOT / seal["studies"][0]["args"]
    args.write_text(args.read_text() + "--delta 1e-8\n")
    manifest = campaign.ROOT / seal["lanes"]["qp"]["calibrate"]
    with pytest.raises(SystemExit, match="changed since"):
        campaign.task(manifest, 0)


@pytest.mark.parametrize("name", ["studies_v1", "studies_v2"])
def test_the_prepared_campaigns_are_sealed_and_consistent(name):
    seal_path = campaign.ROOT / name / "seal.json"
    if not seal_path.exists():
        pytest.skip(f"{name} is not prepared")
    seal = json.loads(seal_path.read_text())
    assert campaign.verify(campaign.ROOT / seal["lanes"]["qp"]["calibrate"])
    for info in seal["lanes"].values():
        total = sum(part["tasks"] for part in info["runs"])
        assert total == info["tasks"]
        assert all(part["tasks"] <= campaign.MAX_ARRAY + 1 for part in info["runs"])
        assert (
            info["run_throttle"] * len(info["runs"]) <= info["throttle"]
            or (info["run_parts_chained"])
        )
        for row in rows(info["calibrate"]):
            parser = study.build_parser(row["module"])
            args = parser.parse_args([f"@{campaign.ROOT / row['args']}"])
            assert args.name == row["study"]


def test_studies_v2_is_studies_v1_with_a_longer_time_limit():
    """The rerun changes the wall-clock stop of the inexact runs and nothing else."""
    identities: dict[str, dict] = {}
    for name in ("studies_v1", "studies_v2"):
        seal_path = campaign.ROOT / name / "seal.json"
        if not seal_path.exists():
            pytest.skip(f"{name} is not prepared")
        for entry in json.loads(seal_path.read_text())["studies"]:
            parser = study.build_parser(entry["module"])
            args = parser.parse_args([f"@{campaign.ROOT / entry['args']}"])
            settings = study.settings_from(args, entry["module"])
            identities.setdefault(entry["name"], {})[name] = study._identity(settings)
    rerun = sorted(n for n, pair in identities.items() if "studies_v2" in pair)
    assert rerun == sorted(n for n in identities if "kaczmarz" not in n)
    for name in rerun:
        before, after = identities[name]["studies_v1"], identities[name]["studies_v2"]
        keys = before.keys() | after.keys()
        changed = {k for k in keys if before.get(k) != after.get(k)}
        assert changed == {"time_limit"}, name
        assert (before["time_limit"], after["time_limit"]) == (300.0, 1200.0)
