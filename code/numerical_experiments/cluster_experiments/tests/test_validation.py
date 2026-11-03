import hashlib
import json

import pytest

from validation.audit import audit
from validation.pilots import configurations
from validation.releases import freeze, verify


def fake_project(root):
    files = {
        "releases/test/experiments/__init__.py": "",
        "releases/test/README.md": "Test release\n",
        "pyproject.toml": "",
        "uv.lock": "",
        "slurm/common.sh": "",
        "slurm/job_array.slurm": "",
        "slurm/cluster.env": "",
        "slurm/validation_v6.slurm": "",
    }
    for name, contents in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents)
    p = root / "releases/test/source_manifest.json"
    p.write_text(
        json.dumps(
            dict(
                source_sha256={
                    "experiments/__init__.py": hashlib.sha256(b"").hexdigest(),
                }
            )
        )
    )


def test_freeze_is_idempotent_and_detects_source_and_archive_changes(tmp_path):
    fake_project(tmp_path)
    first = freeze(tmp_path, "test", "First captured version")
    assert freeze(tmp_path, "test", "Repeated request") == first
    assert (
        len((tmp_path / "release_history/history.jsonl").read_text().splitlines()) == 1
    )
    source = tmp_path / "releases/test/experiments/__init__.py"
    source.write_text("changed")
    with pytest.raises(ValueError, match="sealed source changed"):
        verify(tmp_path, "test")
    source.write_text("")
    archive = tmp_path / "release_history/snapshots" / f"{first['archive_sha256']}.tar"
    archive.chmod(0o644)
    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        verify(tmp_path, "test")


def test_refreshing_a_source_manifest_cannot_reseal_a_changed_release(tmp_path):
    fake_project(tmp_path)
    freeze(tmp_path, "test", "First capture")
    (tmp_path / "releases/test/experiments/added.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="sealed source changed"):
        freeze(tmp_path, "test", "Attempted rewrite")


def test_pilots_cover_families_orientations_and_bounded_small_blocks():
    configs = list(configurations())
    assert len(configs) == 12
    for family in ["lasso", "logistic"]:
        assert {c["orientation"] for c in configs if c["family"] == family} == {
            "native",
            "transpose",
        }
    qp = [c for c in configs if c["family"] == "qp"]
    assert all(c["side"] ** 3 > 50000 and c["kappa"] >= 1e4 for c in qp)
    kacz = next(c for c in qp if c["inner"] == "block_kaczmarz")
    assert kacz["block_size"] == 1 and kacz["max_inner"] == 1
    assert all(
        c["setup_policy"] == "independent" and c["time_limit"] <= 60 for c in configs
    )


def test_missing_pilot_cannot_pass_audit(tmp_path):
    from releases.accounting_v6.experiments.io import write_csv

    c = next(configurations(smoke=True))
    manifest = tmp_path / "pilot.csv"
    write_csv(manifest, [dict(task_id=0, run_id="missing", config=json.dumps(c))])
    checks = audit(manifest, tmp_path)
    assert not checks[0]["passed"] and "missing result" in checks[0]["failures"]


def test_reporting_snapshots_preserve_prior_source_instead_of_refreshing_hashes(
    tmp_path,
):
    from release_tools.reporting import capture

    fake_project(tmp_path)
    for name, text in {
        "plots/__init__.py": "",
        "plots/example.py": "version = 1\n",
        "EXPERIMENT_VALIDATION.md": "Test conventions\n",
        "tests/test_publication.py": "",
        "experiments/io.py": "",
    }.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    first = capture(tmp_path, "reporting_v1", "First capture")
    assert capture(tmp_path, "reporting_v1", "Repeat") == first
    (tmp_path / "plots/example.py").write_text("version = 2\n")
    with pytest.raises(ValueError, match="use a new label"):
        capture(tmp_path, "reporting_v1", "Attempt to overwrite")
    second = capture(tmp_path, "reporting_v2", "Second capture")
    assert first["archive_sha256"] != second["archive_sha256"]
    assert second["previous_entry_sha256"] == first["history_entry_sha256"]
