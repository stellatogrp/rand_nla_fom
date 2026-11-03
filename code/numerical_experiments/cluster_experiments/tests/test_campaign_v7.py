import json

import pytest

from campaign_v7.advance import complete
from campaign_v7.prepare import emit
from releases.accounting_v7.experiments.io import identity, write_csv


def test_followup_cannot_advance_missing_or_unresolved_results(tmp_path):
    config = dict(family="qp", inner="cg", seed=0)
    key = identity(config)
    task = dict(task_id=0, run_id=key, config=json.dumps(config))
    manifest = tmp_path / "input.csv"
    write_csv(manifest, [task])
    with pytest.raises(ValueError, match="incomplete"):
        complete(manifest, tmp_path)
    write_csv(
        tmp_path / key / "result.csv",
        [
            dict(
                task,
                status="unresolved_error",
                error="ValueError: invalid input",
                converged=False,
            )
        ],
    )
    with pytest.raises(ValueError, match="investigate"):
        complete(manifest, tmp_path)


def test_followup_rejects_shared_loading_and_accepts_honest_censoring(tmp_path):
    config = dict(family="qp", inner="cg", seed=0)
    key = identity(config)
    task = dict(task_id=0, run_id=key, config=json.dumps(config))
    manifest = tmp_path / "input.csv"
    write_csv(manifest, [task])
    result = dict(
        task, status="max_inner", converged=False, setup_policy="shared_pilot"
    )
    write_csv(tmp_path / key / "result.csv", [result])
    with pytest.raises(ValueError, match="shared-loading"):
        complete(manifest, tmp_path)
    result["setup_policy"] = "independent"
    write_csv(tmp_path / key / "result.csv", [result])
    assert len(complete(manifest, tmp_path)[1]) == 1


def test_manifests_cannot_be_overwritten_with_different_configs(tmp_path, monkeypatch):
    from campaign_v7 import prepare

    monkeypatch.setattr(prepare, "CAMPAIGN", tmp_path)
    emit("test", [dict(inner="gd")])
    emit("test", [dict(inner="gd")])
    with pytest.raises(ValueError, match="refusing to replace"):
        emit("test", [dict(inner="cg")])
