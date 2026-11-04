"""Ensure submission automation never retries connection or ambiguous failures."""

import csv
import json
import subprocess
import sys

import pytest

from regression_v1 import submit_followups


def setup(monkeypatch, tmp_path, responses):
    (tmp_path / "regression_v1").mkdir()
    monkeypatch.setattr(submit_followups, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["submit_followups", "--initial-delay", "0"])
    monkeypatch.setattr(submit_followups.time, "sleep", lambda _: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        code, output = responses.pop(0)
        return subprocess.CompletedProcess(command, code, output, "")

    monkeypatch.setattr(submit_followups.subprocess, "run", run)
    return calls


def test_connection_failure_stops_before_submission(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path, [(3, "SSH session unavailable")])
    with pytest.raises(SystemExit) as error:
        submit_followups.main()
    assert error.value.code == 3 and len(calls) == 1
    assert "--yes" not in calls[0]


def test_ambiguous_submission_cannot_be_blindly_repeated(monkeypatch, tmp_path):
    responses = [(2, json.dumps({"test_only": {"ok": True}})), (3, "connection lost")]
    calls = setup(monkeypatch, tmp_path, responses)
    with pytest.raises(SystemExit) as error:
        submit_followups.main()
    assert error.value.code == 3 and len(calls) == 2
    assert "--yes" in calls[-1]
    status = next(
        csv.DictReader(
            (tmp_path / "regression_v1/followup_submission_status.csv").open()
        )
    )
    assert status["state"] == "submission_outcome_unknown"
    with pytest.raises(SystemExit, match="refusing duplicate"):
        submit_followups.main()
    assert len(calls) == 2
