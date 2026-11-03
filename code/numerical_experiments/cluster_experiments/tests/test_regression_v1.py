"""Protect objectives, setup policy, and immutable follow-up inputs."""

import hashlib
import json
from pathlib import Path

import pytest

from regression_v1.prepare import configurations, prepare
from releases.accounting_v7.experiments.io import identity, read_csv
from releases.accounting_v7.experiments.limits import budgets

SOURCE = Path("campaign_v7/manifests/regression_pilots.csv")


def test_recovery_pilots_preserve_objectives_and_bound_every_initialization():
    initial, candidates = configurations(SOURCE)
    originals = {
        (c["dataset"], c["family"], c["inner"]): c
        for r in read_csv(SOURCE)
        if (c := json.loads(r["config"]))
    }
    assert len(initial) == 4
    assert len(candidates) == 20
    assert len({identity(c) for c in initial + candidates}) == 24
    for c in initial + candidates:
        original = originals[c["dataset"], c["family"], c["inner"]]
        for field in ("matrix_source", "seed", "orientation", "tolerance"):
            assert c[field] == original[field]
        assert c["setup_policy"] == "independent"
        assert c["blas_threads"] == 1
        assert budgets(c)["time_limit"] == 7200
        assert c["tolerance"] == 1e-6
    for warm in initial:
        variants = [
            c
            for c in initial + candidates
            if c["dataset"] == warm["dataset"]
            and c["inner"] == warm["inner"]
            and c["ratio"] == warm["ratio"]
        ]
        assert {c["inner_initialization"] for c in variants} == {
            "warm",
            "fixed",
            "gaussian",
        }


def test_prepared_manifests_are_sealed_and_cannot_be_replaced(tmp_path):
    paths = prepare(SOURCE, tmp_path)
    for path in paths:
        metadata = json.loads(path.with_suffix(".seal.json").read_text())
        assert (
            metadata["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for row in read_csv(path):
            assert row["run_id"] == identity(json.loads(row["config"]))
    with pytest.raises(ValueError, match="overwrite"):
        prepare(SOURCE, tmp_path)
