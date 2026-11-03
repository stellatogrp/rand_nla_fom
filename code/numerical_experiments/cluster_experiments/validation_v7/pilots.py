"""Prepare bounded offline validation pilots for a sealed numerical release."""

import argparse
import json
from pathlib import Path

from releases.accounting_v7.experiments.io import identity, read_csv, write_csv

from .releases import ROOT, digest, verify


def configurations(smoke=False):
    common = dict(
        implementation="accounting_v7",
        seed=0,
        phase="smoke" if smoke else "pilot",
        ratio=10.0,
        sigma=0.2,
        stride=5,
        inner_initialization="warm",
        setup_policy="independent",
        blas_threads=1,
        check_source="recurrence",
        budget_unit="work",
        time_limit=5.0 if smoke else 60.0,
        max_inner=200 if smoke else 10000,
        max_outer=1000,
        inner_time_limit=5.0 if smoke else 30.0,
        max_inner_per_outer=10000,
        tolerance=1e-6,
        sweep="release_validation",
    )
    for inner in ["cg", "gd", "nystrom_pcg", "block_kaczmarz"]:
        c = dict(
            common,
            family="qp",
            dataset="synthetic",
            orientation="native",
            side=4 if smoke else 40,
            kappa=1e4,
            inner=inner,
        )
        if inner == "nystrom_pcg":
            c["rank"] = 4 if smoke else 32
        if inner == "block_kaczmarz":
            c.update(block_size=1, max_inner=1, stride=32 if smoke else 640)
        yield c
    source = dict(
        kind="random_sparse",
        rows=128 if smoke else 16384,
        cols=64 if smoke else 4096,
        entries_per_row=8,
        distribution="normal",
        seed=314159,
    )
    for family in ["lasso", "logistic"]:
        for orientation in ["native", "transpose"]:
            for inner in ["gd", "pdhg"]:
                yield dict(
                    common,
                    family=family,
                    dataset="validation_random",
                    matrix_source=source,
                    orientation=orientation,
                    inner=inner,
                )


def prepare(output, smoke=False):
    seal = verify(ROOT, "accounting_v7")
    configs = list(configurations(smoke))
    if output.exists():
        expected = [identity(c) for c in configs]
        if [r["run_id"] for r in read_csv(output)] != expected:
            raise ValueError("pilot manifest exists with different configurations")
        metadata = json.loads(output.with_suffix(".seal.json").read_text())
        if (
            metadata["manifest_sha256"] != digest(output.read_bytes())
            or metadata["archive_sha256"] != seal["archive_sha256"]
        ):
            raise ValueError("existing pilot manifest seal does not match")
        return configs
    write_csv(
        output,
        [
            dict(task_id=i, run_id=identity(c), config=json.dumps(c, sort_keys=True))
            for i, c in enumerate(configs)
        ],
    )
    metadata = dict(
        release="accounting_v7",
        archive_sha256=seal["archive_sha256"],
        manifest_sha256=digest(output.read_bytes()),
        tasks=len(configs),
        data_scope="seeded synthetic validation; not benchmark conclusions",
    )
    output.with_suffix(".seal.json").write_text(json.dumps(metadata, indent=2) + "\n")
    output.chmod(0o444)
    output.with_suffix(".seal.json").chmod(0o444)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(
        f"{len(prepare(args.output, args.smoke))} pilot configurations: {args.output}"
    )


if __name__ == "__main__":
    main()
