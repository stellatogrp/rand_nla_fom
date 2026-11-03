"""Freeze a traced numerical release and create fresh QP evaluation inputs."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from releases.figure_v8.experiments.io import identity, read_csv, write_csv
from releases.figure_v8.experiments.plan import emit
from validation_v7.releases import ROOT, freeze

RELEASE = "figure_v8"


def configurations(source, reference_results=None):
    if reference_results is None:
        reference_results = ROOT / "results/qp_evaluate_v7_20261031"
    configs = [json.loads(r["config"]) for r in read_csv(source)]
    selected = []
    for original in configs:
        if (
            original["inner"] not in ("osqp", "scs", "pdhg")
            and original.get("inner_initialization", "warm") != "warm"
        ):
            continue
        modes = ("warm", "fixed", "gaussian")
        if original["inner"] in ("osqp", "scs", "pdhg"):
            modes = ("warm",)
        for mode in modes:
            config = deepcopy(original)
            for key, cast in (("sigma", float), ("stride", int)):
                if config.get(key) == "optimal":
                    path = reference_results / identity(original) / "result.csv"
                    reference = read_csv(path)[0]
                    if json.loads(reference["config"]) != original:
                        raise ValueError("warm reference identity mismatch")
                    config[key] = cast(reference[key])
            config.update(
                implementation=RELEASE,
                campaign_id="main_figures_v8_20261103",
                inner_initialization=mode,
                initialization_seed=original["seed"] + 200003,
                gaussian_scale=1.0,
                fixed_initial_value=0.0,
                trace_inner=original["inner"] not in ("osqp", "scs", "pdhg"),
                max_trace_records=100_000,
            )
            selected.append(config)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=ROOT / "results/qp_evaluate_v7_20261031",
    )
    args = parser.parse_args()
    destination = args.output_dir
    if destination.exists():
        raise ValueError("refusing to overwrite figure campaign inputs")
    directory = ROOT / "releases" / RELEASE
    manifest = dict(
        implementation=RELEASE,
        source_sha256={
            str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((directory / "experiments").glob("*.py"))
        },
    )
    (directory / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    seal = freeze(ROOT, RELEASE, "QP scalar traces and charged boundary verification")
    configs = configurations(args.source, args.reference_results)
    rows = emit(destination / "evaluate.csv", configs)
    pilots = []
    for inner, rank in (("cg", None), ("nystrom_pcg", 128)):
        pilots.append(
            next(
                c
                for c in configs
                if c["inner"] == inner
                and c.get("rank") == rank
                and c["side"] == 100
                and c["seed"] == 3
                and c["inner_initialization"] == "warm"
            )
        )
    emit(destination / "pilots.csv", pilots)
    write_csv(
        destination / "source.csv",
        [
            dict(
                source_manifest=str(args.source),
                source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                numerical_archive_sha256=seal["archive_sha256"],
                initialization_policy=("warm-selected parameters fixed across modes"),
                runs=len(rows),
            )
        ],
    )
    sources = [
        *sorted((ROOT / "figure_campaign").glob("*.py")),
        *sorted((ROOT / "plots").glob("*.py")),
        ROOT / "slurm/figure_v8.slurm",
        ROOT / "slurm/figure_v8_plot.slurm",
    ]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    for path in destination.glob("*.csv"):
        metadata = dict(
            manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            numerical_archive_sha256=seal["archive_sha256"],
            source_sha256=hashes,
        )
        path.with_suffix(".seal.json").write_text(json.dumps(metadata, indent=2) + "\n")
        path.chmod(0o444)
        path.with_suffix(".seal.json").chmod(0o444)
    print(f"{len(rows)} fresh evaluation runs; {len(pilots)} pilots")


if __name__ == "__main__":
    main()
