"""Prepare four independently measured pilots and seal their numerical sources."""

import argparse
import hashlib
import json
from pathlib import Path

from releases.regression_v9.experiments.io import read_csv
from releases.regression_v9.experiments.limits import budgets
from releases.regression_v9.experiments.plan import emit
from validation_v7.releases import ROOT, freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "regression_v1/inputs/followup_candidates.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "regression_v2/inputs"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("refusing to overwrite regression v2 inputs")
    originals = [json.loads(r["config"]) for r in read_csv(args.source)]
    configs = []
    # Put the stopping-rule audit first. Each run loads and sets up independently.
    for dataset in ("realsim", "smallnorb"):
        for family, inner in (("lasso", "nystrom_pcg"), ("logistic", "gd")):
            original = next(
                c
                for c in originals
                if c["dataset"].startswith(dataset + "_rf_")
                and c["family"] == family
                and c["inner"] == inner
                and c["inner_initialization"] == "warm"
            )
            config = dict(
                original,
                implementation="regression_v9",
                campaign_id="regression_v2_20261105",
                terminal_policy="common_certificate",
                certificate_stride=10,
                sweep="certificate_and_convergence_pilot",
                stride=1,
                ratio=(1e4 if dataset == "realsim" else 1e5)
                if family == "lasso"
                else 100.0,
                sigma=0.2 if family == "lasso" else 0.99,
            )
            budgets(config)
            configs.append(config)
    directory = ROOT / "releases/regression_v9"
    (directory / "source_manifest.json").write_text(
        json.dumps(
            dict(
                implementation="regression_v9",
                source_sha256={
                    str(p.relative_to(directory)): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in sorted((directory / "experiments").glob("*.py"))
                },
            ),
            indent=2,
        )
        + "\n"
    )
    seal = freeze(
        ROOT,
        "regression_v9",
        "True common certificates for regression pilots; setup and audit costs charged",
    )
    path = args.output_dir / "pilots.csv"
    emit(path, configs)
    sources = [
        *sorted((ROOT / "regression_v2").glob("*.py")),
        ROOT / "slurm/regression_v2.slurm",
    ]
    metadata = dict(
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        numerical_archive_sha256=seal["archive_sha256"],
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        parent_manifest=str(args.source.relative_to(ROOT)),
        parent_manifest_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
    )
    path.with_suffix(".seal.json").write_text(json.dumps(metadata, indent=2) + "\n")
    path.chmod(0o444)
    path.with_suffix(".seal.json").chmod(0o444)
    print(f"Sealed {len(configs)} regression v2 pilots")


if __name__ == "__main__":
    main()
