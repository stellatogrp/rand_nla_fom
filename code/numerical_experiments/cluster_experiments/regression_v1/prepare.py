"""Prepare immutable regression pilots; never submit or overwrite a manifest."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from campaign_v7.advance import seal_output
from campaign_v7.runner import verify_campaign
from releases.accounting_v7.experiments.io import read_csv
from releases.accounting_v7.experiments.limits import budgets
from releases.accounting_v7.experiments.plan import emit

ROOT = Path(__file__).resolve().parents[1]


def configurations(source):
    """Keep the original data/objectives and independently charge every setup."""
    templates = {}
    for row in read_csv(source):
        config = json.loads(row["config"])
        key = (config["dataset"].split("_rf_")[0], config["family"], config["inner"])
        templates.setdefault(key, config)
    initial, followup = [], []
    for dataset in ("smallnorb", "realsim"):
        methods = []
        for family, inner, ratio, sigma in (
            ("lasso", "nystrom_pcg", 1e6, 0.2),
            ("logistic", "gd", 1e4, 0.9),
        ):
            config = deepcopy(templates[dataset, family, inner])
            config.update(
                campaign_id="regression_v1_20261103",
                sweep="convergence_recovery_pilot",
                phase="pilot",
                setup_policy="independent",
                ratio=ratio,
                sigma=sigma,
                stride=1,
                inner_initialization="warm",
                time_limit=7200.0,
                inner_time_limit=3600.0,
                max_inner=100_000,
                max_inner_per_outer=10_000,
                max_outer=10_000,
                tolerance=1e-6,
            )
            budgets(config)
            initial.append(config)
            methods.append(config)
            # Alternatives are prepared, not automatically submitted. Inspect
            # initial histories, setup costs, and resource stats before choosing.
            for neighbor in (1e4, 1e8) if family == "lasso" else (1e2, 1e6):
                followup.append(dict(config, ratio=neighbor, sweep="penalty_pilot"))
            for mode in ("fixed", "gaussian"):
                followup.append(
                    dict(
                        config, inner_initialization=mode, sweep="initialization_pilot"
                    )
                )
        for config in methods:
            baseline = deepcopy(templates[dataset, config["family"], "pdhg"])
            baseline.update(
                campaign_id=config["campaign_id"],
                sweep="baseline_budget_pilot",
                time_limit=config["time_limit"],
                inner_time_limit=config["inner_time_limit"],
                baseline_step=1.0,
                ratio=1.0,
            )
            followup.append(baseline)
    return initial, followup


def prepare(source, output):
    verify_campaign()
    initial, followup = configurations(source)
    paths = [output / "initial.csv", output / "followup_candidates.csv"]
    if any(p.exists() or p.with_suffix(".seal.json").exists() for p in paths):
        raise ValueError(
            "refusing to overwrite regression inputs; choose a new directory"
        )
    for path, configs in zip(paths, (initial, followup), strict=True):
        emit(path, configs)
        seal_output(path)
        print(f"{path}: {len(configs)} configurations")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "campaign_v7/manifests/regression_pilots.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output_dir)


if __name__ == "__main__":
    main()
