"""Generate follow-up manifests only from complete, independently measured runs."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from releases.accounting_v7.experiments import plan
from releases.accounting_v7.experiments.io import identity, read_csv, write_csv
from releases.accounting_v7.experiments.manifest import clean_seconds
from releases.accounting_v7.experiments.outcomes import blocked, is_method_outcome
from validation_v7.releases import ROOT, verify

from .runner import verify_campaign


def complete(manifest, directory):
    expected = read_csv(manifest)
    if not expected:
        raise ValueError("input manifest is empty")
    results, bad = {}, {}
    for task in expected:
        path = directory / task["run_id"] / "result.csv"
        rows = read_csv(path)
        if not rows:
            raise ValueError(f"incomplete stage; missing {task['run_id']}")
        row = rows[0]
        config = json.loads(task["config"])
        if row.get("run_id") != identity(config) or json.loads(row["config"]) != config:
            raise ValueError("result identity/configuration does not match manifest")
        if row.get("setup_policy", "independent") != "independent":
            raise ValueError("shared-loading pilots cannot advance a main campaign")
        if not is_method_outcome(row):
            bad[task["run_id"]] = row
        elif row.get("converged") == "True" and not math.isfinite(clean_seconds(row)):
            bad[task["run_id"]] = dict(status="no_clean_time")
        results[task["run_id"]] = row
    if bad:
        raise ValueError(blocked(bad))
    return expected, results


def seal_output(path):
    metadata = dict(
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        campaign_seal_sha256=hashlib.sha256(
            (ROOT / "campaign_v7/source_seal.json").read_bytes()
        ).hexdigest(),
        numerical_archive_sha256=verify(ROOT, "accounting_v7")["archive_sha256"],
    )
    path.with_suffix(".seal.json").write_text(json.dumps(metadata, indent=2) + "\n")
    path.chmod(0o444)
    path.with_suffix(".seal.json").chmod(0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["qp-steps", "qp-evaluate", "kaczmarz-expand", "regression-readiness"],
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_campaign()
    tasks, results = complete(args.manifest, args.results_dir)
    if args.output.exists():
        raise ValueError(
            "follow-up output already exists; use a new immutable filename"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "regression-readiness":
        groups = {}
        for task in tasks:
            config = json.loads(task["config"])
            key = (config["dataset"], config["family"], config["inner"])
            groups.setdefault(key, []).append(results[task["run_id"]])
        rows = []
        for (dataset, family, inner), records in groups.items():
            converged = [r for r in records if r.get("converged") == "True"]
            rows.append(
                dict(
                    dataset=dataset,
                    family=family,
                    inner=inner,
                    tested=len(records),
                    converged=len(converged),
                    ready_for_tuning=bool(converged),
                    next_action="prepare full sigma/stride tuning"
                    if converged
                    else "revise pilot budget/parameters before expanding",
                )
            )
        write_csv(args.output, rows)
        print(json.dumps(rows, indent=2))
        return
    if args.stage == "kaczmarz-expand":
        configs = [
            json.loads(r["config"])
            for r in read_csv(ROOT / "campaign_v7/manifests/kaczmarz_candidates.csv")
        ]
        screen = [json.loads(t["config"]) for t in tasks]
        selected, eligible = plan.expand(configs, screen, results)
        if not selected:
            print(
                "No screen pair converged; preserve censored results and do not expand."
            )
            return
        for config in selected:
            config.update(
                setup_policy="independent",
                campaign_id="v7_20261031",
                cpu_feature="icelake",
            )
        plan.emit(args.output, selected)
        print(
            f"Expanded {len(selected)} configurations "
            f"for {len(eligible)} method/block pairs"
        )
    else:
        phase = "steps" if args.stage == "qp-steps" else "evaluate"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "releases.accounting_v7.experiments.manifest",
                phase,
                "--from-manifest",
                str(args.manifest),
                "--from-results",
                str(args.results_dir),
                "--output",
                str(args.output),
                "--families",
                "qp",
                "--implementation",
                "accounting_v7",
                "--time-limit",
                "1800",
                "--max-inner",
                "1000000",
                "--max-outer",
                "50000",
                "--inner-time-limit",
                "300",
                "--max-inner-per-outer",
                "100000",
            ],
            check=True,
            cwd=ROOT,
        )
    seal_output(args.output)
    rows = read_csv(args.output)
    print(
        json.dumps(
            dict(
                stage=args.stage,
                tasks=len(rows),
                batch_size=8,
                batches=(len(rows) + 7) // 8,
                manifest=str(args.output),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
