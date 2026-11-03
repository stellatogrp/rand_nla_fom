"""Derive a narrow core policy from matching timings and whole-job accounting."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from .feature_store import atomic_json
from .io import identity, read_csv


def choose(reference, candidates, profiles):
    """Only successful full-solve profiles on the same node can qualify."""
    if reference["state"] != "COMPLETED" or reference["alloc_cpus"] != 1:
        raise ValueError("reference must be a completed one-core job")
    matches = {}
    for job, stats in candidates.items():
        if stats["state"] != "COMPLETED" or stats["nodes"] != reference["nodes"]:
            continue
        rows = profiles[job]
        if len(rows) < 2 or any(
            r.get("converged") != "True" or r.get("operation") != "solve" for r in rows
        ):
            continue
        count = stats["alloc_cpus"]
        efficiency = stats["total_cpu_s"] / (stats["elapsed_s"] * count)
        speedup = reference["elapsed_s"] / stats["elapsed_s"]
        if count > 1 and efficiency >= 0.9 and speedup >= 1.2:
            matches[job] = dict(
                threads=count,
                efficiency=efficiency,
                speedup=speedup,
                elapsed=stats["elapsed_s"],
            )
    selected = (
        min(matches, key=lambda job: matches[job]["elapsed"]) if matches else None
    )
    return selected, matches


def apply(config, policy):
    config = dict(config)
    if all(config.get(k) == value for k, value in policy["case"].items()):
        config.update(
            blas_threads=policy["threads"],
            core_policy_id=identity(policy),
            core_cpu_model=policy["cpu_model"],
            core_node=policy["node"],
        )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slurm-helper", type=Path, required=True)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--reference-job", required=True)
    parser.add_argument("--candidate-jobs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID"):
        parser.error("derive the policy locally through the Slurm helper")
    accounting, profiles = {}, {}
    for job in [args.reference_job, *args.candidate_jobs]:
        result = subprocess.run(
            [
                sys.executable,
                str(args.slurm_helper),
                "stats",
                job,
                "--config",
                str(args.cluster_config),
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            print(result.stdout + result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
        stats = json.loads(result.stdout)["stats"]
        accounting[job] = stats
        profiles[job] = read_csv(args.profiles_dir / f"qp_{stats['alloc_cpus']}.csv")
    reference = accounting[args.reference_job]
    if any(
        len(rows) < 2
        or any(
            r.get("converged") != "True"
            or r.get("dataset") != "qp_100"
            or r.get("inner") != "nystrom_pcg"
            or r.get("operation") != "solve"
            or int(r["threads"]) != accounting[job]["alloc_cpus"]
            for r in rows
        )
        for job, rows in profiles.items()
    ):
        parser.error(
            "profiles must be matching completed million-variable Nyström solves"
        )
    selected, candidates = choose(
        reference, {j: accounting[j] for j in args.candidate_jobs}, profiles
    )
    models = {r["cpu_model"] for rows in profiles.values() for r in rows}
    if len(models) != 1:
        parser.error("CPU models differ across profiles")
    policy = dict(
        threads=1 if selected is None else candidates[selected]["threads"],
        case=dict(family="qp", inner="nystrom_pcg", side=100, rank=128),
        cpu_model=models.pop(),
        node=reference["nodes"],
        selected_job=selected,
        reference_job=args.reference_job,
        whole_job_accounting=accounting,
        qualifying_candidates=candidates,
        reference_median_seconds=statistics.median(
            float(r["seconds"]) for r in profiles[args.reference_job]
        ),
        min_efficiency=0.9,
        min_speedup=1.2,
        setup_policy="independent",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, policy)
    print(
        f"selected {policy['threads']} cores for {policy['case']}; "
        f"node={policy['node']}"
    )


if __name__ == "__main__":
    main()
