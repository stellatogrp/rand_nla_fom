"""Run one manifest task with atomic results, checkpoints, and process limits."""

import argparse
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from . import baselines, solvers
from .data import digest
from .io import identity, read_csv, write_csv
from .limits import budgets
from .problems import build
from .runtime import cpu_model, threads
from .tuning import resolve


def load_state(path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as f:
        state = json.loads(str(f["metadata"]))
        state.update({name: f[name].copy() for name in ("x", "z", "dual", "w")})
    return state


def save_state(path, state):
    arrays = {name: state[name] for name in ("x", "z", "dual", "w")}
    metadata = {k: v for k, v in state.items() if k not in arrays}
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays, metadata=json.dumps(metadata))
    os.replace(temporary, path)


def execute(config, output, data_dir, problem=None, shared_context=None):
    policy = config.get("setup_policy", "independent")
    if policy not in ("independent", "shared_pilot"):
        raise ValueError("unknown setup policy")
    if policy == "shared_pilot":
        if (
            config.get("phase") not in ("pilot", "smoke", "profile")
            or problem is None
            or shared_context is None
        ):
            raise ValueError("shared loading is restricted to explicit pilot workers")
    elif problem is not None or shared_context is not None:
        raise ValueError("main runs must build and measure their own problem setup")
    limits = budgets(config)
    started = perf_counter()
    cpu_started = process_time()
    source_hashes = {
        f"experiments/{path.name}": digest(path)
        for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
    }
    p = build(config, data_dir) if problem is None else problem
    generation = perf_counter() - started
    original = dict(config)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.npz"
    state = load_state(checkpoint)
    histories = read_csv(output / "history.csv")
    if state:
        histories = [row for row in histories if int(row["iteration"]) < state["outer"]]
    else:
        histories = []

    def callback(current, history):
        current["resolved_sigma"] = config["sigma"]
        current["resolved_stride"] = config["stride"]
        histories.extend(history)
        write_csv(output / "history.csv", histories)
        save_state(checkpoint, current)

    solvers.install_signals()
    tuning = dict(tuning_seconds=0.0)
    if config["inner"] in ("osqp", "scs"):
        result, history = baselines.external(p, config)
    elif config["inner"] == "pdhg":
        result, history = baselines.pdhg(p, config)
    else:
        config, tuning, inner = resolve(p, config, return_inner=True)
        if state:
            config.update(
                sigma=state["resolved_sigma"], stride=state["resolved_stride"]
            )
        result, history = solvers.solve(
            p, config, callback, state, prepared_inner=inner
        )
    histories.extend(history)
    write_csv(output / "history.csv", histories)
    total = (
        generation
        + tuning["tuning_seconds"]
        + result.get("solver_setup_seconds", 0.0)
        + result.get("solve_seconds", 0.0)
    )
    preprocessing = (
        original.get("matrix_source", {})
        .get("feature_provenance", {})
        .get("preparation_seconds", 0.0)
    )
    result.update(
        **limits,
        config=json.dumps(original, sort_keys=True),
        run_id=identity(original),
        sigma=config.get("sigma", ""),
        stride=config.get("stride", ""),
        generation_seconds=generation,
        total_seconds=total,
        data_preprocessing_seconds=preprocessing,
        end_to_end_seconds=total + preprocessing,
        process_seconds=perf_counter() - started,
        cpu_seconds=process_time() - cpu_started,
        blas_threads=threads(original),
        threadpools=json.dumps(threadpool_info(), sort_keys=True),
        implementation=original.get("implementation", "unversioned"),
        setup_policy=policy,
        timing_scope="independent_worker"
        if policy == "independent"
        else "incremental_pilot_after_shared_loading",
        primary_eligible=policy == "independent",
        shared_setup_id="" if shared_context is None else shared_context["id"],
        shared_setup_reference_seconds=(
            0.0 if shared_context is None else shared_context["seconds"]
        ),
        n=p.n,
        L=p.L,
        mu=p.mu,
        provenance=p.provenance,
        **tuning,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        hostname=platform.node(),
        cpu_model=cpu_model(),
        python=platform.python_version(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", "local"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        package_versions=json.dumps(
            {name: version(name) for name in ["numpy", "scipy", "osqp", "scs", "cvxpy"]}
        ),
        source_revision=os.environ.get("EXPERIMENT_SOURCE_REVISION", "unknown"),
        source_sha256=json.dumps(source_hashes, sort_keys=True),
        matrix_provenance=(
            json.dumps(json.loads(p.provenance)["matrix"], sort_keys=True)
            if p.family != "qp"
            else "[]"
        ),
    )
    provenance = original.get("matrix_source", {}).get("feature_provenance", {})
    result.update(
        component_seconds=total,
        total_seconds=perf_counter() - started,
        preprocessing_cost_complete=provenance.get("preparation_cost_complete", True),
        preprocessing_cost_scope=provenance.get(
            "preparation_cost_scope", "successful_artifact_build"
        ),
        preprocessing_campaign_accounting_complete=provenance.get(
            "preparation_campaign_accounting_complete", False
        ),
        setup_accounting_complete=True,
        end_to_end_scope="sum_of_measured_worker_and_one_artifact_build",
        end_to_end_is_contiguous_wall_time=False,
    )
    result["end_to_end_seconds"] = (
        result["total_seconds"] + preprocessing
        if result["preprocessing_cost_complete"] and policy == "independent"
        else float("nan")
    )
    write_csv(output / "result.csv", [result])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--memory-gb", type=float, default=12.0)
    parser.add_argument("--setup-budget", type=float, default=600.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    rows = read_csv(args.manifest)
    if not 0 <= args.task_id < len(rows):
        parser.error("task-id outside manifest")
    row = rows[args.task_id]
    config = json.loads(row["config"])
    if config.get("setup_policy", "independent") != "independent":
        parser.error("shared pilot manifests require experiments.pilot_batch")
    limits = budgets(config)
    worker_threads = threads(config)
    if not math.isfinite(args.setup_budget) or args.setup_budget < 0:
        parser.error("setup-budget must be finite and nonnegative")
    if identity(config) != row["run_id"]:
        parser.error("manifest task hash mismatch")
    output = args.results_dir / row["run_id"]
    if args.worker and args.attempt_output:
        output = args.attempt_output
    previous = read_csv(output / "result.csv")
    if previous and previous[0]["status"] != "interrupted" and not args.force:
        print(f"already complete: {output}")
        return
    if args.force and not args.worker:
        # Explicit rerun starts fresh rather than inheriting a previous optimizer.
        (output / "checkpoint.npz").unlink(missing_ok=True)
        (output / "history.csv").unlink(missing_ok=True)
    if args.worker:
        # Linux only; the scheduler enforces job memory too. The child leaves
        # enough headroom for its parent to write a failure row after an OOM.
        if sys.platform == "linux" and args.memory_gb > 0:
            memory = int(args.memory_gb * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        try:
            with threadpool_limits(limits=worker_threads):
                result = execute(config, output, args.data_dir)
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as error:
            write_csv(
                output / "result.csv",
                [
                    dict(
                        config=row["config"],
                        run_id=row["run_id"],
                        status="error",
                        **limits,
                        converged=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                ],
            )
            raise
        return
    if not os.environ.get("SLURM_JOB_ID") and config.get("phase") != "smoke":
        parser.error(
            "large runs require a Slurm compute allocation; use phase smoke locally"
        )
    from .accounting import Journal

    journal = Journal(output / "attempts")
    attempt_output = output / "attempts" / journal.current["id"]
    command = [
        sys.executable,
        "-m",
        f"{__package__}.run",
        *sys.argv[1:],
        "--worker",
        "--attempt-output",
        str(attempt_output.resolve()),
    ]
    child = subprocess.Popen(command)

    # Forward the Slurm warning; the worker keeps its last valid checkpoint.
    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGUSR1, forward)
    signal.signal(signal.SIGTERM, forward)
    started = journal.started
    try:
        child.wait(timeout=limits["time_limit"] + args.setup_budget)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        write_csv(
            attempt_output / "result.csv",
            [
                dict(
                    config=row["config"],
                    run_id=row["run_id"],
                    status="worker_timeout",
                    **limits,
                    converged=False,
                    process_seconds=perf_counter() - started,
                )
            ],
        )
    else:
        if child.returncode and not (attempt_output / "result.csv").exists():
            write_csv(
                attempt_output / "result.csv",
                [
                    dict(
                        config=row["config"],
                        run_id=row["run_id"],
                        status="worker_failed",
                        **limits,
                        returncode=child.returncode,
                        converged=False,
                    )
                ],
            )
    rows = read_csv(attempt_output / "result.csv")
    result = rows[0]
    prep = config.get("matrix_source", {}).get("feature_provenance", {})
    result.setdefault(
        "data_preprocessing_seconds", prep.get("preparation_seconds", 0.0)
    )
    result.setdefault(
        "preprocessing_cost_complete", prep.get("preparation_cost_complete", True)
    )
    journal.checkpoint(
        result["status"],
        **{
            name: float(result[name]) if result.get(name) not in (None, "") else None
            for name in [
                "generation_seconds",
                "tuning_seconds",
                "solver_setup_seconds",
                "solve_seconds",
            ]
        },
    )
    result.update(
        attempt_worker_wall_seconds=journal.current["elapsed_seconds"],
        recorded_attempt_seconds=journal.known_seconds,
        total_seconds=journal.known_seconds if journal.complete else float("nan"),
        attempts=len(journal.rows),
        setup_accounting_complete=journal.complete,
        setup_policy="independent",
        timing_scope="independent_attempts_including_retries",
        primary_eligible=journal.complete,
    )
    for name in ["generation_seconds", "tuning_seconds", "solver_setup_seconds"]:
        values = [r.get(name) for r in journal.rows]
        result[f"cumulative_{name}"] = (
            sum(values)
            if all(v is not None and math.isfinite(v) for v in values)
            else float("nan")
        )
    result["end_to_end_seconds"] = (
        result["total_seconds"]
        + float(result.get("data_preprocessing_seconds", 0.0) or 0.0)
        if journal.complete
        and str(result.get("preprocessing_cost_complete", True)) == "True"
        else float("nan")
    )
    write_csv(output / "result.csv", [result])
    journal.close()
    if child.returncode:
        raise SystemExit(child.returncode if child.returncode > 0 else 1)


if __name__ == "__main__":
    main()
