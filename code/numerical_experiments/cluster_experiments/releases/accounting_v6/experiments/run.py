"""Run one manifest task with atomic results, clean attempts, and process limits.

Every attempt starts from scratch, so the final attempt is a clean timing
measurement. Earlier attempts are kept for campaign accounting only and are
never added to the time that ranks a configuration.
"""

import argparse
import json
import math
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, process_time

from threadpoolctl import threadpool_info, threadpool_limits

from . import baselines, solvers
from .data import digest
from .io import identity, read_csv, write_csv
from .limits import budgets
from .outcomes import error_status
from .problems import build
from .runtime import cpu_model, threads
from .tuning import resolve


def interrupted_result(config, limits, stage):
    return dict(
        status="interrupted",
        **limits,
        converged=False,
        interrupted_stage=stage,
        solver_setup_seconds=0.0,
        solve_seconds=0.0,
    )


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
    # Handlers precede loading so that a Slurm warning during a long build or
    # calibration yields an orderly "interrupted" record, not a hard kill.
    solvers.INTERRUPTED = False
    solvers.install_signals()
    started = perf_counter()
    cpu_started = process_time()
    source_hashes = {
        f"experiments/{path.name}": digest(path)
        for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
    }
    p = build(config, data_dir) if problem is None else problem
    generation = perf_counter() - started
    work_before = p.work_snapshot()
    original = dict(config)
    output.mkdir(parents=True, exist_ok=True)
    tuning = dict(tuning_seconds=0.0)
    history = []
    if solvers.INTERRUPTED:
        result = interrupted_result(config, limits, "build")
    elif config["inner"] in ("osqp", "scs"):
        result, history = baselines.external(p, config)
    elif config["inner"] == "pdhg":
        result, history = baselines.pdhg(p, config)
    else:
        config, tuning, inner = resolve(p, config, return_inner=True)
        if solvers.INTERRUPTED:
            result = interrupted_result(config, limits, "calibration")
        else:
            # No checkpoint callback: attempts never resume, so periodic
            # checkpoint I/O would only add untimed-for-baselines overhead.
            result, history = solvers.solve(p, config, prepared_inner=inner)
    for name in ("row_updates", "block_factorizations"):
        if name in result:
            result[f"solve_and_setup_{name}"] = result[name]
            result[name] += tuning.get(f"calibration_{name}", 0)
    measured_work = p.work_since(work_before)
    exposed = config["inner"] not in ("osqp", "scs")
    result.update(
        hessian_products_total=(
            measured_work["hessian_products"]
            if exposed and p.family != "logistic"
            else float("nan")
        ),
        **{
            key: value if exposed else float("nan")
            for key, value in measured_work.items()
            if key != "hessian_products"
        },
        work_count_scope="problem_operators_after_build_including_calibration"
        if exposed
        else "native_backend_work_not_exposed",
        work_count_complete=exposed,
        hessian_product_unit="not_applicable"
        if p.family == "logistic"
        else "quadratic_operator_vector_equivalent",
    )
    write_csv(output / "history.csv", history)
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
    provenance = original.get("matrix_source", {}).get("feature_provenance", {})
    result.update(
        **limits,
        config=json.dumps(original, sort_keys=True),
        run_id=identity(original),
        sigma=config.get("sigma", ""),
        stride=config.get("stride", ""),
        generation_seconds=generation,
        component_seconds=total,
        total_seconds=perf_counter() - started,
        data_preprocessing_seconds=preprocessing,
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


def supersede(output):
    """Move every earlier attempt aside so a forced rerun is measured alone."""
    archive = output / "superseded" / str(time.time_ns())
    moved = False
    for name in ("attempts", "result.csv", "history.csv", "checkpoint.npz"):
        source = output / name
        if source.exists():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(archive / name))
            moved = True
    return archive if moved else None


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
                        status=error_status(type(error).__name__, str(error)),
                        error_type=type(error).__name__,
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
    superseded = supersede(output) if args.force else None
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

    # Forward the Slurm warning; the worker records an orderly interruption.
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
                        converged=False,
                        returncode=child.returncode,
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
    clean = journal.current["elapsed_seconds"]
    result.update(
        # The final attempt started from scratch and was fully observed, so
        # its wall time is the clean measurement used for every ranking.
        total_seconds=clean,
        clean_attempt_seconds=clean,
        attempt_worker_wall_seconds=clean,
        all_attempts_seconds=(
            journal.known_seconds if journal.complete else float("nan")
        ),
        recorded_attempt_seconds=journal.known_seconds,
        attempts=len(journal.rows),
        prior_unclean_attempts=sum(
            r["status"] == "unclean_stop" for r in journal.rows[:-1]
        ),
        campaign_accounting_complete=journal.complete,
        setup_accounting_complete=True,
        superseded_attempts="" if superseded is None else str(superseded),
        setup_policy="independent",
        timing_scope="independent_final_attempt",
        primary_eligible=True,
    )
    for name in ["generation_seconds", "tuning_seconds", "solver_setup_seconds"]:
        values = [r.get(name) for r in journal.rows]
        result[f"cumulative_{name}"] = (
            sum(values)
            if all(v is not None and math.isfinite(v) for v in values)
            else float("nan")
        )
    result["end_to_end_seconds"] = (
        clean + float(result.get("data_preprocessing_seconds", 0.0) or 0.0)
        if str(result.get("preprocessing_cost_complete", True)) == "True"
        else float("nan")
    )
    write_csv(output / "result.csv", [result])
    journal.close()
    if child.returncode:
        raise SystemExit(child.returncode if child.returncode > 0 else 1)


if __name__ == "__main__":
    main()
