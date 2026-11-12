"""Sanity experiments for Algorithm 3 on small synthetic SDPs.

One invocation is one study: a problem family and size, a list of seeds
(instances), and the methods below run on every instance.

    uv run python -m sdp.run --allow-local --family planted --n 40 --m 80 \
        --rank 4 --seeds 0 1 --name planted_n40

| Label | ``f``-subproblem | Projection | Test |
| --- | --- | --- | --- |
| ``exact`` | Cholesky | eigendecomposition | none (exact proximal ADMM) |
| ``cg_only_sigma<s>`` | conjugate gradients | eigendecomposition | relative |
| ``lanczos_only_sigma<s>_b<b>`` | Cholesky | block Lanczos | relative |
| ``inexact_sigma<s>_b<b>`` | conjugate gradients | block Lanczos | relative |
| ``one_sided_sigma<s>_b<b>`` | conjugate gradients | one-sided, guard | relative |
| ``certified_sigma<s>_b<b>`` | conjugate gradients | one-sided, certified | relative |
| ``near_exact_b<b>`` | conjugate gradients | block Lanczos | ``E <= tolerance`` |

``inexact``, ``one_sided``, and ``certified`` run for every ``sigma`` and
block size (for the one-sided projection the block size is the number of
guard vectors, and ``certified`` adds the Cholesky certificate to the guard);
``lanczos_only`` and ``near_exact`` use the first block size.
``--groups`` selects the rows.  Results land in
``<results-dir>/<name>/``: ``settings.json``, ``runs.csv`` (one row per run:
work, the certificate of the returned point, the theory checks) and
``histories.csv`` (one row per outer iteration, thinned for long runs).  Runs
are appended as they finish and an interrupted study resumes; ``--force``
recomputes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
from pathlib import Path

import numpy as np

from .admm import Trace, certificate, proximal_admm, theory_checks
from .problems import SDP, maxcut, planted, reference
from .projection import GUARD_FACTOR

GROUPS = (
    "exact",
    "cg_only",
    "lanczos_only",
    "inexact",
    "one_sided",
    "certified",
    "near_exact",
)

# An accepted B counts as outside the normal cone when its largest eigenvalue
# exceeds this fraction of the larger of ||B|| and ||Z||.
VIOLATION = 1e-10

KEY_FIELDS = ["study", "family", "n", "m", "rank", "dual_rank", "seed", "label"]
RUN_FIELDS = KEY_FIELDS + [
    "f_solver",
    "projection",
    "sigma",
    "block_size",
    "stride",
    "c",
    "rho",
    "delta",
    "exact_tolerance",
    "cholesky",
    "guard_factor",
    "converged",
    "stopped_by",
    "outer_iterations",
    "seconds",
    # Inner work: conjugate gradient steps, applications of E and of its
    # adjoint (all uses), products of M with a vector, block Lanczos steps,
    # restarts of the Krylov subspace, full eigendecompositions, and checks.
    "cg_steps",
    "forward_applications",
    "adjoint_applications",
    "lanczos_matvecs",
    "lanczos_steps",
    "lanczos_restarts",
    "eigendecompositions",
    "checks",
    # One-sided projection: pairs that passed a test but not the guard,
    # Cholesky certificates attempted and failed, accepted pairs whose B has a
    # positive eigenvalue (relative to the larger of ||B|| and ||Z||, beyond
    # 1e-10) and the largest such eigenvalue, from the off-the-clock
    # diagnostics; then the wall-clock spent on cone pairs and certificates.
    "guard_rejections",
    "certificates",
    "certificate_failures",
    "violations",
    "worst_violation",
    "projection_seconds",
    "certificate_seconds",
    # Subspace dimension behind the accepted cone pairs, against the smallest
    # dimension that could pass the test (rank_needed), per outer iteration.
    "subspace_dim_median",
    "subspace_dim_max",
    "subspace_dim_last",
    "rank_needed_median",
    "rank_needed_max",
    "rank_needed_last",
    "final_Q",
    "final_E",
    "final_P",
    # Certificate of the returned point (sdp.admm.certificate).
    "min_eigenvalue_Z",
    "max_eigenvalue_B",
    "complementarity",
    "coupling_residual",
    "stationarity_residual",
    "stationarity_bound",
    "primal_feasibility",
    "objective",
    "dual_objective",
    "duality_gap",
    "objective_star",
    "objective_reference",
    "reference_status",
    "solution_error",
    # Theory checks (sdp.admm.theory_checks).
    "initial_distance",
    "decrease_violation",
    "ratio_violation",
    "sum_Q_squared",
    "sum_Q_squared_bound",
    "K_delta_plus",
    "hostname",
    "cpu_model",
    "slurm_job_id",
    "python",
    "numpy",
]
HISTORY_FIELDS = KEY_FIELDS + [
    "k",
    "Q",
    "E",
    "P",
    "e_f",
    "e_g",
    "cg_steps",
    "lanczos_matvecs",
    "lanczos_steps",
    "lanczos_restarts",
    "checks",
    "subspace_dim",
    "rank_positive",
    "rank_negative",
    "rank_needed",
    "guard_rejections",
    "certificates",
    "certificate_failures",
    "violation",
    "accepted_by",
    "elapsed",
    "dist",
    "objective",
]


def source_hash() -> str:
    """SHA-256 of this package's sources, so results name the code behind them."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sanity experiments for Algorithm 3 on small synthetic SDPs."
    )
    parser.add_argument("--family", choices=("planted", "maxcut"), default="planted")
    parser.add_argument("--n", type=int, default=40, help="matrix order")
    parser.add_argument("--m", type=int, default=80, help="constraints (planted)")
    parser.add_argument("--rank", type=int, default=4, help="rank of X* (planted)")
    parser.add_argument(
        "--dual-rank",
        type=int,
        default=None,
        help="rank of S* (planted); default n - rank, strict complementarity",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=10.0,
        help="largest over smallest norm of the constraint matrices (planted)",
    )
    parser.add_argument(
        "--density", type=float, default=0.5, help="edge probability (maxcut)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--c", type=float, default=1.0, help="penalty parameter")
    parser.add_argument("--rho", type=float, default=1.0, help="proximal parameter")
    parser.add_argument("--delta", type=float, default=1e-6, help="stopping tolerance")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--stride", type=int, default=1, help="the paper's s")
    parser.add_argument(
        "--f-share",
        type=float,
        default=0.5,
        help="fraction of sigma P granted to the f-residual",
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.1,
        help="restart tolerance on dropped Ritz values, as a fraction of the "
        "projection's share of sigma P",
    )
    parser.add_argument("--near-exact-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--guard-factor",
        type=float,
        default=GUARD_FACTOR,
        help="safety factor of the guard of the one-sided projection",
    )
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--max-outer", type=int, default=50_000)
    parser.add_argument(
        "--time-limit", type=float, default=600.0, help="seconds per run"
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="also solve every instance with CVXPY (Clarabel)",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="skip the off-the-clock distance and rank_needed diagnostics",
    )
    parser.add_argument(
        "--history-rows",
        type=int,
        default=400,
        help="largest number of outer iterations stored per run in histories.csv",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/sdp"))
    parser.add_argument("--name", default=None, help="study name (default: derived)")
    parser.add_argument("--force", action="store_true", help="recompute the study")
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="compute outside a Slurm allocation (workstations, never login nodes)",
    )
    return parser


def study_name(args) -> str:
    if args.name:
        return args.name
    if args.family == "maxcut":
        return f"maxcut_n{args.n}"
    dual = args.n - args.rank if args.dual_rank is None else args.dual_rank
    return f"planted_n{args.n}_m{args.m}_r{args.rank}_s{dual}"


def make_instance(args, seed: int) -> SDP:
    if args.family == "maxcut":
        return maxcut(args.n, seed=seed, density=args.density)
    return planted(
        args.n,
        args.m,
        args.rank,
        dual_rank=args.dual_rank,
        seed=seed,
        spread=args.spread,
    )


def configurations(args) -> list[dict]:
    """The methods of one study, in the order of the module docstring."""
    first = args.block_sizes[0]
    configs = []
    if "exact" in args.groups:
        configs.append(
            dict(label="exact", f_solver="exact", projection="exact", sigma=0.0)
        )
    for sigma in args.sigmas:
        if "cg_only" in args.groups:
            configs.append(
                dict(
                    label=f"cg_only_sigma{sigma:g}",
                    f_solver="cg",
                    projection="exact",
                    sigma=sigma,
                )
            )
        if "lanczos_only" in args.groups:
            configs.append(
                dict(
                    label=f"lanczos_only_sigma{sigma:g}_b{first}",
                    f_solver="exact",
                    projection="lanczos",
                    sigma=sigma,
                    block_size=first,
                )
            )
        if "inexact" in args.groups:
            for block in args.block_sizes:
                configs.append(
                    dict(
                        label=f"inexact_sigma{sigma:g}_b{block}",
                        f_solver="cg",
                        projection="lanczos",
                        sigma=sigma,
                        block_size=block,
                    )
                )
        for group, cholesky in (("one_sided", False), ("certified", True)):
            if group in args.groups:
                for block in args.block_sizes:
                    configs.append(
                        dict(
                            label=f"{group}_sigma{sigma:g}_b{block}",
                            f_solver="cg",
                            projection="one_sided",
                            sigma=sigma,
                            block_size=block,
                            cholesky=cholesky,
                        )
                    )
    if "near_exact" in args.groups:
        configs.append(
            dict(
                label=f"near_exact_b{first}",
                f_solver="cg",
                projection="lanczos",
                sigma=0.0,
                block_size=first,
                exact_tolerance=args.near_exact_tolerance,
            )
        )
    return configs


def _thin(count: int, limit: int) -> list[int]:
    """At most ``limit`` iteration indices: the first quarter of the budget in
    full, the rest evenly spaced, and always the last."""
    if count <= limit:
        return list(range(count))
    head = limit // 4
    spaced = np.linspace(head, count - 1, limit - head).round().astype(int)
    return sorted(set(range(head)) | set(spaced.tolist()))


def run_rows(args, name: str, sdp: SDP, config: dict, truth: dict) -> tuple[dict, list]:
    """Run one method on one instance; its ``runs.csv`` row and history rows."""
    sigma = config["sigma"]
    block = config.get("block_size", 0)
    tolerance = config.get("exact_tolerance")
    trace: Trace = proximal_admm(
        sdp,
        c=args.c,
        rho=args.rho,
        sigma=sigma,
        delta=args.delta,
        stride=args.stride,
        f_solver=config["f_solver"],
        projection=config["projection"],
        block_size=max(block, 1),
        f_share=args.f_share,
        drop=args.drop,
        exact_tolerance=tolerance,
        cholesky=config.get("cholesky", False),
        guard_factor=args.guard_factor,
        max_outer=args.max_outer,
        time_limit=args.time_limit,
        seed=sdp.seed,
        diagnostics=not args.no_diagnostics,
    )
    one_sided = config["projection"] == "one_sided"
    # The diagnostic is computed for the one-sided projection only.
    violation = [v for v in trace.violation if not math.isnan(v)]
    key = dict(
        study=name,
        family=sdp.family,
        n=sdp.n,
        m=sdp.m,
        rank=sdp.rank,
        dual_rank=sdp.dual_rank,
        seed=sdp.seed,
        label=config["label"],
    )
    row = dict(
        key,
        f_solver=config["f_solver"],
        projection=config["projection"],
        sigma=sigma,
        block_size=block,
        stride=args.stride,
        c=args.c,
        rho=args.rho,
        delta=args.delta,
        exact_tolerance="" if tolerance is None else tolerance,
        cholesky=config.get("cholesky", False),
        guard_factor=args.guard_factor if one_sided else "",
        converged=trace.converged,
        stopped_by=trace.stopped_by,
        outer_iterations=trace.outer_iterations,
        seconds=trace.seconds,
        cg_steps=sum(trace.cg_steps),
        forward_applications=trace.forward,
        adjoint_applications=trace.adjoint,
        lanczos_matvecs=sum(trace.matvecs),
        lanczos_steps=sum(trace.lanczos_steps),
        lanczos_restarts=sum(trace.restarts),
        eigendecompositions=trace.eigendecompositions,
        checks=sum(trace.checks),
        guard_rejections=sum(trace.guard_rejections),
        certificates=sum(trace.certificates),
        certificate_failures=sum(trace.certificate_failures),
        violations=sum(v > VIOLATION for v in violation) if violation else "",
        worst_violation=max(violation) if violation else "",
        projection_seconds=trace.projection_seconds,
        certificate_seconds=trace.certificate_seconds,
        subspace_dim_median=statistics.median(trace.subspace_dim),
        subspace_dim_max=max(trace.subspace_dim),
        subspace_dim_last=trace.subspace_dim[-1],
        rank_needed_median=statistics.median(trace.rank_needed),
        rank_needed_max=max(trace.rank_needed),
        rank_needed_last=trace.rank_needed[-1],
        final_Q=trace.Q[-1],
        final_E=trace.E[-1],
        final_P=trace.P[-1],
        objective_star=sdp.objective_star,
        objective_reference=truth.get("value", float("nan")),
        reference_status=truth.get("status", "not requested"),
        hostname=platform.node(),
        cpu_model=cpu_model(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", "local"),
        python=platform.python_version(),
        numpy=np.__version__,
    )
    row.update(certificate(sdp, trace, args.c, args.rho, args.delta))
    row.update(theory_checks(trace, sigma, args.delta))
    history = [
        dict(
            key,
            k=k,
            Q=trace.Q[k],
            E=trace.E[k],
            P=trace.P[k],
            e_f=trace.e_f[k],
            e_g=trace.e_g[k],
            cg_steps=trace.cg_steps[k],
            lanczos_matvecs=trace.matvecs[k],
            lanczos_steps=trace.lanczos_steps[k],
            lanczos_restarts=trace.restarts[k],
            checks=trace.checks[k],
            subspace_dim=trace.subspace_dim[k],
            rank_positive=trace.rank_positive[k],
            rank_negative=trace.rank_negative[k],
            rank_needed=trace.rank_needed[k],
            guard_rejections=trace.guard_rejections[k],
            certificates=trace.certificates[k],
            certificate_failures=trace.certificate_failures[k],
            violation=trace.violation[k],
            accepted_by=trace.accepted_by[k],
            elapsed=trace.elapsed[k],
            dist=trace.dist[k],
            objective=trace.objective[k],
        )
        for k in _thin(len(trace.Q), args.history_rows)
    ]
    return row, history


def _append(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if not (os.environ.get("SLURM_JOB_ID") or args.allow_local):
        raise SystemExit(
            "these experiments compute inside a Slurm allocation; pass "
            "--allow-local on a workstation (never on a login node)"
        )
    name = study_name(args)
    directory = args.results_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    runs_path = directory / "runs.csv"
    histories_path = directory / "histories.csv"
    settings_path = directory / "settings.json"
    settings = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key not in ("force", "allow_local", "results_dir", "seeds")
    }
    # A study is one version of the code: a resumed study with other sources
    # is refused like one with other settings.
    settings["source_sha256"] = source_hash()
    if args.force:
        for path in (runs_path, histories_path, settings_path):
            path.unlink(missing_ok=True)
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise SystemExit(
                f"{directory} holds a study with other settings or sources; "
                "use another --name or --force"
            )
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    done = set()
    if runs_path.exists():
        with runs_path.open(newline="") as handle:
            done = {(int(row["seed"]), row["label"]) for row in csv.DictReader(handle)}
    configs = configurations(args)
    for seed in args.seeds:
        pending = [c for c in configs if (seed, c["label"]) not in done]
        if not pending:
            continue
        sdp = make_instance(args, seed)
        truth = reference(sdp) if args.reference else {}
        for config in pending:
            row, history = run_rows(args, name, sdp, config, truth)
            # The history first: a run counts as done once it is in runs.csv.
            _append(histories_path, HISTORY_FIELDS, history)
            _append(runs_path, RUN_FIELDS, [row])
            print(
                f"{name} seed={seed} {config['label']}: {row['stopped_by']} after "
                f"{row['outer_iterations']} outer, {row['cg_steps']} CG steps, "
                f"{row['lanczos_matvecs']} matvecs, {row['seconds']:.1f} s",
                flush=True,
            )
    return directory


if __name__ == "__main__":
    main()
