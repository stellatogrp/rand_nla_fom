"""Driver shared by the two studies.

One task is one problem instance: draw it, solve it with CVXPY, compute the
setup quantities and the paper's recommended parameters, then run every
configuration (the exact ADMM or plain gradient descent baseline, the
near-exact baseline, the ``sigma`` sweep with ``sigma*`` of Proposition 5.1,
and the ``b`` sweep with ``b*`` of Lemma 4.2) and reduce each run to one
row of ``results.csv`` and one row per outer iteration of
``histories.csv``.  Instance-level quantities go to
``parameters.csv``.  Instances run on a process pool with one BLAS thread
each and are checkpointed as they finish, so an interrupted study resumes.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from multiprocessing import get_context
from pathlib import Path

# Workers are spawned and re-import this module, so the limits are in place
# before NumPy loads there as well.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402

from . import theory  # noqa: E402
from ._io import append_rows, read_rows  # noqa: E402
from .admm import (  # noqa: E402
    exact_admm,
    exact_residual,
    factor_subproblem,
    inexact_admm,
)
from .aggregate import aggregate_study  # noqa: E402
from .ppm import (  # noqa: E402
    Trace,
    accurate_resolvent,
    gradient_descent,
    inexact_ppm,
)
from .problems import AffineProjector, make_huber, make_qp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    Path(os.environ.get("RAND_NLA_FOM_RESULTS", REPO_ROOT / "results"))
    / "numerical_experiments"
)

STUDIES = {
    "admm_qp": "inexact versus exact ADMM on an equality-constrained convex QP",
    "ppm_huber": "inexact proximal point on Huber-loss fitting versus gradient descent",
}
SIGMAS = (0.01, 0.05, 0.2, 0.5, 0.9)
BS = (1, 2, 5, 10, 20)
DELTA = 1e-9
NEAR_EXACT_TOLERANCE = 1e-12
# Distance-based theorem checks are evaluated only while ||gamma z_k|| is at
# least this large, so that the accuracy of the CVXPY solution cannot bias them.
CHECK_THRESHOLD = 1e-3
RESULTS = "results.csv"
HISTORIES = "histories.csv"
PARAMETERS = "parameters.csv"

nan = float("nan")


def build_parser(study: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=STUDIES[study],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--instances", type=int, default=100, help="problem instances")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the first instance"
    )
    parser.add_argument("--n", type=int, default=1000, help="decision variables")
    if study == "admm_qp":
        parser.add_argument("--m", type=int, default=300, help="equality constraints")
        parser.add_argument(
            "--kappa", type=float, default=100.0, help="condition number of Q"
        )
        parser.add_argument(
            "--ratio",
            type=float,
            default=10.0,
            help="L / c: the penalty is c = L / ratio",
        )
    else:
        parser.add_argument("--m", type=int, default=1500, help="observations")
        parser.add_argument(
            "--threshold", type=float, default=0.5, help="Huber threshold M"
        )
        parser.add_argument(
            "--ratio",
            type=float,
            default=10.0,
            help="gamma L: the stepsize is gamma = ratio / L",
        )
        parser.add_argument(
            "--gd-step",
            type=float,
            default=1.0,
            help="stepsize of the gradient descent baseline, in units of 1/L",
        )
        parser.add_argument(
            "--gd-max-iterations",
            type=int,
            default=1_000_000,
            help="iteration cap of the gradient descent baseline",
        )
    parser.add_argument("--delta", type=float, default=DELTA, help="stopping tolerance")
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS))
    parser.add_argument("--bs", type=int, nargs="+", default=list(BS))
    parser.add_argument(
        "--sigma-sweep-b", type=int, default=1, help="b used in the sigma sweep"
    )
    parser.add_argument(
        "--b-sweep-sigma", type=float, default=0.1, help="sigma used in the b sweep"
    )
    parser.add_argument(
        "--near-exact-tolerance",
        type=float,
        default=NEAR_EXACT_TOLERANCE,
        help="absolute inner residual tolerance of the near-exact baseline",
    )
    parser.add_argument("--max-outer", type=int, default=50_000)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(12, os.cpu_count() or 1))
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--force", action="store_true", help="recompute finished instances"
    )
    parser.add_argument(
        "--no-aggregate", action="store_true", help="skip curves/summary"
    )
    return parser


def settings_from(args: argparse.Namespace, study: str) -> dict:
    settings = dict(vars(args))
    settings["study"] = study
    settings["results_dir"] = str(args.results_dir)
    return settings


# --------------------------------------------------------------------------
# One instance.


def _calibrate(
    study: str, problem, projector, repetitions: int = 100
) -> tuple[float, float]:
    """Median seconds of one inner iteration and of one check (Lemma 4.2)."""
    rng = np.random.default_rng(12345)
    v = rng.standard_normal(problem.n)
    w = rng.standard_normal(problem.n)
    inner, check = [], []
    for _ in range(repetitions + 3):
        started = time.perf_counter()
        g = problem.gradient(v)
        e = g + (v - w)
        inner.append(time.perf_counter() - started)
        started = time.perf_counter()
        if study == "admm_qp":
            z = projector(v - g)
            float(np.linalg.norm(v - z))
            float(np.linalg.norm(z + g - w))
        else:
            float(np.linalg.norm(g))
            float(np.linalg.norm(v - w))
        float(np.linalg.norm(e))
        check.append(time.perf_counter() - started)
    return float(np.median(inner[3:])), float(np.median(check[3:]))


def _configurations(
    study: str, settings: dict, sigma_star: float, b_best: int, b_exact: int
) -> list[dict]:
    """The runs of one instance, in a fixed order.

    The classical baseline is exact ADMM (``admm_qp``) or plain gradient
    descent (``ppm_huber``).  ``b_best`` is the recommendation of Lemma 4.2
    and ``b_exact`` that of Remark 4.4; each is flagged on its grid row or
    added as its own run.
    """
    b_sigma = int(settings["sigma_sweep_b"])
    sigma_b = float(settings["b_sweep_sigma"])
    grid = [int(b) for b in settings["bs"]]

    def config(method, sweep, sigma, b, *, in_grid, sigma_is_optimal=False):
        return dict(
            method=method,
            sweep=sweep,
            sigma=sigma,
            b=b,
            sigma_is_optimal=sigma_is_optimal,
            b_is_optimal=sweep == "b" and b == b_best,
            b_exact_is_optimal=sweep == "b" and b == b_exact,
            in_grid=in_grid,
        )

    configs: list[dict] = []
    if study == "admm_qp":
        configs.append(config("exact", "reference", nan, 0, in_grid=True))
    else:
        configs.append(config("gd", "reference", nan, 0, in_grid=True))
    configs.append(config("near_exact", "reference", nan, 1, in_grid=True))
    for sigma in settings["sigmas"]:
        configs.append(config("inexact", "sigma", float(sigma), b_sigma, in_grid=True))
    configs.append(
        config(
            "inexact",
            "sigma",
            sigma_star,
            b_sigma,
            in_grid=False,
            sigma_is_optimal=True,
        )
    )
    for b in grid:
        configs.append(config("inexact", "b", sigma_b, b, in_grid=True))
    for b in sorted({b_best, b_exact} - set(grid)):
        configs.append(config("inexact", "b", sigma_b, b, in_grid=False))
    return configs


def run_instance(task: tuple[str, int, dict]) -> dict:
    """Draw one instance, solve it, run every configuration, and reduce to rows."""
    study, seed, settings = task
    delta = float(settings["delta"])
    started = time.perf_counter()
    if study == "admm_qp":
        problem = make_qp(seed, settings["n"], settings["m"], settings["kappa"])
    else:
        problem = make_huber(seed, settings["n"], settings["m"], settings["threshold"])
    generation_seconds = time.perf_counter() - started

    truth = problem.ground_truth()

    # Setup shared by the inexact methods: the smoothness constant.
    started = time.perf_counter()
    smoothness = problem.smoothness()
    smoothness_seconds = time.perf_counter() - started

    projector = factor = None
    projector_seconds = factor_seconds = 0.0
    if study == "admm_qp":
        c = smoothness / settings["ratio"]
        gamma = 1.0
        ratio = smoothness / c
        started = time.perf_counter()
        projector = AffineProjector(problem.A, problem.b)
        projector_seconds = time.perf_counter() - started
        started = time.perf_counter()
        factor = factor_subproblem(problem, c)
        factor_seconds = time.perf_counter() - started
        w_star = truth.x + problem.gradient(truth.x) / c
    else:
        c = nan
        gamma = settings["ratio"] / smoothness
        ratio = gamma * smoothness
        w_star = truth.x
    alpha, c0 = theory.inner_constants(ratio)
    dist0 = float(np.linalg.norm(w_star))

    # The paper's parameter recommendations, timed as a practical user would
    # pay for them: sigma* needs only alpha and C_0; b* needs the ratio of the
    # check cost to the inner-iteration cost, which is measured.
    started = time.perf_counter()
    b_sigma = int(settings["sigma_sweep_b"])
    sigma_b = float(settings["b_sweep_sigma"])
    sigma_star = theory.optimal_sigma(alpha, c0, b_sigma)
    sigma_star_expected = theory.optimal_sigma(alpha, c0, b_sigma, deterministic=False)
    sigma_bound = theory.sigma_upper_bound(alpha, c0, b_sigma)
    if settings.get("c_in") is None:
        c_in, c_test = _calibrate(study, problem, projector)
        calibration_seconds = nan
    else:
        c_in, c_test = settings["c_in"], settings["c_test"]
        calibration_seconds = settings["calibration_seconds"]
    b_star = theory.b_star_asymptotic(c_test, c_in, sigma_b, c0, alpha)
    b_best = theory.best_integer_b(
        b_star, lambda b: theory.b_cost_per_outer(b, c_test, c_in, sigma_b, c0, alpha)
    )
    b_exact = theory.optimal_b_exact(c_test, c_in, sigma_b, c0, alpha)
    parameter_seconds = time.perf_counter() - started
    sigma_star_by_b = {
        int(b): theory.optimal_sigma(alpha, c0, int(b)) for b in settings["bs"]
    }

    # Diagnostics for the bounds: the exact fixed-point residual at w_0 = 0.
    started = time.perf_counter()
    w0 = np.zeros(problem.n)
    if study == "admm_qp":
        r0_norm = exact_residual(problem, c, factor, projector, w0)
    else:
        _, r0_norm = accurate_resolvent(problem, gamma, smoothness, w0)
    r0_seconds = time.perf_counter() - started
    k_plus = theory.k_delta_plus(sigma_b, dist0, gamma, delta)
    b_star_oracle = theory.b_star(
        c_test,
        c_in,
        theory.n_prime(k_plus, sigma_b, c0, alpha, 0.0, r0_norm, delta),
        k_plus,
        alpha,
    )

    parameters = dict(
        study=study,
        seed=seed,
        n=problem.n,
        m=problem.m,
        ratio=ratio,
        c=c,
        gamma=gamma,
        delta=delta,
        gd_step=settings.get("gd_step", nan),
        smoothness=smoothness,
        alpha=alpha,
        c0=c0,
        generation_seconds=generation_seconds,
        truth_seconds=truth.seconds,
        truth_status=truth.status,
        truth_tolerance=truth.tolerance,
        truth_value=truth.value,
        truth_stationarity=truth.stationarity,
        truth_feasibility=truth.feasibility,
        truth_planted_error=truth.planted_error,
        smoothness_seconds=smoothness_seconds,
        projector_seconds=projector_seconds,
        factor_seconds=factor_seconds,
        parameter_seconds=parameter_seconds,
        sigma_sweep_b=b_sigma,
        b_sweep_sigma=sigma_b,
        sigma_star=sigma_star,
        sigma_star_expected=sigma_star_expected,
        sigma_upper_bound=sigma_bound,
        **{f"sigma_star_b{b}": value for b, value in sigma_star_by_b.items()},
        calibration_seconds=calibration_seconds,
        c_in_seconds=c_in,
        c_test_seconds=c_test,
        b_star=b_star,
        b_best=b_best,
        b_star_exact=b_exact,
        b_star_oracle=b_star_oracle,
        r0_norm=r0_norm,
        r0_seconds=r0_seconds,
        dist0=dist0,
        k_delta_plus_log10=math.log10(k_plus) if k_plus > 0 else nan,
    )

    results: list[dict] = []
    histories: list[dict] = []
    for config in _configurations(study, settings, sigma_star, b_best, b_exact):
        trace, error = None, ""
        try:
            if config["method"] == "exact":
                trace = exact_admm(
                    problem,
                    c,
                    factor,
                    projector,
                    delta,
                    max_outer=settings["max_outer"],
                )
            elif config["method"] == "gd":
                trace = gradient_descent(
                    problem,
                    smoothness,
                    delta,
                    stepsize_ratio=settings.get("gd_step", 1.0),
                    max_iterations=settings.get("gd_max_iterations", 1_000_000),
                )
            elif study == "admm_qp":
                trace = inexact_admm(
                    problem,
                    c,
                    smoothness,
                    projector,
                    config["sigma"],
                    config["b"],
                    delta,
                    exact_tolerance=(
                        settings["near_exact_tolerance"]
                        if config["method"] == "near_exact"
                        else None
                    ),
                    max_outer=settings["max_outer"],
                )
            else:
                trace = inexact_ppm(
                    problem,
                    gamma,
                    smoothness,
                    config["sigma"],
                    config["b"],
                    delta,
                    exact_tolerance=(
                        settings["near_exact_tolerance"]
                        if config["method"] == "near_exact"
                        else None
                    ),
                    max_outer=settings["max_outer"],
                )
        except RuntimeError as failure:
            error = str(failure)
        identity = dict(study=study, seed=seed, **config)
        if trace is None:
            results.append(dict(**identity, converged=False, error=error))
            continue
        row, rows = _reduce(
            study,
            problem,
            truth,
            trace,
            identity,
            parameters,
            w_star,
            factor,
            projector,
        )
        if study == "admm_qp":
            row["setup_seconds"] = projector_seconds + (
                factor_seconds if config["method"] == "exact" else smoothness_seconds
            )
        else:
            row["setup_seconds"] = smoothness_seconds
        row["parameter_seconds"] = parameter_seconds
        row["calibration_seconds"] = calibration_seconds
        row["truth_seconds"] = truth.seconds
        row["error"] = error
        results.append(row)
        histories.extend(rows)
    return dict(parameters=parameters, results=results, histories=histories)


def _reduce(
    study,
    problem,
    truth,
    trace: Trace,
    identity: dict,
    parameters: dict,
    w_star,
    factor,
    projector,
):
    """Turn a trace into its ``results.csv`` row and ``histories.csv`` rows."""
    method = identity["method"]
    sigma = identity["sigma"]
    b = max(1, int(identity["b"]))
    alpha, c0, delta = parameters["alpha"], parameters["c0"], parameters["delta"]
    gamma = trace.gamma
    c = parameters["c"]
    k_delta = trace.k_delta
    count = len(trace.inner)
    # Gradient descent records a thinned history; every other method has one
    # entry per outer iteration.
    dense = trace.iteration is None
    iteration = np.arange(count) if dense else np.asarray(trace.iteration, dtype=int)
    iterates = np.asarray(trace.iterates)
    ws = np.asarray(trace.w)
    z_norm = np.asarray(trace.z_norm)
    gamma_z = gamma * z_norm
    inner = np.asarray(trace.inner, dtype=float)
    checks = np.asarray(trace.checks, dtype=float)
    elapsed = np.asarray(trace.elapsed)
    e0 = np.asarray(trace.e0_norm)
    y_minus_w = np.asarray(trace.y_minus_w)
    accepted = np.asarray(trace.accepted_by)

    suboptimality = np.array([problem.objective(x) for x in iterates]) - truth.value
    dist = np.linalg.norm(ws - w_star, axis=1)
    if study == "admm_qp":
        feasible = np.asarray(trace.feasible_iterates)
        suboptimality_z = (
            np.array([problem.objective(z) for z in feasible]) - truth.value
        )
        feasibility = np.array([problem.feasibility(x) for x in iterates])
        residual_exact = np.array(
            [exact_residual(problem, c, factor, projector, w) for w in ws]
        )
    else:
        suboptimality_z = np.full(count, nan)
        feasibility = np.full(count, nan)
        residual_exact = np.full(count, nan)

    # Per-step bounds of Theorem 4.3 (deterministic-per-step).  The surrogate
    # replaces ||R(w_k)|| by its computable lower bound (1 - sigma)||y_k - w_k||
    # from Lemma 3.1, valid when the pair passed the relative test.
    bound_surrogate = np.full(count, nan)
    bound_exact = np.full(count, nan)
    beta_surrogate = np.full(count, nan)
    beta_exact = np.full(count, nan)
    relative = accepted == "relative"
    if method == "inexact":
        for k in range(count):
            lower = (1.0 - sigma) * y_minus_w[k] if relative[k] else 0.0
            beta_surrogate[k] = theory.beta(sigma, c0, e0[k], lower, delta)
            bound_surrogate[k] = theory.per_step_bound(beta_surrogate[k], alpha, b)
            if study == "admm_qp":
                beta_exact[k] = theory.beta(sigma, c0, e0[k], residual_exact[k], delta)
                bound_exact[k] = theory.per_step_bound(beta_exact[k], alpha, b)

    sigma_eff = 0.0 if method == "exact" else sigma
    total_inner = float(inner.sum())
    sum_gamma_z_sq = float((gamma_z[:k_delta] ** 2).sum()) if dense else nan
    # A thinned record means one gradient step per iteration.
    per_outer = inner if dense else np.ones(1)
    row = dict(
        **identity,
        converged=trace.converged,
        k_delta=k_delta,
        outer_iterations=k_delta + 1,
        total_inner=total_inner,
        total_checks=float(checks.sum()),
        gradient_evaluations=trace.gradient_evaluations,
        max_inner_per_outer=float(per_outer.max()),
        mean_inner_per_outer=float(per_outer.mean()),
        runtime_seconds=trace.seconds,
        final_suboptimality=float(suboptimality[-1]),
        final_suboptimality_z=float(suboptimality_z[-1]),
        final_feasibility=float(feasibility[-1]),
        final_z_norm=float(z_norm[-1]),
        final_dist=float(dist[-1]),
        sum_gamma_z_sq=sum_gamma_z_sq,
    )

    # Theorem 3.1 and Lemma 3.2 need the relative test (or exact pairs).
    if method in ("exact", "inexact"):
        thm31_bound = (1.0 + sigma_eff) / (1.0 - sigma_eff) * parameters["dist0"] ** 2
        row["thm31_bound"] = thm31_bound
        row["thm31_ratio"] = sum_gamma_z_sq / thm31_bound if thm31_bound > 0 else nan
        qualifying = np.arange(k_delta)[gamma_z[:k_delta] >= CHECK_THRESHOLD]
        if method == "inexact":
            qualifying = qualifying[relative[qualifying]]
        if qualifying.size:
            decrease = dist[qualifying] ** 2 - dist[qualifying + 1] ** 2
            predicted = (1.0 - sigma_eff) / (1.0 + sigma_eff) * gamma_z[qualifying] ** 2
            ratios = decrease / predicted
            row["lemma32_min_ratio"] = float(ratios.min())
            row["lemma32_checked"] = int(qualifying.size)
        else:
            row["lemma32_min_ratio"] = nan
            row["lemma32_checked"] = 0
    else:
        row.update(
            thm31_bound=nan, thm31_ratio=nan, lemma32_min_ratio=nan, lemma32_checked=0
        )

    # Total inner-work bounds (Theorems 4.2 and 4.3) with the observed K_delta.
    if method == "inexact":
        beta0 = theory.beta(sigma, c0, e0[0], parameters["r0_norm"], delta)
        thm43 = theory.total_bound_deterministic(
            beta0, k_delta, sigma, c0, alpha, b, parameters["r0_norm"], delta
        )
        thm42 = theory.total_bound_expected(
            beta0, k_delta, sigma, c0, alpha, b, parameters["r0_norm"], delta
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            step_ratio = np.where(
                bound_surrogate > 0,
                inner / bound_surrogate,
                np.where(inner > 0, np.inf, 0.0),
            )
        row.update(
            beta0=beta0,
            thm43_bound=thm43,
            thm43_ratio=total_inner / thm43,
            thm42_bound=thm42,
            thm42_ratio=total_inner / thm42,
            per_step_violations=int((inner > bound_surrogate).sum()),
            per_step_max_ratio=float(step_ratio.max()),
            per_step_violations_exact=(
                int((inner > bound_exact).sum()) if study == "admm_qp" else nan
            ),
        )
    else:
        row.update(
            beta0=nan,
            thm43_bound=nan,
            thm43_ratio=nan,
            thm42_bound=nan,
            thm42_ratio=nan,
            per_step_violations=nan,
            per_step_max_ratio=nan,
            per_step_violations_exact=nan,
        )

    # Theorem 3.2: an empirical error-bound constant (exact residuals only).
    if study == "admm_qp" and method in ("exact", "inexact"):
        qualifying = np.arange(k_delta)[gamma_z[:k_delta] >= CHECK_THRESHOLD]
        if qualifying.size:
            kappa = float((dist[qualifying] / residual_exact[qualifying]).max())
            kappa = max(kappa, 1.0)
            rate_bound = 1.0 - (1.0 - sigma_eff) / (kappa**2 * (1.0 + sigma_eff))
            contraction = dist[qualifying + 1] ** 2 / dist[qualifying] ** 2
            row.update(
                kappa_empirical=kappa,
                rate_bound=rate_bound,
                rate_observed_max=float(contraction.max()),
                rate_observed_median=float(np.median(contraction)),
            )
        else:
            row.update(
                kappa_empirical=nan,
                rate_bound=nan,
                rate_observed_max=nan,
                rate_observed_median=nan,
            )
    else:
        row.update(
            kappa_empirical=nan,
            rate_bound=nan,
            rate_observed_max=nan,
            rate_observed_median=nan,
        )

    cumulative_inner = np.cumsum(inner)
    rows = [
        dict(
            seed=identity["seed"],
            method=method,
            sweep=identity["sweep"],
            sigma=sigma,
            b=identity["b"],
            iteration=int(iteration[k]),
            suboptimality=_g(suboptimality[k]),
            suboptimality_z=_g(suboptimality_z[k]),
            feasibility=_g(feasibility[k]),
            z_norm=_g(z_norm[k]),
            inner=int(inner[k]),
            cumulative_inner=int(cumulative_inner[k]),
            checks=int(checks[k]),
            elapsed=_g(elapsed[k]),
            dist=_g(dist[k]),
            accepted_by=str(accepted[k]),
            residual_exact=_g(residual_exact[k]),
            bound_surrogate=_g(bound_surrogate[k]),
            bound_exact=_g(bound_exact[k]),
        )
        for k in range(count)
    ]
    return row, rows


# --------------------------------------------------------------------------
# The study.


def calibrate_study(study: str, settings: dict) -> tuple[float, float, float]:
    """Time one inner iteration and one check on a representative instance.

    Done once in the parent process before the pool starts, on an otherwise
    idle machine with one BLAS thread, so that the cost ratio behind ``b*``
    is not distorted by the concurrent workers; returns the two costs and the
    seconds the calibration itself took.
    """
    started = time.perf_counter()
    if study == "admm_qp":
        problem = make_qp(
            settings["seed"], settings["n"], settings["m"], settings["kappa"]
        )
        projector = AffineProjector(problem.A, problem.b)
    else:
        problem = make_huber(
            settings["seed"], settings["n"], settings["m"], settings["threshold"]
        )
        projector = None
    c_in, c_test = _calibrate(study, problem, projector, repetitions=300)
    return c_in, c_test, time.perf_counter() - started


def run_study(study: str, args: argparse.Namespace) -> Path:
    settings = settings_from(args, study)
    c_in, c_test, calibration_seconds = calibrate_study(study, settings)
    settings.update(c_in=c_in, c_test=c_test, calibration_seconds=calibration_seconds)
    print(
        f"[{study}] calibration: inner iteration {c_in * 1e6:.1f} us, "
        f"check {c_test * 1e6:.1f} us ({calibration_seconds:.2f}s)"
    )
    directory = Path(args.results_dir) / study
    directory.mkdir(parents=True, exist_ok=True)
    results_path = directory / RESULTS
    histories_path = directory / HISTORIES
    parameters_path = directory / PARAMETERS
    if args.force:
        for path in (results_path, histories_path, parameters_path):
            path.unlink(missing_ok=True)
    done = {int(row["seed"]) for row in read_rows(parameters_path)}
    seeds = [args.seed + i for i in range(args.instances) if args.seed + i not in done]
    print(
        f"[{study}] {len(done)} instances already done, "
        f"{len(seeds)} to run on {args.workers} workers"
    )
    tasks = [(study, seed, settings) for seed in seeds]
    started = time.perf_counter()
    if tasks:
        context = get_context("spawn")
        with context.Pool(args.workers) as pool:
            for index, outcome in enumerate(
                pool.imap_unordered(run_instance, tasks), start=1
            ):
                append_rows(parameters_path, [outcome["parameters"]])
                append_rows(results_path, _homogeneous(outcome["results"]))
                append_rows(histories_path, outcome["histories"])
                elapsed = time.perf_counter() - started
                print(
                    f"[{study}] instance {outcome['parameters']['seed']} done "
                    f"({index}/{len(tasks)}, {elapsed:.0f}s elapsed)",
                    flush=True,
                )
    if not args.no_aggregate:
        aggregate_study(directory)
    return directory


def _g(value: float) -> str:
    """Eight significant digits: plenty for a curve, half the bytes of repr."""
    return f"{float(value):.8g}"


def _homogeneous(rows: list[dict]) -> list[dict]:
    """Give failed runs the same columns as the successful ones."""
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    return [{name: row.get(name, nan) for name in fields} for row in rows]
