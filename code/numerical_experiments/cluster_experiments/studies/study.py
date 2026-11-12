"""Driver shared by the three cluster studies, in the format of the original.

A port of ``numerical_experiments.study``.  One task is one problem instance:
draw it, compute its ground truth, the setup quantities, and the paper's
recommended parameters, then run every configuration (the exact ADMM or
plain gradient descent baseline, the near-exact baseline, the ``sigma`` sweep
with ``sigma*`` of Proposition 5.1, and the stride sweep, ``b`` here and ``s``
in the paper, with ``b*`` of Lemma 4.2) and reduce each run to one row of
``results.csv`` and one row per outer iteration of ``histories.csv``.
Instance-level quantities go to ``parameters.csv``; ``curves.csv`` and
``summary.csv`` are aggregated from them exactly as in the original.  Every
inexact run has a hard stop (``--max-total-inner`` inner steps or
``--time-limit`` seconds; the near-exact baseline has the step cap only),
since the theory guarantees termination only for ``sigma < 1`` and the sweep
includes ``sigma = 1``; ``results.csv`` records why each run ended in
``stopped_by``.

Stages (``--stage``):

``all``
    The original ``run_study``: calibrate once in the parent, run the
    instances on a local process pool, collect, and aggregate.  For a
    workstation (``--allow-local``) or a single Slurm allocation.
``calibrate``
    Set up the first instance and time one inner iteration and one check on
    its own inner solver (``calibration.json``), as the original's parent
    process does once per study, fix the study's configuration list
    (``configurations.csv``), and set up every other instance
    (``parts/<seed>/setup/``).
``run``
    One configuration (``--configuration``) of every instance, or of one
    (``--task-index``, seed ``--seed`` plus the index): a Slurm array task, so
    that a configuration that stalls holds up no other.  The runs of an
    instance share its stored setup (computed by whichever task needs it first
    if calibrating did not).
``task``
    Every configuration of one instance in one process, resuming after the
    last finished configuration.
``collect``
    Merge the finished runs into the three tables and aggregate; with
    ``--allow-partial`` the unfinished ones are listed in ``missing.csv``
    instead of holding up the others.

An instance's setup and runs live in ``<study>/parts/<seed>/`` until collected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from releases.regression_v9.experiments import theory
from releases.regression_v9.experiments.matrices import freeze_spec
from releases.regression_v9.experiments.runtime import cpu_model

from . import ENGINE
from ._io import read_rows, write_rows
from .aggregate import aggregate_study
from .algorithms import (
    MAX_TOTAL_INNER,
    TIME_LIMIT,
    Diagnostics,
    Initialization,
    Trace,
    accurate_resolvent,
    exact_admm,
    gradient_descent,
    inexact_admm,
    inexact_ppm,
)
from .inner import (
    CALIBRATION_REPETITIONS,
    RESTART,
    InnerFactory,
    calibrate,
    contraction,
    finite,
    recommendations,
    sigma_star_by_b,
)
from .problems import FAMILIES, GroundTruth, Instance, exact_residual

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = (
    Path(os.environ.get("RAND_NLA_FOM_RESULTS", ROOT / "results")) / "studies"
)

STUDIES = {
    "admm_qp": "inexact versus exact ADMM on the periodic-grid equality QP",
    "admm_lasso": "inexact versus exact ADMM on the split LASSO",
    "ppm_logistic": (
        "inexact proximal point on ridge logistic regression versus gradient descent"
    ),
}
FAMILY = {"admm_qp": "qp", "admm_lasso": "lasso", "ppm_logistic": "logistic"}
# The cluster's inner solvers for each study.  Block Kaczmarz needs the
# explicit sparse QP Hessian; logistic regression has gradient descent only.
INNER = {
    "admm_qp": (
        "gmres",
        "gd",
        "agd",
        "cg",
        "nystrom_pcg",
        "block_kaczmarz",
        "cyclic_kaczmarz",
    ),
    "admm_lasso": ("gmres", "gd", "agd", "cg", "nystrom_pcg"),
    "ppm_logistic": ("gd",),
}
SIGMAS = (0.01, 0.05, 0.2, 0.5, 0.9, 1.0)
BS = (1, 2, 5, 10, 20)
DELTA = 1e-9
NEAR_EXACT_TOLERANCE = 1e-12
# Distance-based theorem checks are evaluated only while ||gamma z_k|| is at
# least this large, so that the accuracy of the ground truth cannot bias them.
CHECK_THRESHOLD = 1e-3
RESULTS = "results.csv"
HISTORIES = "histories.csv"
PARAMETERS = "parameters.csv"
SETTINGS = "settings.json"
CALIBRATION = "calibration.json"
CONFIGURATIONS = "configurations.csv"
MISSING = "missing.csv"
PARTS = "parts"
SETUP = "setup"
STAGES = ("all", "calibrate", "task", "run", "collect")
CONFIGURATION_FIELDS = (
    "method",
    "sweep",
    "sigma",
    "b",
    "sigma_is_optimal",
    "b_is_optimal",
    "b_exact_is_optimal",
    "in_grid",
)
# Keys that control how a study runs rather than what it computes; they may
# differ between the stages and tasks of one study.
RUN_CONTROL = {
    "stage",
    "task_index",
    "configuration",
    "workers",
    "force",
    "no_aggregate",
    "max_restarts",
    "allow_local",
    "allow_partial",
    "instances",
    "results_dir",
    "data_dir",
    "dataset_spec",
    "calibration_repetitions",
    "c_in",
    "c_test",
    "calibration_seconds",
    "crash_seed",
}

nan = float("nan")


class _Parser(argparse.ArgumentParser):
    """Reads ``@file`` arguments, several per line; ``#`` starts a comment."""

    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        return arg_line.split("#", 1)[0].split()


def build_parser(study: str) -> argparse.ArgumentParser:
    family = FAMILY[study]
    admm = FAMILIES[family] == "admm"
    parser = _Parser(
        description=STUDIES[study],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--instances", type=int, default=100, help="problem instances")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the first instance"
    )
    if family == "qp":
        parser.add_argument(
            "--side",
            type=int,
            default=40,
            help="grid side: n = side^3 variables and n/4 equality constraints",
        )
        parser.add_argument(
            "--kappa", type=float, default=1e4, help="condition number of Q"
        )
    else:
        parser.add_argument(
            "--dataset-spec",
            type=Path,
            default=ROOT / "datasets" / "random.json",
            help="JSON mapping of matrix sources (experiments.matrices)",
        )
        parser.add_argument(
            "--dataset", default="random_normal", help="source in --dataset-spec"
        )
        parser.add_argument(
            "--orientation", choices=["native", "transpose"], default="native"
        )
        parser.add_argument(
            "--rows", type=int, help="override the rows of a random_sparse source"
        )
        parser.add_argument(
            "--cols", type=int, help="override the columns of a random_sparse source"
        )
        parser.add_argument("--data-dir", type=Path, default=Path("data"))
        if family == "lasso":
            parser.add_argument(
                "--lambda-ratio",
                type=float,
                default=0.01,
                help="lambda = ratio * ||A'b||_inf",
            )
    if admm:
        parser.add_argument(
            "--ratio",
            type=float,
            default=10.0,
            help="L / c: the penalty is c = L / ratio",
        )
    else:
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
    parser.add_argument(
        "--bs",
        type=int,
        nargs="+",
        default=list(BS),
        help="stride grid (s in the paper)",
    )
    parser.add_argument(
        "--sigma-sweep-b", type=int, default=1, help="stride b used in the sigma sweep"
    )
    parser.add_argument(
        "--b-sweep-sigma",
        type=float,
        default=0.1,
        help="sigma used in the stride sweep",
    )
    parser.add_argument(
        "--near-exact-tolerance",
        type=float,
        default=NEAR_EXACT_TOLERANCE,
        help="absolute inner residual tolerance of the near-exact baseline",
    )
    parser.add_argument(
        "--near-exact-time-limit",
        type=float,
        help="wall-clock stop of the near-exact baseline (default: none, as in "
        "the original); a safety cap for runs that would outlast a Slurm task",
    )
    parser.add_argument("--max-outer", type=int, default=50_000)
    parser.add_argument(
        "--max-total-inner",
        type=int,
        default=MAX_TOTAL_INNER,
        help="hard stop: total inner steps of one inexact run",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=TIME_LIMIT,
        help="hard stop: wall-clock seconds of one inexact run",
    )
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
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=10,
        help="times the worker pool is rebuilt after a worker dies",
    )
    # The cluster's inner solvers and starts.
    parser.add_argument(
        "--inner",
        choices=INNER[study],
        default="gmres" if admm else "gd",
        help="inner solver of the subproblem",
    )
    parser.add_argument(
        "--rank", type=int, default=64, help="sketch rank of nystrom_pcg"
    )
    parser.add_argument(
        "--block-size", type=int, default=32, help="block rows of the Kaczmarz solvers"
    )
    parser.add_argument(
        "--gd-factor",
        type=float,
        default=1.0,
        help="inner gradient descent step, in units of the minimax step",
    )
    parser.add_argument(
        "--restart", type=int, default=RESTART, help="Krylov dimension of a GMRES cycle"
    )
    parser.add_argument(
        "--inner-initialization",
        choices=["warm", "fixed", "gaussian"],
        default="warm",
        help="inner start of each outer iteration (warm: the original studies)",
    )
    parser.add_argument("--gaussian-scale", type=float, default=1.0)
    parser.add_argument("--fixed-initial-value", type=float, default=0.0)
    parser.add_argument(
        "--initialization-seed",
        type=int,
        help="seed of the gaussian starts (default: instance seed + 200003)",
    )
    # The ground truth.
    parser.add_argument(
        "--truth-solver",
        choices=["iterative", "clarabel"],
        default="iterative",
        help="reference solver of the regression studies: FISTA with polish or "
        "Newton-CG, or CVXPY with Clarabel as in the original (small designs); "
        "the QP always uses CG on its KKT system",
    )
    parser.add_argument(
        "--truth-tolerance",
        type=float,
        default=1e-12,
        help="relative stationarity of the iterative reference solves",
    )
    parser.add_argument("--truth-time-limit", type=float, default=3600.0)
    parser.add_argument("--truth-max-iterations", type=int, default=100_000)
    if admm:
        parser.add_argument(
            "--exact-residuals",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="record ||R(w_k)|| (one exact x-update per outer iteration)",
        )
        parser.add_argument(
            "--exact-direct-max",
            type=int,
            default=5_000,
            help="largest Gram matrix factored for the exact LASSO x-update",
        )
    parser.add_argument(
        "--calibration-repetitions",
        type=int,
        default=CALIBRATION_REPETITIONS,
        help="timings of each of c_in and c_test in the study-level calibration",
    )
    # Stages.
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--task-index",
        type=int,
        help="instance of --stage task or run (seed --seed plus the index)",
    )
    parser.add_argument(
        "--configuration",
        type=int,
        help="configuration of --stage run, in the order of configurations.csv",
    )
    parser.add_argument("--name", help="study directory (default: from the settings)")
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="compute outside a Slurm allocation (workstations, never login nodes)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="collect even if some instances are unfinished",
    )
    return parser


def default_name(settings: dict) -> str:
    parts = [settings["study"]]
    if settings["family"] == "qp":
        parts.append(f"side{settings['side']}")
    else:
        parts.append(str(settings["dataset"]))
        if settings["orientation"] != "native":
            parts.append(settings["orientation"])
    inner = settings["inner"]
    parts.append(inner)
    if inner == "nystrom_pcg":
        parts.append(f"rank{settings['rank']}")
    if "kaczmarz" in inner:
        parts.append(f"block{settings['block_size']}")
    if settings["inner_initialization"] != "warm":
        parts.append(settings["inner_initialization"])
    return "_".join(parts)


def settings_from(args: argparse.Namespace, study: str, *, pin: bool = True) -> dict:
    """The settings of a study; ``pin=False`` leaves local data files unread.

    A local matrix source is pinned by the SHA256 of its file, which needs the
    file; collecting results needs only the tables, so it does not pin.
    """
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    family = FAMILY[study]
    settings.update(study=study, family=family, algorithm=FAMILIES[family])
    if family != "qp":
        catalog = json.loads(Path(args.dataset_spec).read_text())
        if args.dataset not in catalog:
            raise SystemExit(f"{args.dataset!r} is not in {args.dataset_spec}")
        spec = dict(catalog[args.dataset])
        for key in ("rows", "cols"):
            if getattr(args, key) is not None:
                if spec.get("kind") != "random_sparse":
                    raise SystemExit(f"--{key} applies to random_sparse sources only")
                spec[key] = int(getattr(args, key))
        if pin:
            # Pin the selected source only (its SHA256 for local files), not
            # the whole catalog, which can hold several large dense designs.
            spec = freeze_spec(spec, Path(args.data_dir))
        settings["matrix_source"] = spec
    if args.sigma_sweep_b < 1 or min(args.bs) < 1:
        raise SystemExit("strides must be positive")
    settings["name"] = args.name or default_name(settings)
    settings["engine"] = ENGINE
    return settings


def _identity(settings: dict) -> dict:
    """The settings that determine the computed numbers."""
    kept = {k: v for k, v in settings.items() if k not in RUN_CONTROL}
    return json.loads(json.dumps(kept, sort_keys=True))


def study_directory(settings: dict) -> Path:
    return Path(settings["results_dir"]) / settings["name"]


def _prepare_directory(settings: dict, *, create: bool = True) -> Path:
    """Create the study directory, or check that it holds the same study.

    With ``create=False`` (collecting) the study must exist, and an unpinned
    matrix source only has to agree with the pinned one on the keys it has.
    """
    directory = study_directory(settings)
    path = directory / SETTINGS
    record = dict(identity=_identity(settings), engine=ENGINE)
    if not create:
        if not path.exists():
            raise SystemExit(f"{directory} holds no study ({SETTINGS} is missing)")
        stored = json.loads(path.read_text())["identity"]
        source = record["identity"].get("matrix_source")
        pinned = stored.get("matrix_source")
        if isinstance(source, dict) and isinstance(pinned, dict):
            if all(pinned.get(key) == value for key, value in source.items()):
                record["identity"]["matrix_source"] = pinned
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(f".{SETTINGS}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        try:
            # Concurrent tasks may race to create it; exactly one link wins.
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    stored = json.loads(path.read_text())
    if stored["identity"] != record["identity"]:
        changed = sorted(
            key
            for key in set(stored["identity"]) | set(record["identity"])
            if stored["identity"].get(key) != record["identity"].get(key)
        )
        raise SystemExit(
            f"{directory} holds a study with other settings ({', '.join(changed)}); "
            "choose another --name or rerun with --stage all --force"
        )
    return directory


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_table(path: Path, rows: list[dict]) -> None:
    """``write_rows`` to a temporary file, then an atomic rename."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_rows(temporary, rows)
    os.replace(temporary, path)


# --------------------------------------------------------------------------
# One instance.


def _configurations(
    settings: dict, sigma_star: float, b_best: int | None, b_exact: int | None
) -> list[dict]:
    """The runs of one instance, in a fixed order.

    The classical baseline is exact ADMM (``admm_*``) or plain gradient
    descent (``ppm_logistic``).  ``b_best`` is the recommendation of Lemma 4.2
    and ``b_exact`` that of Remark 4.4; each is flagged on its grid row or
    added as its own run.  An inner solver without verified contraction
    constants has neither, nor ``sigma*``, and those runs are left out.
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

    baseline = "exact" if settings["algorithm"] == "admm" else "gd"
    configs = [
        config(baseline, "reference", nan, 0, in_grid=True),
        config("near_exact", "reference", nan, 1, in_grid=True),
    ]
    for sigma in settings["sigmas"]:
        configs.append(config("inexact", "sigma", float(sigma), b_sigma, in_grid=True))
    if math.isfinite(sigma_star):
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
    for b in sorted({b for b in (b_best, b_exact) if b is not None} - set(grid)):
        configs.append(config("inexact", "b", sigma_b, b, in_grid=False))
    return configs


def _time_limit(method: str, settings: dict) -> float | None:
    """The wall-clock stop of one run: none for the near-exact baseline.

    The time limit guards the inexact runs, whose termination the theory
    guarantees only for ``sigma < 1``; the near-exact baseline always
    terminates, and capping its time would understate its cost.  The cap on
    total inner steps still applies to it, and ``--near-exact-time-limit``
    can add one where a run would outlast its Slurm task (Kaczmarz, whose
    near-exact baseline checks after every block update).
    """
    if method == "near_exact":
        return settings.get("near_exact_time_limit")
    return settings.get("time_limit", TIME_LIMIT)


def _optional_int(value) -> int | None:
    return None if value is None or not finite(float(value)) else int(value)


@dataclass
class Setup:
    """One instance with everything its configurations share.

    The original studies run every configuration of an instance in one process
    after one setup.  Here the setup (``L``, the ground truth, the paper's
    parameters, and the configuration list) is stored once per instance in
    ``parts/<seed>/setup/``, so that the configurations can also run as
    separate Slurm tasks and still share it exactly.
    """

    seed: int
    instance: Instance
    parameters: dict
    truth: GroundTruth
    smoothness: float
    c: float
    gamma: float
    exact: object
    factory: InnerFactory | None
    diagnostics: Diagnostics
    initialization: Initialization
    configurations: list
    # The calibration made on this instance (``calibrate``'s costs, its seconds,
    # and the seconds of its timed measurements), or None.
    calibration: dict | None = None


def _signature(config: dict) -> tuple:
    """A configuration as comparable values (``sigma`` exactly, ``nan`` as text)."""
    sigma = float(config["sigma"])
    return (
        str(config["method"]),
        str(config["sweep"]),
        "nan" if math.isnan(sigma) else repr(sigma),
        int(config["b"]),
    )


def _write_configurations(path: Path, configurations: list[dict]) -> None:
    rows = [{key: c[key] for key in CONFIGURATION_FIELDS} for c in configurations]
    _write_table(path, rows)


def _read_configurations(path: Path) -> list[dict] | None:
    return read_rows(path) if path.exists() else None


def _load_setup(part: Path) -> tuple[dict, GroundTruth] | None:
    """The stored setup of an instance: its parameters and its ground truth."""
    setup = part / SETUP
    if not (setup / "truth.npy").exists():
        return None
    parameters = read_rows(setup / PARAMETERS)[0]
    truth = GroundTruth(
        x=np.load(setup / "truth.npy"),
        value=float(parameters["truth_value"]),
        status=str(parameters["truth_status"]),
        tolerance=float(parameters["truth_tolerance"]),
        seconds=float(parameters["truth_seconds"]),
        stationarity=float(parameters["truth_stationarity"]),
        feasibility=float(parameters["truth_feasibility"]),
        planted_error=float(parameters["truth_planted_error"]),
        solver=str(parameters["truth_solver"]),
    )
    return parameters, truth


def _store_setup(
    part: Path, parameters: dict, truth: GroundTruth, configurations: list[dict]
) -> bool:
    """Store an instance's setup unless another task has; ``True`` if this did.

    The files are written to a private directory that is renamed into place in
    one step, so no task reads part of a setup, and concurrent tasks of one
    instance all end up with whichever setup was stored first.
    """
    target = part / SETUP
    if target.exists():
        return False
    part.mkdir(parents=True, exist_ok=True)
    temporary = part / f".{SETUP}.{os.getpid()}.{time.time_ns()}"
    temporary.mkdir()
    _write_table(temporary / PARAMETERS, [parameters])
    np.save(temporary / "truth.npy", truth.x)
    _write_configurations(temporary / CONFIGURATIONS, configurations)
    try:
        os.rename(temporary, target)
    except OSError:
        shutil.rmtree(temporary, ignore_errors=True)
        return False
    return True


def prepare_instance(
    study: str, seed: int, settings: dict, part: Path | None = None
) -> Setup:
    """Draw one instance and compute, or load, what its configurations share.

    In order: the smoothness constant by Lanczos, the ground truth, the exact
    ``x``-update and the inner solver's setup (ADMM), the inner-contraction
    constants, and the paper's parameters, timed as in the original.  With a
    ``part`` directory the setup is stored there, or read from there when a
    task of the same instance stored it before.
    """
    family, algorithm = settings["family"], settings["algorithm"]
    delta = float(settings["delta"])
    ratio = float(settings["ratio"])

    started = time.perf_counter()
    instance = Instance(family, seed, settings)
    generation_seconds = time.perf_counter() - started

    stored = _load_setup(part) if part is not None else None
    calibrated = None
    if stored is not None:
        parameters, truth = stored
        smoothness = float(parameters["smoothness"])
    else:
        parameters = None
        # Setup shared by the methods: the smoothness constant by Lanczos.  The
        # LASSO reference takes its step from it, so it comes first here.
        started = time.perf_counter()
        smoothness = instance.smoothness()
        smoothness_seconds = time.perf_counter() - started
        truth = instance.ground_truth(settings, smoothness=smoothness)
    instance.use_smoothness(smoothness)

    exact = factory = None
    # Neither projection needs a factorization (the original factors AA').
    projector_seconds = factor_seconds = inner_setup_seconds = 0.0
    if algorithm == "admm":
        c = smoothness / ratio
        gamma = 1.0
        effective_ratio = smoothness / c
        started = time.perf_counter()
        exact = instance.exact_solver(c, int(settings.get("exact_direct_max", 5_000)))
        factor_seconds = time.perf_counter() - started
        factory = InnerFactory(instance, c, settings)
        inner_setup_seconds = factory.setup_seconds
    else:
        c = nan
        gamma = ratio / smoothness
        effective_ratio = gamma * smoothness
    alpha, c0, kind = contraction(instance, c, effective_ratio, settings)

    def fixed_point(x):
        """``w*``: the solution itself (PPM), or ``x* + grad f(x*)/c`` (ADMM)."""
        return x + instance.gradient(x) / c if algorithm == "admm" else x

    if parameters is None:
        dist0 = float(np.linalg.norm(fixed_point(truth.x)))
        b_sigma = int(settings["sigma_sweep_b"])
        sigma_b = float(settings["b_sweep_sigma"])
        # The paper's parameter recommendations, timed as a practical user
        # would pay for them: sigma* needs only alpha and C_0; b* needs the
        # ratio of the check cost to the inner-iteration cost, which is
        # measured.  Without the study's calibration in ``settings`` (the
        # first instance, when the study calibrates), the costs are measured
        # here, on this instance's own inner solver: nothing is built for it.
        if settings.get("c_in") is None:
            repetitions = settings.get("calibration_repetitions")
            started = time.perf_counter()
            c_in, c_test, measured = calibrate(
                instance,
                c,
                gamma,
                factory,
                repetitions=int(repetitions or CALIBRATION_REPETITIONS),
            )
            calibration_seconds = time.perf_counter() - started
            calibrated = dict(
                c_in=c_in,
                c_test=c_test,
                calibration_seconds=calibration_seconds,
                measurement_seconds=measured,
                repetitions=int(repetitions or CALIBRATION_REPETITIONS),
            )
        else:
            c_in, c_test = float(settings["c_in"]), float(settings["c_test"])
            calibration_seconds = float(settings["calibration_seconds"])
        started = time.perf_counter()
        recommended = recommendations(alpha, c0, kind, settings, c_in, c_test)
        parameter_seconds = time.perf_counter() - started
        by_b = sigma_star_by_b(alpha, c0, kind, settings["bs"])

        # Diagnostics for the bounds: the exact fixed-point residual at w_0 = 0.
        started = time.perf_counter()
        w0 = np.zeros(instance.n)
        if algorithm == "admm":
            r0_norm = exact_residual(instance, exact, c, w0)
        else:
            _, r0_norm = accurate_resolvent(instance, gamma, w0)
        r0_seconds = time.perf_counter() - started
        k_plus = theory.k_delta_plus(sigma_b, dist0, gamma, delta)
        b_star_oracle = (
            theory.b_star(
                c_test,
                c_in,
                theory.n_prime(k_plus, sigma_b, c0, alpha, 0.0, r0_norm, gamma * delta),
                k_plus,
                alpha,
            )
            if kind != "none"
            else nan
        )
        inner = settings["inner"]
        parameters = dict(
            study=settings["name"],
            seed=seed,
            n=instance.n,
            m=instance.m,
            ratio=effective_ratio,
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
            sigma_star=recommended["sigma_star"],
            sigma_star_expected=recommended["sigma_star_expected"],
            sigma_upper_bound=recommended["sigma_upper_bound"],
            **{f"sigma_star_b{b}": value for b, value in by_b.items()},
            calibration_seconds=calibration_seconds,
            c_in_seconds=c_in,
            c_test_seconds=c_test,
            b_star=recommended["b_star"],
            b_best=_nan_if_none(recommended["b_best"]),
            b_star_exact=_nan_if_none(recommended["b_star_exact"]),
            b_star_oracle=b_star_oracle,
            r0_norm=r0_norm,
            r0_seconds=r0_seconds,
            dist0=dist0,
            k_delta_plus_log10=math.log10(k_plus) if k_plus > 0 else nan,
            # Not in the original tables: what identifies a cluster study.
            family=family,
            algorithm=algorithm,
            inner=inner if algorithm == "admm" else "gd",
            rank=settings["rank"] if inner == "nystrom_pcg" else nan,
            block_size=settings["block_size"] if "kaczmarz" in inner else nan,
            restart=settings["restart"] if inner == "gmres" else nan,
            gd_factor=settings["gd_factor"] if inner == "gd" else nan,
            inner_initialization=settings["inner_initialization"],
            contraction=kind,
            mu=float(instance.problem.mu),
            smoothness_certified=instance.smoothness_certified,
            inner_setup_seconds=inner_setup_seconds,
            exact_solver=getattr(exact, "kind", ""),
            truth_solver=truth.solver,
            source="periodic_grid" if family == "qp" else settings["dataset"],
            orientation=settings.get("orientation", "native"),
        )
        planned = _configurations(
            settings,
            parameters["sigma_star"],
            recommended["b_best"],
            recommended["b_star_exact"],
        )
        if part is not None and not _store_setup(part, parameters, truth, planned):
            # Another task of this instance stored its setup first: use that,
            # so that every run of the instance shares one set of parameters.
            parameters, truth = _load_setup(part)
            if float(parameters["smoothness"]) != smoothness:
                raise RuntimeError(f"two setups of instance {seed} disagree on L")
    configurations = _configurations(
        settings,
        float(parameters["sigma_star"]),
        _optional_int(parameters["b_best"]),
        _optional_int(parameters["b_star_exact"]),
    )
    diagnostics = Diagnostics(
        instance,
        truth.value,
        fixed_point(truth.x),
        c=c,
        exact=exact if settings.get("exact_residuals", True) else None,
    )
    return Setup(
        seed=seed,
        instance=instance,
        parameters=parameters,
        truth=truth,
        smoothness=smoothness,
        c=c,
        gamma=gamma,
        exact=exact,
        factory=factory,
        diagnostics=diagnostics,
        initialization=Initialization(settings, seed),
        configurations=configurations,
        calibration=calibrated,
    )


def run_configuration(setup: Setup, settings: dict, index: int):
    """Run configuration ``index`` of an instance; its ``results.csv`` row and
    its ``histories.csv`` rows."""
    config = setup.configurations[index]
    parameters = setup.parameters
    algorithm = settings["algorithm"]
    trace, error = None, ""
    try:
        trace = _run(
            config,
            setup.instance,
            settings,
            c=setup.c,
            gamma=setup.gamma,
            smoothness=setup.smoothness,
            exact=setup.exact,
            factory=setup.factory,
            diagnostics=setup.diagnostics,
            initialization=setup.initialization,
        )
    except (RuntimeError, ArithmeticError) as failure:
        error = f"{type(failure).__name__}: {failure}"
    identity = dict(study=settings["name"], seed=setup.seed, **config)
    if trace is None:
        return dict(**identity, converged=False, stopped_by="error", error=error), []
    row, rows = _reduce(algorithm, trace, identity, parameters)
    if algorithm == "admm":
        row["setup_seconds"] = parameters["projector_seconds"] + (
            parameters["factor_seconds"]
            if config["method"] == "exact"
            else parameters["smoothness_seconds"] + parameters["inner_setup_seconds"]
        )
    else:
        row["setup_seconds"] = parameters["smoothness_seconds"]
    row["parameter_seconds"] = parameters["parameter_seconds"]
    row["calibration_seconds"] = parameters["calibration_seconds"]
    row["truth_seconds"] = parameters["truth_seconds"]
    row["error"] = error
    return row, rows


def _run_files(part: Path, index: int) -> dict[str, Path]:
    runs = part / "runs"
    return dict(
        results=runs / f"{index:03d}.results.csv",
        histories=runs / f"{index:03d}.histories.csv",
        provenance=runs / f"{index:03d}.provenance.json",
    )


def _store_run(
    part: Path, index: int, row: dict, rows: list[dict], started: float
) -> None:
    """Store one run; its ``results`` file, written last, marks it finished."""
    files = _run_files(part, index)
    files["results"].parent.mkdir(parents=True, exist_ok=True)
    if rows:
        _write_table(files["histories"], rows)
    _write_json(files["provenance"], _provenance(started))
    _write_table(files["results"], [row])


def run_instance(task: tuple) -> dict:
    """Draw one instance, solve it, run every configuration, and reduce to rows.

    ``task`` is ``(study, seed, settings)`` or ``(study, seed, settings, part)``.
    With a ``part`` directory the setup and every finished configuration are
    stored there as they become available, and a rerun resumes from them.
    """
    study, seed, settings = task[:3]
    part = Path(task[3]) if len(task) > 3 and task[3] is not None else None
    # Test hook: the first attempt at ``crash_seed`` kills its worker outright,
    # as a segmentation fault would; a marker file makes the retry succeed.
    if settings.get("crash_seed") == seed:
        marker = Path(settings["results_dir"]) / f".crashed-{seed}"
        if not marker.exists():
            marker.touch()
            os._exit(1)
    setup = prepare_instance(study, seed, settings, part)
    results: list[dict] = []
    histories: list[dict] = []
    for index in range(len(setup.configurations)):
        if part is not None and _run_files(part, index)["results"].exists():
            continue
        started = time.perf_counter()
        row, rows = run_configuration(setup, settings, index)
        if part is not None:
            _store_run(part, index, row, rows, started)
        else:
            results.append(row)
            histories.extend(rows)
    if part is not None:
        return dict(seed=seed, runs=len(setup.configurations))
    return dict(
        parameters=setup.parameters, results=_homogeneous(results), histories=histories
    )


def _run(
    config,
    instance,
    settings,
    *,
    c,
    gamma,
    smoothness,
    exact,
    factory,
    diagnostics,
    initialization,
) -> Trace:
    method = config["method"]
    delta = float(settings["delta"])
    if method == "exact":
        return exact_admm(
            instance,
            c,
            exact,
            delta,
            max_outer=settings["max_outer"],
            diagnostics=diagnostics,
        )
    if method == "gd":
        return gradient_descent(
            instance,
            smoothness,
            delta,
            stepsize_ratio=settings.get("gd_step", 1.0),
            max_iterations=settings.get("gd_max_iterations", 1_000_000),
            diagnostics=diagnostics,
        )
    exact_tolerance = (
        settings["near_exact_tolerance"] if method == "near_exact" else None
    )
    if settings["algorithm"] == "admm":
        return inexact_admm(
            instance,
            c,
            factory.new(),
            config["sigma"],
            config["b"],
            delta,
            exact_tolerance=exact_tolerance,
            max_outer=settings["max_outer"],
            max_total_inner=settings.get("max_total_inner", MAX_TOTAL_INNER),
            time_limit=_time_limit(method, settings),
            diagnostics=diagnostics,
            initialization=initialization,
        )
    return inexact_ppm(
        instance,
        gamma,
        config["sigma"],
        config["b"],
        delta,
        exact_tolerance=exact_tolerance,
        max_outer=settings["max_outer"],
        max_total_inner=settings.get("max_total_inner", MAX_TOTAL_INNER),
        time_limit=_time_limit(method, settings),
        gd_factor=float(settings.get("gd_factor", 1.0)),
        diagnostics=diagnostics,
        initialization=initialization,
    )


def _nan_if_none(value):
    return nan if value is None else value


def _reduce(algorithm: str, trace: Trace, identity: dict, parameters: dict):
    """Turn a trace into its ``results.csv`` row and ``histories.csv`` rows.

    The original ``_reduce`` on the metrics the trace already holds.  The
    bounds that need the inner-contraction constants are ``nan`` for a
    solver without them, and the ones that need ``||R(w_k)||`` are ``nan``
    when the exact residuals are not recorded.  For a contraction that holds
    only in expectation (block Kaczmarz) the deterministic bounds of Theorem
    4.3 (``bound_*``, ``per_step_*``, ``thm43_*``) are ``nan`` too: the theory
    does not claim them there, and Theorem 4.2 is the bound that applies.
    """
    admm = algorithm == "admm"
    method = identity["method"]
    sigma = identity["sigma"]
    b = max(1, int(identity["b"]))
    alpha, c0, delta = (
        float(parameters["alpha"]),
        float(parameters["c0"]),
        float(parameters["delta"]),
    )
    bounded = finite(alpha, c0)
    # The original's solvers contract deterministically; its tables predate
    # the column, so a missing one means the same.
    deterministic = (
        bounded and parameters.get("contraction", "deterministic") == "deterministic"
    )
    gamma = trace.gamma
    # The bounds see the terminal test through ||gamma z|| <= gamma delta.
    gamma_delta = gamma * delta
    k_delta = trace.k_delta
    count = len(trace.inner)
    # Gradient descent records a thinned history; every other method has one
    # entry per outer iteration.
    dense = trace.iteration is None
    iteration = np.arange(count) if dense else np.asarray(trace.iteration, dtype=int)
    z_norm = np.asarray(trace.z_norm, dtype=float)
    gamma_z = gamma * z_norm
    inner = np.asarray(trace.inner, dtype=float)
    checks = np.asarray(trace.checks, dtype=float)
    elapsed = np.asarray(trace.elapsed, dtype=float)
    e0 = np.asarray(trace.e0_norm, dtype=float)
    y_minus_w = np.asarray(trace.y_minus_w, dtype=float)
    accepted = np.asarray(trace.accepted_by)
    suboptimality = np.asarray(trace.suboptimality, dtype=float)
    suboptimality_z = np.asarray(trace.suboptimality_z, dtype=float)
    feasibility = np.asarray(trace.feasibility, dtype=float)
    dist = np.asarray(trace.dist, dtype=float)
    residual_exact = np.asarray(trace.residual_exact, dtype=float)
    exact_known = admm and bool(np.all(np.isfinite(residual_exact)))

    # Per-step bounds of Theorem 4.3 (deterministic-per-step).  The surrogate
    # replaces ||R(w_k)|| by its computable lower bound (1 - sigma)||y_k - w_k||
    # from Lemma 3.1, valid when the pair passed the relative test.
    bound_surrogate = np.full(count, nan)
    bound_exact = np.full(count, nan)
    relative = accepted == "relative"
    if method == "inexact" and deterministic:
        for k in range(count):
            lower = (1.0 - sigma) * y_minus_w[k] if relative[k] else 0.0
            beta_surrogate = theory.beta(sigma, c0, e0[k], lower, gamma_delta)
            bound_surrogate[k] = theory.per_step_bound(beta_surrogate, alpha, b)
            if exact_known:
                beta_exact = theory.beta(
                    sigma, c0, e0[k], residual_exact[k], gamma_delta
                )
                bound_exact[k] = theory.per_step_bound(beta_exact, alpha, b)

    sigma_eff = 0.0 if method == "exact" else sigma
    total_inner = float(inner.sum())
    sum_gamma_z_sq = float((gamma_z[:k_delta] ** 2).sum()) if dense else nan
    # A thinned record means one gradient step per iteration.
    per_outer = inner if dense else np.ones(1)
    row = dict(
        **identity,
        converged=trace.converged,
        stopped_by=trace.stopped_by,
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

    # Theorem 3.1 and Lemma 3.2 need the relative test (or exact pairs).  Both
    # are vacuous at sigma = 1, which the theorems do not cover: the bound is
    # infinite and the predicted decrease is zero, so the ratios are left nan.
    if method in ("exact", "inexact"):
        thm31_bound = (
            (1.0 + sigma_eff) / (1.0 - sigma_eff) * float(parameters["dist0"]) ** 2
            if sigma_eff < 1.0
            else math.inf
        )
        row["thm31_bound"] = thm31_bound
        row["thm31_ratio"] = (
            sum_gamma_z_sq / thm31_bound if 0.0 < thm31_bound < math.inf else nan
        )
        qualifying = np.arange(k_delta)[gamma_z[:k_delta] >= CHECK_THRESHOLD]
        if method == "inexact":
            qualifying = qualifying[relative[qualifying]]
        if qualifying.size and sigma_eff < 1.0:
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

    # Total inner-work bounds (Theorems 4.2 and 4.3) with the observed K_delta;
    # both are infinite at sigma = 1 and their ratios are then left nan.
    if method == "inexact" and bounded:
        r0_norm = float(parameters["r0_norm"])
        beta0 = theory.beta(sigma, c0, e0[0], r0_norm, gamma_delta)
        thm43 = theory.total_bound_deterministic(
            beta0, k_delta, sigma, c0, alpha, b, r0_norm, gamma_delta
        )
        thm42 = theory.total_bound_expected(
            beta0, k_delta, sigma, c0, alpha, b, r0_norm, gamma_delta
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
            thm43_ratio=total_inner / thm43 if math.isfinite(thm43) else nan,
            thm42_bound=thm42,
            thm42_ratio=total_inner / thm42 if math.isfinite(thm42) else nan,
            per_step_violations=int((inner > bound_surrogate).sum()),
            per_step_max_ratio=float(step_ratio.max()),
            per_step_violations_exact=(
                int((inner > bound_exact).sum()) if exact_known else nan
            ),
        )
        if not deterministic:
            row.update(
                thm43_bound=nan,
                thm43_ratio=nan,
                per_step_violations=nan,
                per_step_max_ratio=nan,
                per_step_violations_exact=nan,
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

    # Theorem 3.2: an empirical error-bound constant Lambda (exact residuals
    # only), stored as ``kappa_empirical``.
    qualifying = (
        np.arange(k_delta)[gamma_z[:k_delta] >= CHECK_THRESHOLD]
        if exact_known and method in ("exact", "inexact")
        else np.arange(0)
    )
    if qualifying.size:
        error_bound = float((dist[qualifying] / residual_exact[qualifying]).max())
        error_bound = max(error_bound, 1.0)
        rate_bound = 1.0 - (1.0 - sigma_eff) / (error_bound**2 * (1.0 + sigma_eff))
        contraction = dist[qualifying + 1] ** 2 / dist[qualifying] ** 2
        row.update(
            kappa_empirical=error_bound,
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


def _calibration(directory: Path) -> dict | None:
    path = directory / CALIBRATION
    return json.loads(path.read_text()) if path.exists() else None


def _store_calibration(directory: Path, study: str, settings: dict) -> dict:
    """Calibrate the study on its first instance's own setup and record it.

    The first instance is set up (and stored in ``parts/<seed>/setup``) with
    the costs measured on its instance and inner solver, as built for its
    runs, so nothing is built twice; the other instances read the record,
    so that the cost ratio behind ``b*`` is the same for every instance.
    ``calibration_seconds`` is the time of the calibration alone and
    ``measurement_seconds`` that of its timed operations.
    """
    seed = int(settings["seed"])
    own = {
        k: v
        for k, v in settings.items()
        if k not in ("c_in", "c_test", "calibration_seconds")
    }
    setup = prepare_instance(study, seed, own, directory / PARTS / str(seed))
    values = setup.calibration
    if values is None:
        # The first instance was set up before (a calibration cut short after
        # storing it): its parameters hold the costs it was set up with.
        parameters = setup.parameters
        values = dict(
            c_in=float(parameters["c_in_seconds"]),
            c_test=float(parameters["c_test_seconds"]),
            calibration_seconds=float(parameters["calibration_seconds"]),
            measurement_seconds=nan,
            repetitions=None,
        )
    record = dict(**values, seed=seed, **_provenance(None))
    _write_json(directory / CALIBRATION, record)
    return record


def _use_calibration(settings: dict, record: dict | None) -> None:
    if record is not None:
        settings.update(
            c_in=record["c_in"],
            c_test=record["c_test"],
            calibration_seconds=record["calibration_seconds"],
        )


def _wanted(settings: dict) -> list[int]:
    return [int(settings["seed"]) + i for i in range(int(settings["instances"]))]


def _complete(directory: Path, seed: int) -> bool:
    """Every configuration of the instance has finished."""
    part = directory / PARTS / str(seed)
    configurations = _read_configurations(part / SETUP / CONFIGURATIONS)
    return configurations is not None and all(
        _run_files(part, index)["results"].exists()
        for index in range(len(configurations))
    )


def _plan(directory: Path, study: str, settings: dict) -> list[dict]:
    """Set up the first instance and fix the study's configuration list.

    ``configurations.csv`` lists the runs of every instance: they share the
    calibration and, on these problems, the inner-contraction constants, so
    they share ``sigma*`` and ``s*``.  Each separately run configuration checks
    its instance's list against it, so a manifest index means one run.
    """
    path = directory / CONFIGURATIONS
    planned = _read_configurations(path)
    if planned is not None:
        return planned
    seed = int(settings["seed"])
    setup = prepare_instance(study, seed, settings, directory / PARTS / str(seed))
    _write_configurations(path, setup.configurations)
    return _read_configurations(path)


def _announce(settings: dict, record: dict) -> None:
    print(
        f"[{settings['name']}] calibration: inner iteration "
        f"{record['c_in'] * 1e6:.1f} us, check {record['c_test'] * 1e6:.1f} us "
        f"({record['calibration_seconds']:.2f}s)",
        flush=True,
    )


def run_study(study: str, args: argparse.Namespace) -> Path:
    """The original ``run_study``: calibrate, run the instances on a pool, collect."""
    settings = settings_from(args, study)
    directory = study_directory(settings)
    if args.force and directory.exists():
        for name in (
            RESULTS,
            HISTORIES,
            PARAMETERS,
            SETTINGS,
            CALIBRATION,
            CONFIGURATIONS,
            MISSING,
        ):
            (directory / name).unlink(missing_ok=True)
        shutil.rmtree(directory / PARTS, ignore_errors=True)
    _prepare_directory(settings)
    _verify_engine()
    record = _calibration(directory) or _store_calibration(directory, study, settings)
    _use_calibration(settings, record)
    _announce(settings, record)
    _plan(directory, study, settings)
    wanted = _wanted(settings)

    def pending() -> list[int]:
        return [seed for seed in wanted if not _complete(directory, seed)]

    seeds = pending()
    print(
        f"[{settings['name']}] {len(wanted) - len(seeds)} instances already done, "
        f"{len(seeds)} to run on {args.workers} workers"
    )
    started = time.perf_counter()
    finished = 0
    total = len(seeds)
    restarts = 0
    # A worker that dies (a segmentation fault in a spawned interpreter has been
    # seen under load) breaks the pool and loses the instances it held; the
    # pool is then rebuilt on the instances not yet finished, instead of
    # waiting forever as multiprocessing.Pool does.
    while seeds:
        tasks = [
            (study, seed, settings, directory / PARTS / str(seed)) for seed in seeds
        ]
        try:
            with ProcessPoolExecutor(
                args.workers, mp_context=get_context("spawn")
            ) as pool:
                futures = [pool.submit(run_instance, task) for task in tasks]
                for future in as_completed(futures):
                    outcome = future.result()
                    finished += 1
                    elapsed = time.perf_counter() - started
                    print(
                        f"[{settings['name']}] instance {outcome['seed']} done "
                        f"({finished}/{total}, {elapsed:.0f}s elapsed)",
                        flush=True,
                    )
        except BrokenProcessPool:
            restarts += 1
            if restarts > args.max_restarts:
                raise SystemExit(
                    f"[{settings['name']}] a worker died {restarts} times; "
                    "rerun to resume"
                ) from None
            print(
                f"[{settings['name']}] a worker died; restarting the pool on the "
                f"remaining instances (restart {restarts}/{args.max_restarts})",
                flush=True,
            )
        seeds = pending()
    collect(directory, wanted, allow_partial=False)
    if not args.no_aggregate:
        aggregate_study(directory)
    return directory


def run_calibration(study: str, args: argparse.Namespace) -> Path:
    """Calibrate once, set up the first instance, and fix the configuration list."""
    settings = settings_from(args, study)
    directory = _prepare_directory(settings)
    record = _calibration(directory)
    if record is not None and args.force:
        if any((directory / PARTS).glob("*/runs/*.results.csv")):
            raise SystemExit(
                f"{directory} has runs made with the stored calibration; start a "
                "new study (--name) or rerun everything with --stage all --force"
            )
        (directory / CONFIGURATIONS).unlink(missing_ok=True)
        shutil.rmtree(directory / PARTS, ignore_errors=True)
        record = None
    if record is None:
        _verify_engine()
        record = _store_calibration(directory, study, settings)
    else:
        print(f"[{settings['name']}] calibration already stored; --force redoes it")
    _use_calibration(settings, record)
    _announce(settings, record)
    planned = _plan(directory, study, settings)
    print(f"[{settings['name']}] {len(planned)} configurations per instance")
    # Set up every instance here, once, so that the configurations, which run
    # as separate tasks, load their setups instead of computing them side by
    # side (a regression setup includes a long reference solve).
    for seed in _wanted(settings):
        part = directory / PARTS / str(seed)
        if (part / SETUP).exists():
            continue
        started = time.perf_counter()
        setup = prepare_instance(study, seed, settings, part)
        if [_signature(c) for c in setup.configurations] != [
            _signature(c) for c in planned
        ]:
            raise SystemExit(
                f"instance {seed} has other configurations than {CONFIGURATIONS}; "
                "its parameters differ from the first instance's"
            )
        print(
            f"[{settings['name']}] instance {seed} set up "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
    return directory


def run_task(study: str, args: argparse.Namespace) -> Path:
    """Every configuration of one instance, in one process (resumable)."""
    settings = settings_from(args, study)
    if args.task_index is None or not 0 <= args.task_index < args.instances:
        raise SystemExit("--stage task needs 0 <= --task-index < --instances")
    directory = _prepare_directory(settings)
    seed = int(settings["seed"]) + int(args.task_index)
    part = directory / PARTS / str(seed)
    if _complete(directory, seed) and not args.force:
        print(f"[{settings['name']}] instance {seed} already complete")
        return directory
    if args.force:
        shutil.rmtree(part, ignore_errors=True)
    _verify_engine()
    # Without a stored calibration the task calibrates on its own instance and
    # records calibration_seconds as nan, as the original does in a worker.
    _use_calibration(settings, _calibration(directory))
    started = time.perf_counter()
    run_instance((study, seed, settings, part))
    print(
        f"[{settings['name']}] instance {seed} done "
        f"({time.perf_counter() - started:.0f}s)",
        flush=True,
    )
    return directory


def run_run(study: str, args: argparse.Namespace) -> Path:
    """One configuration, for one instance or for every instance: a Slurm task.

    ``--configuration`` picks the run in the order of ``configurations.csv``,
    which ``--stage calibrate`` fixes; an index beyond the list is a no-op, so
    a manifest can reserve room for the recommended strides before they are
    known.  With ``--task-index`` it runs that instance only; without, every
    instance in turn, each run stored as it finishes and finished runs
    skipped, so a slow or stopped run delays only this configuration's.
    """
    settings = settings_from(args, study)
    if args.task_index is not None and not 0 <= args.task_index < args.instances:
        raise SystemExit("--stage run needs 0 <= --task-index < --instances")
    if args.configuration is None or args.configuration < 0:
        raise SystemExit("--stage run needs --configuration >= 0")
    directory = _prepare_directory(settings)
    record = _calibration(directory)
    planned = _read_configurations(directory / CONFIGURATIONS)
    if record is None or planned is None:
        raise SystemExit(f"{directory} is not calibrated; run --stage calibrate")
    index = int(args.configuration)
    if index >= len(planned):
        print(f"[{settings['name']}] no configuration {index}: nothing to run")
        return directory
    seeds = (
        _wanted(settings)
        if args.task_index is None
        else [int(settings["seed"]) + int(args.task_index)]
    )
    _verify_engine()
    _use_calibration(settings, record)
    signature = [_signature(c) for c in planned]
    for seed in seeds:
        part = directory / PARTS / str(seed)
        files = _run_files(part, index)
        if files["results"].exists() and not args.force:
            print(f"[{settings['name']}] instance {seed}, run {index} already done")
            continue
        for path in files.values():
            path.unlink(missing_ok=True)
        started = time.perf_counter()
        setup = prepare_instance(study, seed, settings, part)
        if [_signature(c) for c in setup.configurations] != signature:
            raise SystemExit(
                f"instance {seed} has other configurations than {CONFIGURATIONS}; "
                "its parameters differ from the first instance's"
            )
        row, rows = run_configuration(setup, settings, index)
        _store_run(part, index, row, rows, started)
        print(
            f"[{settings['name']}] instance {seed}, run {index} "
            f"({row['method']}, {row['sweep']}, sigma {row['sigma']}, b {row['b']}): "
            f"{row['stopped_by']} ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
    return directory


def run_collect(study: str, args: argparse.Namespace) -> Path:
    settings = settings_from(args, study, pin=False)
    directory = _prepare_directory(settings, create=False)
    collect(directory, _wanted(settings), allow_partial=args.allow_partial)
    if not args.no_aggregate:
        aggregate_study(directory)
    print(directory)
    return directory


def collect(directory: Path, seeds: list[int], *, allow_partial: bool) -> list[dict]:
    """Merge the finished runs, in seed and configuration order, into the tables.

    Rows are copied as text, so the collected tables hold exactly what the runs
    wrote; failed runs get the columns of the others, as ``nan``.  Unfinished
    runs are listed in ``missing.csv`` (and returned); with ``allow_partial``
    the tables hold the finished ones, so that a stalled run delays nothing
    else, and otherwise collecting stops until every run has finished.
    """
    planned = _read_configurations(directory / CONFIGURATIONS)
    parameters, results, histories, missing = [], [], [], []
    for seed in seeds:
        part = directory / PARTS / str(seed)
        stored = _read_configurations(part / SETUP / CONFIGURATIONS)
        configurations = stored if stored is not None else planned
        if stored is not None:
            parameters.extend(_read_text_rows(part / SETUP / PARAMETERS))
        if configurations is None:
            missing.append(dict(seed=seed, configuration="", reason="no setup"))
            continue
        for index, config in enumerate(configurations):
            files = _run_files(part, index)
            if files["results"].exists():
                results.extend(_read_text_rows(files["results"]))
                if files["histories"].exists():
                    histories.append(files["histories"])
                continue
            missing.append(
                dict(
                    seed=seed,
                    configuration=index,
                    method=config["method"],
                    sweep=config["sweep"],
                    sigma=config["sigma"],
                    b=config["b"],
                    reason="not finished" if stored is not None else "no setup",
                )
            )
    if missing:
        _write_text_rows(directory / MISSING, _homogeneous(missing, fill=""))
    else:
        (directory / MISSING).unlink(missing_ok=True)
    if missing and not allow_partial:
        raise SystemExit(
            f"{len(missing)} runs are unfinished (listed in {directory / MISSING}); "
            "finish them or collect with --allow-partial"
        )
    if not results:
        raise SystemExit(f"{directory} has no finished runs")
    if missing:
        print(f"{len(missing)} unfinished runs are left out; see {MISSING}")
    _write_text_rows(directory / PARAMETERS, _homogeneous(parameters, fill="nan"))
    _write_text_rows(directory / RESULTS, _homogeneous(results, fill="nan"))
    _concatenate(histories, directory / HISTORIES)
    return missing


def _read_text_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_text_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _concatenate(paths: list[Path], target: Path) -> None:
    """Concatenate CSV files with one header, checking that every header agrees."""
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    header = None
    with temporary.open("w", newline="") as output:
        for path in paths:
            with path.open(newline="") as stream:
                first = stream.readline()
                if header is None:
                    header = first
                    output.write(first)
                elif first != header:
                    raise ValueError(f"{path} has columns {first!r}, not {header!r}")
                shutil.copyfileobj(stream, output)
    if header is None:
        temporary.unlink()
        target.unlink(missing_ok=True)
        return
    os.replace(temporary, target)


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        f"studies/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.py"))
    }


_ENGINE_SEAL: dict = {}


def _verify_engine() -> str:
    """Check the sealed engine release; mandatory inside a Slurm allocation."""
    if "archive" in _ENGINE_SEAL:
        return _ENGINE_SEAL["archive"]
    try:
        from validation_v7.releases import verify

        archive = verify(ROOT, ENGINE)["archive_sha256"]
    except Exception as error:  # recorded, or fatal under Slurm
        if os.environ.get("SLURM_JOB_ID"):
            raise SystemExit(
                f"engine release {ENGINE} failed verification: {error}"
            ) from error
        archive = f"unverified: {type(error).__name__}: {error}"
    _ENGINE_SEAL["archive"] = archive
    return archive


def _provenance(started: float | None) -> dict:
    record = dict(
        hostname=platform.node(),
        cpu_model=cpu_model(),
        python=platform.python_version(),
        packages={name: version(name) for name in ("numpy", "scipy", "cvxpy")},
        slurm_job_id=os.environ.get("SLURM_JOB_ID", "local"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        engine=ENGINE,
        engine_archive_sha256=_verify_engine(),
        studies_sha256=_source_hashes(),
        command=sys.argv,
    )
    if started is not None:
        record["instance_seconds"] = time.perf_counter() - started
    return record


def _g(value: float) -> str:
    """Eight significant digits: plenty for a curve, half the bytes of repr."""
    return f"{float(value):.8g}"


def _homogeneous(rows: list[dict], fill=nan) -> list[dict]:
    """Give failed runs the same columns as the successful ones.

    The columns of the widest row come first, in its order, so that a failed
    run does not move its few columns to the front.
    """
    if not rows:
        return rows
    fields = list(max(rows, key=len))
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    return [{name: row.get(name, fill) for name in fields} for row in rows]


def main(study: str) -> None:
    parser = build_parser(study)
    args = parser.parse_args()
    if args.stage != "collect" and not (
        os.environ.get("SLURM_JOB_ID") or args.allow_local
    ):
        parser.error(
            "these studies compute inside a Slurm allocation; pass --allow-local "
            "on a workstation (never on a login node)"
        )
    stage = dict(
        all=run_study,
        calibrate=run_calibration,
        task=run_task,
        run=run_run,
        collect=run_collect,
    )[args.stage]
    stage(study, args)
