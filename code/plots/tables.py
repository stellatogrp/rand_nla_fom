"""LaTeX tables for the paper, generated from the CSVs under ``results/``.

Each table is a ``tabular`` fragment written to ``results/tables/`` so that
the paper can ``\\input`` it.  Numbers reach the manuscript only through these
files: nothing here is typed by hand, and a table whose study has not run is
skipped, exactly like a figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from experiments._paths import HISTORIES, SUMMARY, study_dir

from ._data import MissingResults, load, missing, select
from ._style import SOLVER_LABELS
from .lasso import HYBRID_ORDER

TABLES = "tables"
TARGET_ACCURACY = 1e-7
LP_TARGET_ACCURACY = 1e-4
CONDITION_TEXT = {
    "moderately-conditioned": r"$\kappa(A)=10$",
    "mildly-ill-conditioned": r"$\kappa(A)=20$",
    "ill-conditioned": r"$\kappa(A)=100$",
}
DESIGN_TEXT = {
    "independent": "independent",
    "correlated": r"correlated",
}
SOLVER_ORDER = (
    "schur_cg",
    "schur_block_kaczmarz",
    "schur_randomized_kaczmarz",
    "coupled_gmres",
)


def sci(value: float, digits: int = 2) -> str:
    """Typeset a float in scientific notation, or a dash when missing."""
    if value is None or not np.isfinite(value):
        return "--"
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def integer(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{int(round(value))}"


def decimal(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}g}"


def setting_label(label: str) -> str:
    """Typeset a study's run label, which names parameters in plain text."""
    text = label.replace("_", r"\_")
    # ``gamma`` goes first so that it never matches inside ``\gamma_x``.
    for plain, typeset in (
        ("gamma", r"$\gamma$"),
        ("gx", r"$\gamma_x$"),
        ("gl", r"$\gamma_\lambda$"),
        ("sigma", r"$\sigma$"),
        ("theta", r"$\theta$"),
    ):
        text = text.replace(plain, typeset)
    return text


def mean_sd(mean: float, sd: float) -> str:
    if mean is None or not np.isfinite(mean):
        return "--"
    return rf"${mean:.0f} \pm {sd:.0f}$"


def tabular(
    columns: str,
    header: Sequence[str],
    body: Sequence[Sequence[str]],
    header_groups: Sequence[tuple[str, int]] | None = None,
) -> str:
    """Render a booktabs table body as LaTeX.

    ``header_groups`` optionally adds a first header line of spanning labels,
    each given with the number of columns it covers.
    """
    lines = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule"]
    if header_groups:
        cells, rules, start = [], [], 1
        for label, span in header_groups:
            cells.append(rf"\multicolumn{{{span}}}{{c}}{{{label}}}")
            if label:
                rules.append(rf"\cmidrule(lr){{{start}-{start + span - 1}}}")
            start += span
        lines.append(" & ".join(cells) + r" \\")
        lines.extend(rules)
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    lines.extend(" & ".join(row) + r" \\" for row in body)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def solver_sweep_table(results_dir: Path | None, study: str = "solver_sweep") -> str:
    """Mean and standard deviation of the work of each inner solver."""
    summary = load(study, SUMMARY, results_dir)
    body = []
    for kind in ("equality QP", "LP", "inequality QP"):
        for condition in ("moderately-conditioned", "mildly-ill-conditioned"):
            first = True
            for solver in SOLVER_ORDER:
                rows = select(
                    summary, problem=kind, matrix=condition, inner_solver=solver
                )
                if not rows:
                    continue
                row = rows[0]
                body.append(
                    [
                        kind if first else "",
                        CONDITION_TEXT.get(condition, condition) if first else "",
                        SOLVER_LABELS.get(solver, solver),
                        f"{int(row['reached'])}/{int(row['instances'])}",
                        mean_sd(row["outer_mean"], row["outer_sd"]),
                        mean_sd(row["inner_mean"], row["inner_sd"]),
                        decimal(row["time_mean"], 3),
                    ]
                )
                first = False
    return tabular(
        "llllrrr",
        ["family", "matrix", "inner solver", "reached", "outer", "inner", "time (s)"],
        body,
    )


LASSO_STAGE_GROUPS = {
    # stage -> (heading of the grouping column, how to name a variant)
    "solver": ("design", lambda row: DESIGN_TEXT.get(row["variant"], row["variant"])),
    "size": ("instance", lambda row: f"${int(row['m'])} \\times {int(row['p'])}$"),
    "hybrid": ("penalty", lambda row: row["variant"]),
}


def lasso_work_table(results_dir: Path | None, stage: str) -> str:
    """Work of each inner solver on every variant of one LASSO stage."""
    results = select(load("lasso", results_dir=results_dir), stage=stage)
    if not results:
        raise MissingResults(f"lasso results carry no {stage} stage")
    heading, name = LASSO_STAGE_GROUPS[stage]
    variants: dict[str, None] = {}
    for row in results:
        variants.setdefault(row["variant"], None)
    if stage == "hybrid":
        # Same display order as the figures.
        ordered = [name for name in HYBRID_ORDER if name in variants]
        variants = dict.fromkeys(ordered + [v for v in variants if v not in ordered])
    body = []
    for variant in variants:
        first = True
        for solver in SOLVER_ORDER:
            rows = select(results, variant=variant, inner_solver=solver)
            if not rows:
                continue
            row = rows[0]
            body.append(
                [
                    name(row) if first else "",
                    SOLVER_LABELS.get(solver, solver),
                    integer(row["outer_iterations"]),
                    integer(row["inner_iterations"]),
                    f"{row['mean_inner_iterations']:.1f}",
                    integer(row["max_inner_iterations"]),
                    decimal(row["runtime_seconds"], 3),
                ]
            )
            first = False
    return tabular(
        "llrrrrr",
        [
            heading,
            "inner solver",
            "outer",
            "inner",
            r"mean $N_k$",
            r"max $N_k$",
            "time (s)",
        ],
        body,
    )


def lasso_hybrid_instance_table(results_dir: Path | None) -> str:
    """Coefficients, optimum, and solution structure of the hybrid penalties."""
    results = select(load("lasso", results_dir=results_dir), stage="hybrid")
    if not results:
        raise MissingResults("lasso results carry no hybrid stage")
    body = []
    seen: dict[str, None] = {}
    first_rows = {}
    for row in results:
        first_rows.setdefault(row["variant"], row)
    ordered = [name for name in HYBRID_ORDER if name in first_rows]
    ordered += [name for name in first_rows if name not in ordered]
    for row in (first_rows[name] for name in ordered):
        if row["variant"] in seen:
            continue
        seen[row["variant"]] = None
        body.append(
            [
                row["variant"],
                decimal(row["c1"], 3),
                decimal(row["c2"], 3),
                decimal(row["c3"], 3),
                decimal(row["c4"], 3),
                decimal(row["optimum"], 6),
                "--"
                if row["solution_nonzeros"] < 0
                else integer(row["solution_nonzeros"]),
                "--" if row["zero_residuals"] < 0 else integer(row["zero_residuals"]),
            ]
        )
    return tabular(
        "lrrrrrrr",
        [
            "penalty",
            "$c_1$",
            "$c_2$",
            "$c_3$",
            "$c_4$",
            "$F^\\star$",
            r"$\|x^\star\|_0$",
            "zero residuals",
        ],
        body,
    )


def lasso_instance_table(results_dir: Path | None) -> str:
    """Dimensions, penalty, and condition numbers of the LASSO instances."""
    results = load("lasso", results_dir=results_dir)
    body = []
    for design in ("independent", "correlated"):
        rows = select(results, design=design)
        if not rows:
            continue
        row = rows[0]
        body.append(
            [
                DESIGN_TEXT.get(design, design),
                f"{int(row['m'])}",
                f"{int(row['p'])}",
                decimal(row["penalty"], 3),
                decimal(row["design_condition"], 3),
                decimal(row["matrix_condition"], 3),
            ]
        )
    return tabular(
        "lrrrrr",
        ["design", "$m$", "$p$", r"$\tau$", r"$\kappa(D)$", r"$\kappa(A)$"],
        body,
    )


def work_at_target(
    histories: list[dict], keys: dict, target: float
) -> tuple[float, float]:
    """Return the outer and cumulative inner iterations at the first iterate
    whose accuracy is at most ``target``, or ``nan`` when none is."""
    trace = sorted(select(histories, **keys), key=lambda row: row["iteration"])
    for row in trace:
        if row["accuracy"] <= target:
            return float(row["iteration"]), float(row["cumulative_inner_iterations"])
    return np.nan, np.nan


def transfer_table(
    study: str, results_dir: Path | None, target: float = TARGET_ACCURACY
) -> str:
    """Large-instance runs of a tuning study: settings and the work they cost.

    Runs stop at their own tolerance, at different accuracies, so the table
    also reports the work at the first iterate meeting a common target.
    """
    results = select(load(study, results_dir=results_dir), problem="large")
    if not results:
        raise MissingResults(f"{study} results carry no large instance")
    histories = load(study, HISTORIES, results_dir)
    body = []
    for row in results:
        keys = dict(problem="large", stage=row["stage"], label=row["label"])
        if "adaptive" in row:
            keys["adaptive"] = row["adaptive"]
        outer, inner = work_at_target(histories, keys, target)
        body.append(
            [
                setting_label(row["label"]),
                decimal(row["gamma_x"], 3),
                decimal(row["gamma_lambda"], 3),
                decimal(row["sigma"], 2),
                decimal(row["theta"], 2),
                integer(outer),
                integer(inner),
                integer(row["outer_iterations"]),
                integer(row["inner_iterations"]),
                sci(row["final_merit"]),
            ]
        )
    return tabular(
        "lrrrrrrrrr",
        [
            "setting",
            r"$\gamma_x$",
            r"$\gamma_\lambda$",
            r"$\sigma$",
            r"$\theta$",
            "outer",
            "inner",
            "outer",
            "inner",
            "final merit",
        ],
        body,
        header_groups=[
            ("", 5),
            (f"to accuracy {sci(target, 1)}", 2),
            ("at stop", 3),
        ],
    )


def standard_table(results_dir: Path | None) -> str:
    """Every run of the small QP and LP comparisons."""
    results = load("standard", results_dir=results_dir)
    body = []
    for problem in ("QP", "LP"):
        first = True
        for row in select(results, problem=problem):
            body.append(
                [
                    problem if first else "",
                    row["group"],
                    setting_label(row["label"]),
                    integer(row["outer_iterations"]),
                    integer(row["inner_iterations"]),
                    sci(row["final_objective_error"]),
                    sci(row["final_feasibility"]),
                ]
            )
            first = False
    return tabular(
        "lllrrrr",
        [
            "problem",
            "group",
            "setting",
            "outer",
            "inner",
            "objective error",
            r"$\|Ax-b\|$",
        ],
        body,
    )


def make(results_dir: Path | None = None) -> list[Path]:
    output = study_dir(TABLES, results_dir)
    builders = {
        "standard.tex": lambda: standard_table(results_dir),
        # The large LP never reaches 1e-7 within its iteration budget.
        "lp_tuning_transfer.tex": lambda: transfer_table(
            "lp_tuning", results_dir, target=LP_TARGET_ACCURACY
        ),
        "inequality_qp_transfer.tex": lambda: transfer_table(
            "inequality_qp_tuning", results_dir
        ),
        "solver_sweep.tex": lambda: solver_sweep_table(results_dir),
        "solver_sweep_randomized.tex": lambda: solver_sweep_table(
            results_dir, "solver_sweep_randomized_spectrum"
        ),
        "lasso_instances.tex": lambda: lasso_instance_table(results_dir),
        "lasso_solvers.tex": lambda: lasso_work_table(results_dir, "solver"),
        "lasso_sizes.tex": lambda: lasso_work_table(results_dir, "size"),
        "lasso_hybrid.tex": lambda: lasso_work_table(results_dir, "hybrid"),
        "lasso_hybrid_instances.tex": lambda: lasso_hybrid_instance_table(results_dir),
    }
    written = []
    for name, build in builders.items():
        try:
            content = build()
        except MissingResults:
            continue
        written.append(write(output / name, content))
    if not written:
        raise MissingResults(missing(output))
    return written
