"""Figures for the two studies, drawn from the CSV tables alone.

    python -m numerical_experiments.plots            # both studies
    python -m numerical_experiments.plots admm_qp    # one of them

Per study the figures are ``convergence_sigma`` and ``convergence_b``
(suboptimality and feasibility or stationarity over outer iterations and
over cumulative inner iterations: median, interquartile band, and sparse
10--90 percentile whiskers), ``work_sigma`` and ``work_b`` (outer
iterations, inner iterations against the bound of Theorem 4.3, checks, and
wall-clock), ``per_step`` (inner iterations per outer iteration against the
per-step bound of Theorem 4.3), ``walltime`` (setup, parameter
computation, and algorithm time per configuration with 10--90 percentile
whiskers, and the CVXPY ground truth for scale), ``walltime_sgm`` (the
same with shifted geometric means in place of medians), ``bstar_model``
(the cost model behind ``b*`` with and without the relaxation of Remark 4.4, next to
the measured time per outer iteration), and ``convergence_sigma_zoom`` and
``convergence_b_zoom`` (the
outer-iteration panels restricted to the early transient and to the final
approach, as median lines only, so that curves that overlap at full scale
can be told apart).  ``trajectory_2d`` draws the paths of the iterates on
the two-dimensional instance of :mod:`numerical_experiments.trajectory_2d`
over the contours of the loss, and ``trajectory_2d_exact`` singles out one
inexact run against the near-exact one, with the exact solution of each of
its subproblems.  The baseline of the PPM study, plain gradient descent,
has no outer iterations: it is drawn against inner work (one of its
iterations is one gradient step) and in the time figures only.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import style, theory
from ._io import read_columns, read_rows
from .aggregate import config_key

style.apply()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

LABELS = {
    "admm_qp": dict(
        exact="exact ADMM (Cholesky)",
        gd=None,
        near="near-exact ADMM (inner tol $10^{-12}$)",
        residual=r"feasibility $\|Ax_k-b\|$",
        inner="cumulative inner iterations (gradient steps)",
        checks="projections (checks)",
        title="Inexact ADMM, equality-constrained QP",
    ),
    "ppm_huber": dict(
        exact=None,
        gd="gradient descent, step ${step}/L$",
        near="near-exact PPM (inner tol $10^{-12}$)",
        residual=r"stationarity $\|\nabla f(x_k)\|$",
        inner="cumulative inner iterations (gradient steps)",
        checks="checks",
        title="Inexact PPM, Huber-loss fitting",
    ),
}
SUBOPT = r"suboptimality $|f(x_k)-f^\star|$"
SIGMA = r"\sigma"
BSTAR = r"b^\star"
BEXACT = r"b^\star_{\mathrm{ex}}"
SIGMASTAR = r"\sigma^\star"
nan = float("nan")
# Shift, in seconds, of the geometric mean in ``walltime_sgm`` (``--walltime-shift``).
WALLTIME_SHIFT = 1.0
TRAJECTORY_STUDY = "ppm_huber_2d"


@dataclass
class Series:
    key: tuple
    label: str
    color: str
    marker: str = "o"
    linestyle: str = "-"
    zorder: int = 2
    linewidth: float = 1.5

    def handle(self) -> Line2D:
        """A legend handle showing the color, the line style, and the marker."""
        return Line2D(
            [],
            [],
            color=self.color,
            linewidth=self.linewidth,
            linestyle=self.linestyle,
            marker=self.marker,
            markersize=4,
            markeredgecolor=style.SURFACE,
            markeredgewidth=0.6,
        )


@dataclass
class Study:
    name: str
    directory: Path
    curves: dict
    summary: list[dict]
    parameters: list[dict]
    results: list[dict]

    def curve(self, key: tuple, axis: str, metric: str) -> dict[str, np.ndarray] | None:
        rows = self.curves.get((key, axis, metric))
        if not rows:
            return None
        rows = sorted(rows, key=lambda r: r["x"])
        names = ("x", "p10", "p25", "p50", "p75", "p90", "mean", "active")
        return {name: np.array([r[name] for r in rows], dtype=float) for name in names}

    def row(self, key: tuple) -> dict | None:
        for row in self.summary:
            if config_key(row) == key:
                return row
        return None

    @property
    def instances(self) -> int:
        return len(self.parameters)

    def constant(self, field: str) -> float:
        """The median over instances of an instance-level quantity."""
        return float(np.median([p[field] for p in self.parameters]))

    def recommended_b(self) -> tuple[int, int]:
        """``b*`` of Lemma 4.2 and the unrelaxed ``b*`` of Remark 4.4 (modes)."""
        best = _mode([p["b_best"] for p in self.parameters])
        exact = _mode([p.get("b_star_exact", nan) for p in self.parameters], best)
        return best, exact


def load(name: str, results_dir: Path) -> Study:
    directory = results_dir / name
    curves: dict = defaultdict(list)
    for row in read_rows(directory / "curves.csv"):
        curves[(config_key(row), row["axis"], row["metric"])].append(row)
    summary = read_rows(directory / "summary.csv")
    parameters = read_rows(directory / "parameters.csv")
    results = read_rows(directory / "results.csv")
    if not summary or not curves:
        raise FileNotFoundError(
            f"{directory} has no summary.csv or curves.csv; run the study first"
        )
    return Study(name, directory, dict(curves), summary, parameters, results)


# --------------------------------------------------------------------------
# Series selection.


def _fmt(value: float) -> str:
    return f"{value:.3g}"


def _mode(values: list, default: int = 1) -> int:
    counts: dict[int, int] = defaultdict(int)
    for v in values:
        if not (isinstance(v, float) and math.isnan(v)):
            counts[int(v)] += 1
    return max(counts, key=counts.get) if counts else default


def _gd_label(study: Study) -> str | None:
    """The gradient descent baseline's label with its stepsize, if it ran."""
    template = LABELS[study.name]["gd"]
    if not template:
        return None
    steps = [p.get("gd_step", nan) for p in study.parameters]
    steps = [s for s in steps if not (isinstance(s, float) and math.isnan(s))]
    return template.format(step=_fmt(float(np.median(steps))) if steps else "1")


def _b_suffix(b: int, b_best: int, b_exact: int) -> str:
    suffix = ""
    if b == b_best:
        suffix += f" $= {BSTAR}$ (Lemma 4.2)"
    if b == b_exact:
        suffix += f" $= {BEXACT}$ (Remark 4.4)"
    return suffix


def sweep_series(study: Study, sweep: str) -> list[Series]:
    """The grid in the fixed sweep palette, the recommended values in the accent.

    Lemma 4.2's ``b*`` is drawn solid with a plus marker and Remark 4.4's
    unrelaxed ``b*`` dashed with a star marker when it is a separate run.
    """
    rows = [
        r for r in study.summary if r["sweep"] == sweep and r["method"] == "inexact"
    ]
    field = "sigma" if sweep == "sigma" else "b"
    grid = sorted((r for r in rows if r["in_grid"]), key=lambda r: r[field])
    added = sorted((r for r in rows if not r["in_grid"]), key=lambda r: r[field])
    series = []
    colors = style.sweep_colors(len(grid))
    markers = style.sweep_markers(len(grid))
    b_best, b_exact = study.recommended_b()
    for row, color, marker in zip(grid, colors, markers, strict=True):
        if sweep == "sigma":
            label = f"${SIGMA} = {_fmt(row['sigma'])}$"
        else:
            label = f"$b = {int(row['b'])}$" + _b_suffix(int(row["b"]), b_best, b_exact)
        series.append(Series(config_key(row), label, color, marker))
    for row in added:
        linestyle, marker = "-", style.ACCENT_MARKER
        if sweep == "sigma":
            label = f"${SIGMASTAR} = {_fmt(row['sigma'])}$ (Prop. 5.1)"
        else:
            b = int(row["b"])
            if b == b_best:
                label = f"${BSTAR} = {b}$ (Lemma 4.2)"
                if b == b_exact:
                    label += f" $= {BEXACT}$ (Remark 4.4)"
            elif b == b_exact:
                label = f"${BEXACT} = {b}$ (Remark 4.4)"
                linestyle, marker = "--", "*"
            else:
                label = f"$b = {b}$"
        if row["n_runs"] < study.instances:
            label += f" [{int(row['n_runs'])}/{study.instances} instances]"
        series.append(
            Series(
                config_key(row),
                label,
                style.ACCENT,
                marker,
                linestyle,
                zorder=4,
                linewidth=1.8,
            )
        )
    return series


def reference_series(study: Study) -> list[Series]:
    series = []
    labels = LABELS[study.name]
    gd_label = _gd_label(study)
    for row in study.summary:
        if row["method"] == "exact" and labels["exact"]:
            series.append(
                Series(
                    config_key(row),
                    labels["exact"],
                    style.REFERENCE,
                    style.REFERENCE_MARKER,
                    zorder=3,
                )
            )
        elif row["method"] == "gd" and gd_label:
            series.append(
                Series(
                    config_key(row),
                    gd_label,
                    style.REFERENCE,
                    style.REFERENCE_MARKER,
                    zorder=3,
                )
            )
        elif row["method"] == "near_exact":
            series.append(
                Series(
                    config_key(row),
                    labels["near"],
                    style.CONTEXT,
                    style.CONTEXT_MARKER,
                    zorder=1,
                )
            )
    return series


# --------------------------------------------------------------------------
# Drawing primitives.


def _whisker_positions(
    x: np.ndarray, count: int, log: bool, index: int, total: int
) -> np.ndarray:
    """``count`` abscissae of ``x`` for the whiskers, staggered by series.

    Positions snap to abscissae that exist, so a whisker always sits on a
    real sample; neighbouring series are shifted by a few samples so that
    their whiskers do not overlap.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    low, high = float(x.min()), float(x.max())
    offset = index - (total - 1) / 2
    if log:
        low = max(low, 1.0)
        base = np.geomspace(low, high, count) if high > low else np.array([high])
        targets = base * (1.0 + 0.045 * offset)
    else:
        base = np.linspace(low, high, count)
        targets = base + offset * max(1.0, 0.012 * (high - low))
    return np.unique([x[np.argmin(np.abs(x - t))] for t in targets])


def draw_curve(
    ax,
    curve: dict,
    series: Series,
    index: int,
    total: int,
    *,
    log_x: bool,
    whiskers: int = 7,
) -> None:
    x, p50 = curve["x"], curve["p50"]
    ok = np.isfinite(p50) & (p50 > 0)
    if not ok.any():
        return
    ax.fill_between(
        x[ok],
        curve["p25"][ok],
        curve["p75"][ok],
        color=series.color,
        alpha=style.BAND_ALPHA,
        linewidth=0,
        zorder=series.zorder - 1,
    )
    ax.plot(
        x[ok],
        p50[ok],
        color=series.color,
        linewidth=series.linewidth,
        linestyle=series.linestyle,
        zorder=series.zorder,
        label=series.label,
    )
    positions = _whisker_positions(x[ok], whiskers, log_x, index, total)
    mask = np.isin(x, positions) & ok
    if not mask.any():
        return
    low = np.maximum(p50[mask] - curve["p10"][mask], 0)
    high = np.maximum(curve["p90"][mask] - p50[mask], 0)
    ax.errorbar(
        x[mask],
        p50[mask],
        yerr=[low, high],
        fmt=series.marker,
        color=series.color,
        markersize=3.6,
        markeredgecolor=style.SURFACE,
        markeredgewidth=0.7,
        elinewidth=0.8,
        capsize=1.8,
        capthick=0.8,
        zorder=series.zorder + 1,
    )


def _finish(ax, xlabel: str | None, ylabel: str | None, *, log_x=False, log_y=True):
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


def _figure_legend(fig, handles, labels, ncol: int, title: str) -> None:
    """A legend above the panels and the title above the legend."""
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        handlelength=1.8,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    rows = math.ceil(len(labels) / ncol)
    height = fig.get_size_inches()[1]
    fig.suptitle(
        title,
        y=1.0 + (0.035 * rows + 0.03) * (5.2 / height),
        color=style.INK,
        fontsize=9,
    )


def _fixed_label(study: Study, sweep: str) -> str:
    row = next(
        r for r in study.summary if r["sweep"] == sweep and r["method"] == "inexact"
    )
    if sweep == "sigma":
        return f"$b = {int(row['b'])}$"
    return f"${SIGMA} = {_fmt(row['sigma'])}$"


def _vertical(ax, x: float, text: str, linestyle: str = "-") -> None:
    """An accent vertical line with its label at the top of the axes."""
    ax.axvline(x, color=style.ACCENT, linewidth=0.9, linestyle=linestyle, zorder=1)
    ax.annotate(
        text,
        xy=(x, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
        color=style.ACCENT,
    )


def _mark_recommended(ax, study: Study, sweep: str, sigma_values: np.ndarray) -> None:
    """Vertical lines at the paper's recommended parameter values."""
    if sweep == "sigma":
        for value in sigma_values:
            _vertical(ax, value, f"${SIGMASTAR}$")
        return
    b_best, b_exact = study.recommended_b()
    if b_best == b_exact:
        _vertical(ax, b_best, f"${BSTAR} = {BEXACT}$")
    else:
        _vertical(ax, b_best, f"${BSTAR}$")
        _vertical(ax, b_exact, f"${BEXACT}$", linestyle="--")


# --------------------------------------------------------------------------
# Figures.


def convergence(study: Study, sweep: str) -> list[Path]:
    labels = LABELS[study.name]
    series = sweep_series(study, sweep) + reference_series(study)
    residual = "feasibility" if study.name == "admm_qp" else "z_norm"
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharey="row")
    panels = [
        ("outer", "suboptimality", 0, 0),
        ("inner", "suboptimality", 0, 1),
        ("outer", residual, 1, 0),
        ("inner", residual, 1, 1),
    ]
    for axis, metric, r, c in panels:
        ax = axes[r, c]
        # Exact ADMM has no inner iterations; gradient descent no outer ones.
        hidden = "exact" if axis == "inner" else "gd"
        drawn = [s for s in series if s.key[0] != hidden]
        for i, s in enumerate(drawn):
            curve = study.curve(s.key, axis, metric)
            if curve is not None:
                draw_curve(ax, curve, s, i, len(drawn), log_x=(axis == "inner"))
        ylabel = None
        if c == 0:
            ylabel = SUBOPT if metric == "suboptimality" else labels["residual"]
        xlabel = "outer iteration $k$" if axis == "outer" else labels["inner"]
        _finish(ax, xlabel if r == 1 else None, ylabel, log_x=(axis == "inner"))
        if r == 0:
            title = (
                "against outer iterations" if axis == "outer" else "against inner work"
            )
            ax.set_title(title, loc="left")
    handles = [s.handle() for s in series]
    which = f"${SIGMA}$ sweep" if sweep == "sigma" else "$b$ sweep"
    title = (
        f"{labels['title']}: {which} at {_fixed_label(study, sweep)}, "
        "$\\delta = 10^{-9}$"
    )
    _figure_legend(fig, handles, [s.label for s in series], 4, title)
    fig.tight_layout()
    return style.save(fig, study.directory / f"convergence_{sweep}")


def _median_termination(study: Study, series: list[Series]) -> float:
    """The outer iteration at which half of the instances have stopped."""
    ends = []
    for s in series:
        curve = study.curve(s.key, "outer", "suboptimality")
        if curve is None:
            continue
        alive = curve["x"][curve["active"] >= curve["active"].max() / 2]
        ends.append(float(alive.max()) if alive.size else float(curve["x"].max()))
    return float(np.median(ends)) if ends else 1.0


def _plateau_start(study: Study, series: list[Series], metric: str) -> float:
    """The outer iteration at which the median of ``metric`` reaches its plateau.

    The first iteration whose median is within a factor two of the final
    median, taken over each series and then the median over series; this is
    where a metric stops resolving progress (the stopping tolerance, or the
    floating-point floor of the ground truth).
    """
    starts = []
    for s in series:
        curve = study.curve(s.key, "outer", metric)
        if curve is None:
            continue
        x, p50 = curve["x"], curve["p50"]
        ok = np.isfinite(p50) & (p50 > 0)
        if not ok.any():
            continue
        final = p50[ok][-1]
        entered = x[ok][p50[ok] <= 2.0 * final]
        starts.append(float(entered.min()) if entered.size else float(x[ok][-1]))
    return float(np.median(starts)) if starts else 1.0


def convergence_zoom(study: Study, sweep: str) -> list[Path]:
    """The outer-iteration panels on two windows, as median lines with markers.

    The early window covers the first 12% of the median run.  The late
    window of each metric ends shortly after that metric's median reaches
    its plateau and starts 20% of the median run earlier, so it shows the
    final approach rather than the floor; curves that coincide at full
    scale separate, and bands and whiskers are left out on purpose.
    """
    labels = LABELS[study.name]
    series = [
        s
        for s in sweep_series(study, sweep) + reference_series(study)
        if s.key[0] != "gd"
    ]
    residual = "feasibility" if study.name == "admm_qp" else "z_norm"
    k_ref = _median_termination(study, series)
    early = (0, max(10, round(0.12 * k_ref)))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    for row, metric in enumerate(("suboptimality", residual)):
        plateau = _plateau_start(study, series, metric)
        late = (max(0, round(plateau - 0.2 * k_ref)), round(plateau + 0.05 * k_ref))
        for col, (name, (low, high)) in enumerate(
            (("early iterations", early), ("final approach", late))
        ):
            ax = axes[row, col]
            lowest, highest = np.inf, -np.inf
            for i, s in enumerate(series):
                curve = study.curve(s.key, "outer", metric)
                if curve is None:
                    continue
                x, p50 = curve["x"], curve["p50"]
                ok = np.isfinite(p50) & (p50 > 0) & (x >= low) & (x <= high)
                if not ok.any():
                    continue
                ax.plot(
                    x[ok],
                    p50[ok],
                    color=s.color,
                    linestyle=s.linestyle,
                    linewidth=s.linewidth,
                    zorder=s.zorder,
                )
                positions = _whisker_positions(x[ok], 5, False, i, len(series))
                mask = np.isin(x, positions) & ok
                ax.plot(
                    x[mask],
                    p50[mask],
                    linestyle="none",
                    marker=s.marker,
                    color=s.color,
                    markersize=4.2,
                    markeredgecolor=style.SURFACE,
                    markeredgewidth=0.7,
                    zorder=s.zorder + 1,
                )
                lowest = min(lowest, float(p50[ok].min()))
                highest = max(highest, float(p50[ok].max()))
            ax.set_xlim(low, high)
            if np.isfinite(lowest):
                ax.set_ylim(lowest / 1.5, highest * 1.5)
            ylabel = None
            if col == 0:
                ylabel = SUBOPT if metric == "suboptimality" else labels["residual"]
            _finish(ax, "outer iteration $k$" if row == 1 else None, ylabel)
            ax.set_title(f"{name}: $k \\in [{low}, {high}]$", loc="left")
    handles = [s.handle() for s in series]
    which = f"${SIGMA}$ sweep" if sweep == "sigma" else "$b$ sweep"
    title = (
        f"{labels['title']}: {which} at {_fixed_label(study, sweep)}, "
        "zoom on outer iterations (medians over instances)"
    )
    _figure_legend(fig, handles, [s.label for s in series], 4, title)
    fig.tight_layout()
    return style.save(fig, study.directory / f"convergence_{sweep}_zoom")


def _summary_points(study: Study, sweep: str, field: str) -> tuple[np.ndarray, dict]:
    rows = [
        r for r in study.summary if r["sweep"] == sweep and r["method"] == "inexact"
    ]
    x_field = "sigma" if sweep == "sigma" else "b"
    rows.sort(key=lambda r: r[x_field])
    x = np.array([r[x_field] for r in rows], dtype=float)
    stats = {
        s: np.array([r.get(f"{field}_{s}", nan) for r in rows], dtype=float)
        for s in ("median", "p10", "p90", "mean", "sd")
    }
    stats["sigma_optimal"] = np.array([bool(r["sigma_is_optimal"]) for r in rows])
    return x, stats


def _points(ax, x, stats, color, *, marker="o", zorder=3):
    low = np.maximum(stats["median"] - stats["p10"], 0)
    high = np.maximum(stats["p90"] - stats["median"], 0)
    ax.errorbar(
        x,
        stats["median"],
        yerr=[low, high],
        fmt=marker,
        color=color,
        markersize=4,
        markeredgecolor=style.SURFACE,
        markeredgewidth=0.7,
        elinewidth=0.9,
        capsize=2,
        capthick=0.9,
        zorder=zorder,
    )
    ax.plot(x, stats["median"], color=color, linewidth=1.2, zorder=zorder - 1)


def _horizontal(ax, value: float, color: str, text: str) -> None:
    ax.axhline(value, color=color, linewidth=0.9, zorder=1)
    ax.annotate(
        text,
        xy=(1.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=style.INK_SECONDARY,
    )


def _reference_lines(ax, study: Study, field: str, *, log_y: bool):
    labels = LABELS[study.name]
    for row in study.summary:
        value = row.get(f"{field}_median", nan)
        if not np.isfinite(value) or (log_y and value <= 0):
            continue
        if row["method"] == "exact" and labels["exact"]:
            _horizontal(ax, value, style.REFERENCE, "exact")
        elif row["method"] == "gd" and labels["gd"] and field != "k_delta":
            _horizontal(ax, value, style.REFERENCE, "gradient descent")
        elif row["method"] == "near_exact":
            _horizontal(ax, value, style.CONTEXT, "near-exact")


def _recommendation_handles(study: Study, sweep: str) -> tuple[list, list[str]]:
    if sweep == "sigma":
        return (
            [Line2D([], [], color=style.ACCENT, linewidth=0.9)],
            [f"recommended ${SIGMASTAR}$ (Prop. 5.1)"],
        )
    b_best, b_exact = study.recommended_b()
    handles = [Line2D([], [], color=style.ACCENT, linewidth=0.9)]
    names = [f"recommended ${BSTAR}$ (Lemma 4.2)"]
    if b_exact != b_best:
        handles.append(
            Line2D([], [], color=style.ACCENT, linewidth=0.9, linestyle="--")
        )
        names.append(f"unrelaxed ${BEXACT}$ (Remark 4.4)")
    else:
        names[0] = f"recommended ${BSTAR}$ (Lemma 4.2) $= {BEXACT}$ (Remark 4.4)"
    return handles, names


def work(study: Study, sweep: str) -> list[Path]:
    labels = LABELS[study.name]
    xlabel = f"inexactness ${SIGMA}$" if sweep == "sigma" else "check interval $b$"
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    x, k = _summary_points(study, sweep, "k_delta")
    sigma_values = x[k["sigma_optimal"]]
    panels = [
        (
            axes[0, 0],
            "k_delta",
            None,
            "accepted updates $K_\\delta$",
            False,
            "outer iterations",
        ),
        (
            axes[0, 1],
            "total_inner",
            "thm43_bound",
            "total inner iterations",
            True,
            "inner work against its bound",
        ),
        (
            axes[1, 0],
            "total_checks",
            None,
            f"total {labels['checks']}",
            True,
            "checks of the stopping tests",
        ),
        (
            axes[1, 1],
            "runtime_seconds",
            None,
            "algorithm wall-clock (s)",
            True,
            "wall-clock time",
        ),
    ]
    for ax, field, bound, ylabel, log_y, title in panels:
        _, stats = _summary_points(study, sweep, field)
        _points(ax, x, stats, style.MEASURED)
        if bound is not None:
            _, bound_stats = _summary_points(study, sweep, bound)
            _points(ax, x, bound_stats, style.BOUND, marker="s")
        _reference_lines(ax, study, field, log_y=log_y)
        _mark_recommended(ax, study, sweep, sigma_values)
        bottom = ax is axes[1, 0] or ax is axes[1, 1]
        _finish(ax, xlabel if bottom else None, ylabel, log_x=True, log_y=log_y)
        ax.set_title(title, loc="left")

    handles = [
        Line2D([], [], color=style.MEASURED, marker="o", markersize=4, linewidth=1.2),
        Line2D([], [], color=style.BOUND, marker="s", markersize=4, linewidth=1.2),
    ]
    names = [
        "measured (median; whiskers 10–90%)",
        "Theorem 4.3 bound with the observed $K_\\delta$ (median)",
    ]
    if labels["exact"]:
        handles.append(Line2D([], [], color=style.REFERENCE, linewidth=0.9))
        names.append(labels["exact"])
    if _gd_label(study):
        handles.append(Line2D([], [], color=style.REFERENCE, linewidth=0.9))
        names.append(_gd_label(study) + " (no outer iterations)")
    handles.append(Line2D([], [], color=style.CONTEXT, linewidth=0.9))
    names.append(labels["near"])
    extra_handles, extra_names = _recommendation_handles(study, sweep)
    handles.extend(extra_handles)
    names.extend(extra_names)
    which = f"${SIGMA}$" if sweep == "sigma" else "$b$"
    title = (
        f"{labels['title']}: work and time over the {which} sweep "
        f"at {_fixed_label(study, sweep)}"
    )
    _figure_legend(fig, handles, names, 2, title)
    fig.tight_layout()
    return style.save(fig, study.directory / f"work_{sweep}")


def per_step(study: Study) -> list[Path]:
    labels = LABELS[study.name]
    series = sweep_series(study, "sigma")
    count = len(series)
    cols = 3
    rows_n = math.ceil(count / cols)
    fig, axes = plt.subplots(
        rows_n, cols, figsize=(7.4, 2.3 * rows_n), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    for index, s in enumerate(series):
        ax = axes[index // cols, index % cols]
        measured = study.curve(s.key, "outer", "inner")
        surrogate = study.curve(s.key, "outer", "bound_surrogate")
        exact = study.curve(s.key, "outer", "bound_exact")
        if measured is not None:
            ax.fill_between(
                measured["x"],
                measured["p25"],
                measured["p75"],
                color=style.MEASURED,
                alpha=style.BAND_ALPHA,
                linewidth=0,
            )
            ax.plot(measured["x"], measured["p50"], color=style.MEASURED, zorder=3)
        if surrogate is not None:
            ax.plot(surrogate["x"], surrogate["p50"], color=style.BOUND, zorder=2)
        if exact is not None and np.isfinite(exact["p50"]).any():
            ax.plot(exact["x"], exact["p50"], color=style.BOUND_EXACT, zorder=2)
        row = study.row(s.key)
        violations = row.get("per_step_violations_total", nan) if row else nan
        shown = int(violations) if np.isfinite(violations) else "–"
        title = s.label.replace(" (Prop. 5.1)", "")
        ax.set_title(f"{title}; violations: {shown}", loc="left")
        xlabel = "outer iteration $k$" if index // cols == rows_n - 1 else None
        ylabel = "inner iterations $N_k$" if index % cols == 0 else None
        _finish(ax, xlabel, ylabel, log_y=False)
    for index in range(count, rows_n * cols):
        axes[index // cols, index % cols].set_visible(False)
    handles = [
        Line2D([], [], color=style.MEASURED, linewidth=1.4),
        Line2D([], [], color=style.BOUND, linewidth=1.4),
    ]
    names = [
        "measured $N_k$ (median, interquartile band)",
        r"Theorem 4.3 bound with $\|R(w_k)\| \geq (1-\sigma)\|y_k-w_k\|$ (median)",
    ]
    if study.name == "admm_qp":
        handles.append(Line2D([], [], color=style.BOUND_EXACT, linewidth=1.4))
        names.append(r"Theorem 4.3 bound with the exact $\|R(w_k)\|$ (median)")
    title = (
        f"{labels['title']}: inner iterations per outer iteration "
        f"against the per-step bound, {_fixed_label(study, 'sigma')}"
    )
    _figure_legend(fig, handles, names, 2, title)
    fig.tight_layout()
    return style.save(fig, study.directory / "per_step")


def _recommended_row(row: dict) -> bool:
    return bool(
        row.get("sigma_is_optimal")
        or row.get("b_optimal_fraction", 0.0) > 0.5
        or row.get("b_exact_optimal_fraction", 0.0) > 0.5
    )


def _shifted_geometric_mean(values: np.ndarray, shift: float) -> float:
    """``exp(mean(log(t + shift))) - shift``, the usual benchmark statistic.

    The shift keeps the few very short runs from dominating a geometric
    mean, while the logarithm keeps the few very long runs from dominating
    an arithmetic one.
    """
    return float(np.exp(np.mean(np.log(values + shift))) - shift)


def _walltime_chart(
    study: Study, stem: str, statistic, title_note: str, axis_note: str
) -> list[Path]:
    """Stacked bars of setup, parameter, and algorithm time per configuration.

    ``statistic`` reduces the per-instance times of each cost to its bar
    segment.  The whisker at the end of a bar spans the 10th to 90th
    percentile over instances of the algorithm time (of the CVXPY time for
    the first bar), placed so that the statistic of that time sits at the
    end of the bar.
    """
    labels = LABELS[study.name]
    rows = []
    gd_label = _gd_label(study)
    for row in study.summary:
        if row["method"] == "exact":
            rows.append((labels["exact"], row))
        elif row["method"] == "gd" and gd_label:
            rows.append((gd_label, row))
        elif row["method"] == "near_exact":
            rows.append((labels["near"], row))
    for s in sweep_series(study, "sigma"):
        row = study.row(s.key)
        name = s.label.replace(" (Prop. 5.1)", "") + f", $b = {int(row['b'])}$"
        rows.append((name, row))
    for s in sweep_series(study, "b"):
        row = study.row(s.key)
        name = s.label.replace(" (Lemma 4.2)", "").replace(" (Remark 4.4)", "")
        rows.append((name + f", ${SIGMA} = {_fmt(row['sigma'])}$", row))
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for result in study.results:
        by_key[config_key(result)].append(result)

    def times(row: dict, *fields: str) -> np.ndarray:
        """Per-instance sums of the given time columns of one configuration."""
        values = np.array(
            [
                sum(float(r.get(f, nan)) for f in fields)
                for r in by_key[config_key(row)]
            ],
            dtype=float,
        )
        return values[np.isfinite(values)]

    truth_times = np.array([p["truth_seconds"] for p in study.parameters], dtype=float)
    names = [name for name, _ in rows]
    setup = np.array([statistic(times(r, "setup_seconds")) for _, r in rows])
    params = np.array(
        [
            statistic(times(r, "parameter_seconds", "calibration_seconds"))
            if _recommended_row(r)
            else 0.0
            for _, r in rows
        ]
    )
    spreads = [truth_times] + [times(r, "runtime_seconds") for _, r in rows]
    centers = np.array([statistic(t) for t in spreads])
    truth, algorithm = centers[0], centers[1:]
    totals = np.concatenate([[truth], setup + params + algorithm])
    left = totals - centers + np.array([np.percentile(t, 10) for t in spreads])
    right = totals - centers + np.array([np.percentile(t, 90) for t in spreads])
    count = len(rows) + 1
    fig, ax = plt.subplots(figsize=(7.4, 0.26 * count + 1.6), constrained_layout=True)
    y = np.arange(count)[::-1]
    height = 0.62
    gap = 0.004 * float(totals.max())
    start = np.zeros(len(rows))
    segments = (
        (setup, style.SEGMENTS[0], "setup (factorizations, $L$)"),
        (
            params,
            style.SEGMENTS[1],
            f"computing ${SIGMASTAR}$ / ${BSTAR}$ (incl. one-off calibration)",
        ),
        (algorithm, style.SEGMENTS[2], "algorithm"),
    )
    for values, color, label in segments:
        widths = np.maximum(values - gap, 0)
        ax.barh(
            y[1:], widths, left=start + gap / 2, height=height, color=color, label=label
        )
        start = start + values
    ax.barh(
        y[:1],
        [truth],
        height=height,
        color=style.CONTEXT,
        label="CVXPY ground truth (per instance; Clarabel)",
    )
    ax.errorbar(
        (left + right) / 2,
        y,
        xerr=(right - left) / 2,
        fmt="none",
        ecolor=style.INK_SECONDARY,
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.9,
        zorder=4,
    )
    for yi, total, reach in zip(y, totals, right, strict=True):
        ax.annotate(
            f"{total:.3g} s",
            xy=(max(total, reach), yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=6.5,
            color=style.INK_SECONDARY,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(["CVXPY ground truth"] + names)
    ax.set_xlabel(f"wall-clock seconds per instance ({axis_note})")
    ax.set_xlim(0, float(max(totals.max(), right.max())) * 1.18)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    handles, legend_names = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color=style.INK_SECONDARY,
            linewidth=0.9,
            marker="|",
            markersize=6,
            markeredgewidth=0.9,
        )
    )
    legend_names.append("whiskers: 10–90% of instances")
    ax.legend(handles, legend_names, loc="lower right", ncol=1)
    ax.set_title(
        f"{labels['title']}: wall-clock per configuration ({title_note})",
        loc="left",
    )
    return style.save(fig, study.directory / stem)


def walltime(study: Study) -> list[Path]:
    """Wall-clock per configuration as medians over instances."""
    return _walltime_chart(
        study,
        "walltime",
        lambda t: float(np.median(t)),
        "median over instances",
        "median",
    )


def walltime_sgm(study: Study) -> list[Path]:
    """Wall-clock per configuration as shifted geometric means over instances."""
    shift = WALLTIME_SHIFT
    return _walltime_chart(
        study,
        "walltime_sgm",
        lambda t: _shifted_geometric_mean(t, shift),
        f"shifted geometric mean over instances, shift {shift:g} s",
        f"shifted geometric mean, shift {shift:g} s",
    )


def _per_outer_seconds(study: Study, method: str, b: int | None = None) -> np.ndarray:
    """Algorithm seconds per outer iteration of every instance of one run."""
    values = []
    for row in study.results:
        if row["method"] != method:
            continue
        if b is not None and (row["sweep"] != "b" or int(row["b"]) != b):
            continue
        if row.get("converged") and row.get("outer_iterations", 0) > 0:
            values.append(row["runtime_seconds"] / row["outer_iterations"])
    return np.array(values, dtype=float)


def bstar_model(study: Study) -> list[Path]:
    """The cost model of Lemma 4.2 with and without the relaxation of Remark 4.4."""
    labels = LABELS[study.name]
    c_in = study.constant("c_in_seconds")
    c_test = study.constant("c_test_seconds")
    alpha = study.constant("alpha")
    c0 = study.constant("c0")
    sigma = study.constant("b_sweep_sigma")
    b_best, b_exact = study.recommended_b()
    grid = sorted(
        {
            int(r["b"])
            for r in study.summary
            if r["sweep"] == "b" and r["method"] == "inexact"
        }
    )
    b_max = max(max(grid), b_best, b_exact)
    bs = np.arange(1, int(1.5 * b_max) + 1)
    relaxed = np.array(
        [theory.b_cost_per_outer(b, c_test, c_in, sigma, c0, alpha) for b in bs]
    )
    exact = np.array(
        [
            theory.b_cost_per_outer(b, c_test, c_in, sigma, c0, alpha, relaxed=False)
            for b in bs
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    ax = axes[0]
    ax.plot(bs, relaxed, color=style.BOUND, marker="s", markersize=2.8, linewidth=1.3)
    ax.plot(
        bs, exact, color=style.BOUND_EXACT, marker="^", markersize=2.8, linewidth=1.3
    )
    _mark_recommended(ax, study, "b", np.array([]))
    _finish(
        ax, "check interval $b$", "modelled seconds per outer iteration", log_x=True
    )
    ax.set_title("cost model of Lemma 4.2", loc="left")

    ax = axes[1]
    medians, lows, highs = [], [], []
    for b in grid:
        values = _per_outer_seconds(study, "inexact", b)
        medians.append(np.median(values) if values.size else nan)
        lows.append(np.percentile(values, 10) if values.size else nan)
        highs.append(np.percentile(values, 90) if values.size else nan)
    stats = dict(median=np.array(medians), p10=np.array(lows), p90=np.array(highs))
    _points(ax, np.array(grid, dtype=float), stats, style.MEASURED)
    for method, color, text in (
        ("exact", style.REFERENCE, "exact"),
        ("near_exact", style.CONTEXT, "near-exact"),
    ):
        values = _per_outer_seconds(study, method)
        if values.size and (method != "exact" or labels["exact"]):
            _horizontal(ax, float(np.median(values)), color, text)
    _mark_recommended(ax, study, "b", np.array([]))
    _finish(
        ax, "check interval $b$", "measured seconds per outer iteration", log_x=True
    )
    ax.set_title("measured, same $b$ sweep", loc="left")

    handles = [
        Line2D([], [], color=style.BOUND, marker="s", markersize=3, linewidth=1.3),
        Line2D(
            [], [], color=style.BOUND_EXACT, marker="^", markersize=3, linewidth=1.3
        ),
        Line2D([], [], color=style.MEASURED, marker="o", markersize=4, linewidth=1.2),
    ]
    names = [
        r"relaxed bound: $c_{in}Fb + c_{test}F + c_{test}\log D/(b\log(1/\alpha))$, "
        r"$F = 1 + 1/(1-\alpha)$",
        r"unrelaxed bound (Remark 4.4): the same with $F = 1 + 1/(1-\alpha^b)$",
        "measured algorithm time per outer iteration (median; whiskers 10–90%)",
    ]
    extra_handles, extra_names = _recommendation_handles(study, "b")
    handles.extend(extra_handles)
    names.extend(extra_names)
    title = (
        f"{labels['title']}: choosing $b$ at ${SIGMA} = {_fmt(sigma)}$ "
        f"($c_{{test}}/c_{{in}} = {c_test / c_in:.2f}$, $\\alpha = {alpha:.3f}$)"
    )
    _figure_legend(fig, handles, names, 1, title)
    fig.tight_layout()
    return style.save(fig, study.directory / "bstar_model")


# --------------------------------------------------------------------------
# The two-dimensional trajectory figure.


def _power_label(value: float) -> str:
    """``10^k`` in mathtext when ``value`` is a power of ten, else ``value``."""
    exponent = math.log10(value)
    if abs(exponent - round(exponent)) < 1e-9:
        return f"$10^{{{round(exponent)}}}$"
    return f"${_fmt(value)}$"


def _contour_levels(high: float) -> tuple[list[float], list[float]]:
    """Powers of ten and their triples over three decades below ``high``.

    Returns the levels and the subset (the powers of ten) to label.
    """
    top = math.floor(math.log10(high))
    exponents = range(top - 3, top + 1)
    levels = sorted(v for k in exponents for v in (10.0**k, 3.0 * 10.0**k) if v <= high)
    return levels, [10.0**k for k in exponents if 10.0**k <= high]


def _trajectory_series(runs: list[dict], instance: dict) -> list[Series]:
    """One series per run, in the sweep styling of the main study."""

    def work(row: dict) -> str:
        steps = f"{int(row['total_inner'])} gradient steps"
        if row["method"] == "gd":
            return steps
        return f"$K_\\delta = {int(row['k_delta'])}$, {steps}"

    grid = sorted(
        (r for r in runs if r["method"] == "inexact" and not r["sigma_is_optimal"]),
        key=lambda r: r["sigma"],
    )
    series = []
    colors = style.sweep_colors(len(grid))
    markers = style.sweep_markers(len(grid))
    for row, color, marker in zip(grid, colors, markers, strict=True):
        label = f"${SIGMA} = {_fmt(row['sigma'])}$ ({work(row)})"
        series.append(Series(config_key(row), label, color, marker))
    for row in runs:
        if row["method"] == "inexact" and row["sigma_is_optimal"]:
            label = f"${SIGMASTAR} = {_fmt(row['sigma'])}$, Prop. 5.1 ({work(row)})"
            series.append(
                Series(
                    config_key(row),
                    label,
                    style.ACCENT,
                    style.ACCENT_MARKER,
                    zorder=4,
                    linewidth=1.8,
                )
            )
        elif row["method"] == "near_exact":
            tolerance = _power_label(float(instance["near_exact_tolerance"]))
            label = f"near-exact PPM, inner tol {tolerance} ({work(row)})"
            series.append(
                Series(
                    config_key(row),
                    label,
                    style.CONTEXT,
                    style.CONTEXT_MARKER,
                    "--",
                    zorder=1,
                )
            )
        elif row["method"] == "gd":
            step = _fmt(float(instance["gd_step"]))
            label = f"gradient descent, step ${step}/L$ ({work(row)})"
            series.append(
                Series(
                    config_key(row),
                    label,
                    style.REFERENCE,
                    style.REFERENCE_MARKER,
                    zorder=2,
                    linewidth=1.0,
                )
            )
    return series


def _trajectory_tables(directory: Path) -> tuple[dict, list[dict], list[dict], dict]:
    instances = read_rows(directory / "instance.csv")
    runs = read_rows(directory / "runs.csv")
    rows = read_rows(directory / "trajectories.csv")
    grid = read_columns(directory / "contours.csv")
    if not (instances and runs and rows and grid):
        raise FileNotFoundError(
            f"{directory} has no trajectory tables; run numerical_experiments."
            "trajectory_2d first"
        )
    return instances[0], runs, rows, grid


def _trajectory_paths(rows: list[dict]) -> dict[tuple, list[dict]]:
    """The rows of each run in iteration order."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[config_key(row)].append(row)
    return {
        key: sorted(value, key=lambda r: r["iteration"])
        for key, value in grouped.items()
    }


def _trajectory_title(instance: dict, middle: str) -> str:
    delta = _power_label(float(instance["delta"]))
    return (
        "Inexact PPM, Huber-loss fitting in two dimensions "
        f"($m = {int(instance['m'])}$ observations, seed {int(instance['seed'])}): "
        f"{middle}, $\\delta = ${delta}; contours of $f - f^\\star$"
    )


def _trajectory_figure(
    directory: Path,
    instance: dict,
    grid: dict,
    series: list[Series],
    paths: dict[tuple, list[dict]],
    extra: list[tuple[Line2D, str]],
    title: str,
    stem: str,
    overlay=None,
) -> list[Path]:
    """Two panels, the full view and a zoom on the optimum, over the contours.

    ``paths`` maps a series key to its rows in iteration order; ``overlay``,
    if given, is called with the axes, the panel name, and the windows of
    both panels to draw further artists; ``extra`` lists legend handles and
    labels for them.
    """
    x_star = np.array([instance["x_star_1"], instance["x_star_2"]], dtype=float)
    x0 = np.array([instance["x0_1"], instance["x0_2"]], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0))
    panels = (
        ("full", "from the start to the optimum"),
        ("zoom", "near the optimum (outlined on the left)"),
    )
    surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for panel, _ in panels:
        mask = grid["panel"] == panel
        x1 = np.unique(grid["x1"][mask])
        x2 = np.unique(grid["x2"][mask])
        values = grid["f_minus_fstar"][mask].reshape(len(x2), len(x1))
        surfaces[panel] = (x1, x2, values)
    windows = {
        panel: (x1.min(), x1.max(), x2.min(), x2.max())
        for panel, (x1, x2, _) in surfaces.items()
    }
    for ax, (panel, name) in zip(axes, panels, strict=True):
        x1, x2, values = surfaces[panel]
        levels, labelled = _contour_levels(float(values.max()))
        contour = ax.contour(
            x1, x2, values, levels=levels, colors=style.AXIS, linewidths=0.6, zorder=0
        )
        ax.clabel(
            contour,
            labelled,
            fmt=_power_label,
            fontsize=5.5,
            colors=style.MUTED,
            inline=True,
            inline_spacing=2,
        )
        for s in series:
            rows = paths.get(s.key)
            if not rows:
                continue
            path = np.array([[r["x1"], r["x2"]] for r in rows], dtype=float)
            every = max(1, len(path) // 12) if s.key[0] == "gd" else 1
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=s.color,
                linestyle=s.linestyle,
                linewidth=s.linewidth,
                marker=s.marker,
                markersize=3.8,
                markeredgecolor=style.SURFACE,
                markeredgewidth=0.6,
                markevery=every,
                zorder=s.zorder,
            )
        if overlay is not None:
            overlay(ax, panel, windows)
        ax.plot(
            [x_star[0]],
            [x_star[1]],
            linestyle="none",
            marker="*",
            markersize=11,
            color=style.INK,
            markeredgecolor=style.SURFACE,
            markeredgewidth=0.7,
            zorder=7,
        )
        if panel == "full":
            ax.plot(
                [x0[0]],
                [x0[1]],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=style.SURFACE,
                markeredgecolor=style.INK,
                markeredgewidth=1.0,
                zorder=7,
            )
        ax.set_xlim(x1.min(), x1.max())
        ax.set_ylim(x2.min(), x2.max())
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(name, loc="left")
    left, right, bottom, top = windows["zoom"]
    axes[0].add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            fill=False,
            edgecolor=style.INK_SECONDARY,
            linewidth=0.7,
            linestyle="--",
            zorder=6,
        )
    )
    handles = [s.handle() for s in series] + [handle for handle, _ in extra]
    labels = [s.label for s in series] + [label for _, label in extra]
    handles += [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            markersize=9,
            color=style.INK,
            markeredgecolor=style.SURFACE,
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5.5,
            markerfacecolor=style.SURFACE,
            markeredgecolor=style.INK,
        ),
    ]
    labels += ["optimum $x^\\star$ (CVXPY)", "start $x_0 = 0$"]
    _figure_legend(fig, handles, labels, 2, title)
    fig.tight_layout()
    return style.save(fig, directory / stem)


def trajectory_2d(directory: Path) -> list[Path]:
    """The paths of the accepted iterates on the two-dimensional Huber instance.

    Left, the full view from the start to the optimum; right, a zoom on the
    optimum, outlined on the left.  The contours are ``f - f*`` at powers
    of ten and their triples, and the series follow the sweep styling of
    the main study.
    """
    directory = Path(directory)
    instance, runs, rows, grid = _trajectory_tables(directory)
    series = _trajectory_series(runs, instance)
    title = _trajectory_title(instance, "accepted iterates at $b = 1$")
    return _trajectory_figure(
        directory,
        instance,
        grid,
        series,
        _trajectory_paths(rows),
        [],
        title,
        "trajectory_2d",
    )


# The inexact run singled out against the near-exact one in ``trajectory_2d_exact``.
TRAJECTORY_EXACT_SIGMA = 0.9


def trajectory_2d_exact(
    directory: Path, sigma: float = TRAJECTORY_EXACT_SIGMA
) -> list[Path]:
    """The near-exact path and one inexact run with its exact subproblem solutions.

    At every iteration of the inexact run the exact solution ``J(w_k)`` of
    the subproblem it approximated is drawn as a hollow marker, joined to
    the accepted iterate by a dotted segment.  Both are numbered by the
    iteration, and a point that passed the acceptance test of several
    consecutive subproblems without any inner step carries all their
    numbers, so that the several exact solutions tied to it are not read
    as several solutions of one subproblem.  The near-exact iterates are
    their own exact solutions to within the inner tolerance.
    """
    directory = Path(directory)
    instance, runs, rows, grid = _trajectory_tables(directory)
    paths = _trajectory_paths(rows)
    x_star = np.array([instance["x_star_1"], instance["x_star_2"]], dtype=float)
    chosen = [
        s
        for s in _trajectory_series(runs, instance)
        if s.key[0] == "near_exact"
        or (s.key[0] == "inexact" and s.key[2] == round(sigma, 10))
    ]
    inexact = next((s for s in chosen if s.key[0] == "inexact"), None)
    if inexact is None:
        raise FileNotFoundError(f"{directory} has no inexact run at sigma = {sigma}")
    steps = [r for r in paths[inexact.key] if math.isfinite(r["exact1"])]
    iterations = [int(r["iteration"]) for r in steps]
    accepted = np.array([[r["x1"], r["x2"]] for r in steps], dtype=float)
    exact = np.array([[r["exact1"], r["exact2"]] for r in steps], dtype=float)
    # Consecutive iterations that accepted the same point share one label.
    groups: list[list[int]] = []
    for i in range(len(steps)):
        if groups and np.linalg.norm(accepted[i] - accepted[groups[-1][0]]) < 1e-12:
            groups[-1].append(i)
        else:
            groups.append([i])

    def inside(point: np.ndarray, window: tuple[float, float, float, float]) -> bool:
        left, right, bottom, top = window
        return bool(left <= point[0] <= right and bottom <= point[1] <= top)

    def overlay(ax, panel: str, windows: dict) -> None:
        window = windows[panel]
        scale = window[1] - window[0]
        placed: list[np.ndarray] = []

        def labelled(point: np.ndarray) -> bool:
            """Inside this panel, not under the zoom outline, clear of the star
            and of the labels already placed."""
            if not inside(point, window):
                return False
            if panel == "full" and inside(point, windows["zoom"]):
                return False
            if np.linalg.norm(point - x_star) <= 0.03 * scale:
                return False
            return all(np.linalg.norm(point - q) > 0.04 * scale for q in placed)

        def annotate(text: str, point: np.ndarray, offset: tuple[int, int]) -> None:
            placed.append(point)
            ax.annotate(
                text,
                xy=(point[0], point[1]),
                xytext=offset,
                textcoords="offset points",
                fontsize=6,
                color=style.INK_SECONDARY,
                zorder=8,
            )

        for a, e in zip(accepted, exact, strict=True):
            ax.plot(
                [e[0], a[0]],
                [e[1], a[1]],
                color=inexact.color,
                linewidth=0.8,
                linestyle=":",
                zorder=3,
            )
        ax.plot(
            exact[:, 0],
            exact[:, 1],
            linestyle="none",
            marker="o",
            markersize=4.8,
            markerfacecolor=style.SURFACE,
            markeredgecolor=inexact.color,
            markeredgewidth=1.0,
            zorder=5,
        )
        for i, (k, e) in enumerate(zip(iterations, exact, strict=True)):
            if labelled(e) and np.linalg.norm(e - accepted[i]) > 0.01 * scale:
                annotate(str(k), e, (3, 3))
        for group in groups:
            if labelled(accepted[group[0]]):
                text = ", ".join(str(iterations[i]) for i in group)
                annotate(text, accepted[group[0]], (3, -9))

    extra = [
        (
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=4.8,
                markerfacecolor=style.SURFACE,
                markeredgecolor=inexact.color,
                markeredgewidth=1.0,
            ),
            "Exact Solution to Subproblem $k$",
        ),
        (
            Line2D([], [], color=inexact.color, linewidth=0.8, linestyle=":"),
            "Gap to accepted iterate",
        ),
    ]
    title = "Trajectories of Exact vs. Inexact PPM"
    return _trajectory_figure(
        directory,
        instance,
        grid,
        chosen,
        paths,
        extra,
        title,
        "trajectory_2d_exact",
        overlay,
    )


FIGURES = {
    "convergence_sigma": lambda s: convergence(s, "sigma"),
    "convergence_b": lambda s: convergence(s, "b"),
    "convergence_sigma_zoom": lambda s: convergence_zoom(s, "sigma"),
    "convergence_b_zoom": lambda s: convergence_zoom(s, "b"),
    "work_sigma": lambda s: work(s, "sigma"),
    "work_b": lambda s: work(s, "b"),
    "per_step": per_step,
    "walltime": walltime,
    "walltime_sgm": walltime_sgm,
    "bstar_model": bstar_model,
}


def make(name: str, results_dir: Path, figures=tuple(FIGURES)) -> list[Path]:
    study = load(name, results_dir)
    written: list[Path] = []
    for figure in figures:
        written.extend(FIGURES[figure](study))
    return written


def main() -> None:
    global WALLTIME_SHIFT
    from .study import DEFAULT_RESULTS_DIR, STUDIES

    parser = argparse.ArgumentParser(description="draw the figures of the studies")
    parser.add_argument(
        "studies", nargs="*", choices=list(STUDIES) + [TRAJECTORY_STUDY], default=[]
    )
    parser.add_argument(
        "--figures", nargs="+", choices=list(FIGURES), default=list(FIGURES)
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--walltime-shift",
        type=float,
        default=WALLTIME_SHIFT,
        help="shift in seconds of the geometric mean in walltime_sgm",
    )
    args = parser.parse_args()
    WALLTIME_SHIFT = args.walltime_shift
    for study in args.studies or list(STUDIES) + [TRAJECTORY_STUDY]:
        try:
            if study == TRAJECTORY_STUDY:
                directory = args.results_dir / TRAJECTORY_STUDY
                paths = trajectory_2d(directory) + trajectory_2d_exact(directory)
            else:
                paths = make(study, args.results_dir, args.figures)
            for path in paths:
                print(path)
        except FileNotFoundError as error:
            print(f"skipping {study}: {error}")


if __name__ == "__main__":
    main()
