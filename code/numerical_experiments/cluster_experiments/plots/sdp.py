"""Figures of the SDP sanity experiments (``sdp/run.py``).

    uv run python -m plots.sdp --results-dir results/sdp_v1

One figure per study, ``<results-dir>/<study>/sanity.{png,pdf}``, drawn from
that study's ``histories.csv`` for one instance (``--seed``, default the
smallest) and one block size (``--block-size``, default the smallest):

* the terminal quantity ``Q_k`` of exact proximal ADMM and of the doubly
  inexact method at every ``sigma``, with the tolerance ``delta``;
* the achieved ratio ``E_k / P_k`` of the accepted pairs against ``sigma``;
* the conjugate gradient steps of each outer iteration;
* for the largest ``sigma``, the dimension of the Krylov subspace behind the
  accepted projection, the smallest dimension that could have passed the test
  (``rank_needed``), and the number of positive Ritz values, against ``n``.

A study with the one-sided projection (``certified_`` and ``one_sided_``
labels) also gets ``<study>/one_sided.{png,pdf}``:

* ``Q_k`` of exact proximal ADMM and of the certified one-sided method;
* the products with ``M`` of each outer iteration, next to those of the
  two-sided pair (when the study ran it) and to ``n``;
* for the middle ``sigma``, the dimension of the Krylov subspace behind the
  accepted pair and its number of positive Ritz values, against ``n``;
* for the guard without the certificate, the largest eigenvalue of every
  accepted ``B`` relative to the larger of ``||B||`` and ``||Z||`` (the
  off-the-clock diagnostic): a point above the threshold is a ``B`` outside
  the normal cone.

Long runs are thinned in ``histories.csv`` (``sdp.run --history-rows``), so the
curves show the stored iterations.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from plots import style

FIGURE_SIZE = (style.TEXT_WIDTH, 4.6)


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(rows: list[dict], key: str, cast=float) -> list:
    return [cast(row[key]) for row in rows]


def _markevery(count: int) -> int:
    return max(count // 8, 1)


def figure(directory: Path, seed: int | None, block: int | None) -> list[Path]:
    """Draw the sanity figure of the study in ``directory``."""
    runs = _read(directory / "runs.csv")
    seeds = sorted({int(row["seed"]) for row in runs})
    seed = seeds[0] if seed is None else seed
    runs = [row for row in runs if int(row["seed"]) == seed]
    inexact = [row for row in runs if row["label"].startswith("inexact_")]
    if not inexact:
        raise SystemExit(f"{directory} has no doubly inexact runs for seed {seed}")
    blocks = sorted({int(row["block_size"]) for row in inexact})
    block = blocks[0] if block is None else block
    inexact = sorted(
        (row for row in inexact if int(row["block_size"]) == block),
        key=lambda row: float(row["sigma"]),
    )
    histories: dict[str, list[dict]] = defaultdict(list)
    for row in _read(directory / "histories.csv"):
        if int(row["seed"]) == seed:
            histories[row["label"]].append(row)
    first = inexact[0]
    n, delta = int(first["n"]), float(first["delta"])

    style.apply()
    fig, axes = style.plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex=True)
    (ax_q, ax_ratio), (ax_cg, ax_dim) = axes
    colors = style.sweep_colors(len(inexact))
    markers = style.sweep_markers(len(inexact))

    handles, labels = [], []
    if "exact" in histories:
        rows = histories["exact"]
        (line,) = ax_q.plot(
            _series(rows, "k", int),
            _series(rows, "Q"),
            color=style.REFERENCE,
            marker=style.REFERENCE_MARKER,
            markersize=4,
            markevery=_markevery(len(rows)),
            linewidth=1.2,
        )
        handles.append(line)
        labels.append("Exact proximal ADMM")
    for run, color, marker in zip(inexact, colors, markers, strict=True):
        rows = histories[run["label"]]
        k = _series(rows, "k", int)
        sigma = float(run["sigma"])
        common = dict(
            color=color,
            marker=marker,
            markersize=3.5,
            markevery=_markevery(len(rows)),
        )
        (line,) = ax_q.plot(k, _series(rows, "Q"), linewidth=1.2, **common)
        handles.append(line)
        labels.append(rf"Doubly inexact, $\sigma = {sigma:g}$")
        # Only the accepted pairs have a ratio to compare with sigma.
        accepted = [row for row in rows if row["accepted_by"] == "relative"]
        ax_ratio.plot(
            _series(accepted, "k", int),
            [float(row["E"]) / float(row["P"]) for row in accepted],
            linewidth=0.8,
            **dict(common, markevery=_markevery(len(accepted))),
        )
        ax_ratio.axhline(sigma, color=color, linewidth=0.8, linestyle=(0, (4, 2)))
        ax_cg.plot(k, _series(rows, "cg_steps", int), linewidth=0.8, **common)

    ax_q.axhline(delta, color=style.INK, linewidth=0.6, linestyle=(0, (4, 2)))
    ax_q.annotate(
        r"$\delta$",
        (1.0, delta),
        xycoords=("axes fraction", "data"),
        xytext=(-3, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
        color=style.INK,
    )
    ax_q.set_yscale("log")
    ax_q.set_title("Terminal quantity $Q_k$")
    ax_ratio.set_title("Achieved ratio $E_k / P_k$ (dashed: $\\sigma$)")
    ax_ratio.set_ylim(0.0, 1.0)
    ax_cg.set_title("Conjugate gradient steps per outer iteration")
    ax_cg.set_ylim(bottom=0)

    # The projection panel: one sigma (the largest, the loosest test), three
    # measures of one unit on one axis.
    widest = inexact[-1]
    rows = histories[widest["label"]]
    k = _series(rows, "k", int)
    every = _markevery(len(rows))
    ax_dim.axhline(n, color=style.INK, linewidth=0.6, linestyle=(0, (4, 2)))
    ax_dim.plot(
        k,
        _series(rows, "subspace_dim", int),
        color=colors[-1],
        marker=markers[-1],
        markersize=3.5,
        markevery=every,
        linewidth=1.2,
        label="Krylov subspace used",
    )
    ax_dim.plot(
        k,
        _series(rows, "rank_needed", int),
        color=style.ACCENT,
        marker=style.ACCENT_MARKER,
        markersize=3.5,
        markevery=(every // 2, every),
        linewidth=0.9,
        linestyle=(0, (1, 1)),
        label="Smallest that passes the test",
    )
    ax_dim.plot(
        k,
        _series(rows, "rank_positive", int),
        color=style.CONTEXT,
        marker=style.CONTEXT_MARKER,
        markersize=3.5,
        markevery=every,
        linewidth=0.9,
        label="Positive Ritz values",
    )
    ax_dim.set_ylim(0, 1.25 * n)
    ax_dim.set_title(
        rf"Projection subspace, $\sigma = {float(widest['sigma']):g}$ (dashed: $n$)"
    )
    ax_dim.set_ylabel("Dimension")
    ax_dim.legend(loc="center right", handlelength=2.4)
    for ax in (ax_cg, ax_dim):
        ax.set_xlabel("Outer iteration $k$")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.04),
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.suptitle(
        f"{first['study']}: {first['family']} SDP, $n = {n}$, $m = {first['m']}$, "
        f"seed {seed}, block size {block}",
        fontsize=9,
        color=style.INK,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return style.save(fig, directory / "sanity")


# Largest relative eigenvalue of B that still counts as rounding error
# (``sdp.run.VIOLATION``), and the floor at which the log axis shows the rest.
VIOLATION = 1e-10
FLOOR = 1e-17


def _instance(directory: Path, seed: int | None):
    """The runs and histories of one instance of a study."""
    runs = _read(directory / "runs.csv")
    seeds = sorted({int(row["seed"]) for row in runs})
    seed = seeds[0] if seed is None else seed
    runs = [row for row in runs if int(row["seed"]) == seed]
    histories: dict[str, list[dict]] = defaultdict(list)
    for row in _read(directory / "histories.csv"):
        if int(row["seed"]) == seed:
            histories[row["label"]].append(row)
    return seed, runs, histories


def _sweep(runs: list[dict], prefix: str, block: int | None) -> list[dict]:
    """The runs of one method at one block size, by increasing sigma."""
    rows = [row for row in runs if row["label"].startswith(prefix)]
    if not rows:
        return []
    blocks = sorted({int(row["block_size"]) for row in rows})
    block = blocks[0] if block is None or block not in blocks else block
    rows = [row for row in rows if int(row["block_size"]) == block]
    return sorted(rows, key=lambda row: float(row["sigma"]))


def one_sided_figure(
    directory: Path, seed: int | None, block: int | None
) -> list[Path]:
    """Draw the figure of the one-sided projection for the study in ``directory``."""
    seed, runs, histories = _instance(directory, seed)
    certified = _sweep(runs, "certified_", block)
    guarded = _sweep(runs, "one_sided_", block)
    if not certified:
        raise SystemExit(f"{directory} has no certified one-sided runs")
    first = certified[0]
    n, delta = int(first["n"]), float(first["delta"])
    block = int(first["block_size"])
    middle = certified[len(certified) // 2]
    two_sided = [
        row
        for row in _sweep(runs, "inexact_", block)
        if float(row["sigma"]) == float(middle["sigma"])
    ]

    style.apply()
    fig, axes = style.plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex=True)
    (ax_q, ax_work), (ax_dim, ax_cone) = axes
    colors = style.sweep_colors(len(certified))
    markers = style.sweep_markers(len(certified))

    handles, labels = [], []
    if "exact" in histories:
        rows = histories["exact"]
        (line,) = ax_q.plot(
            _series(rows, "k", int),
            _series(rows, "Q"),
            color=style.REFERENCE,
            marker=style.REFERENCE_MARKER,
            markersize=4,
            markevery=_markevery(len(rows)),
            linewidth=1.2,
        )
        handles.append(line)
        labels.append("Exact proximal ADMM")
    if two_sided:
        rows = histories[two_sided[0]["label"]]
        (line,) = ax_work.plot(
            _series(rows, "k", int),
            [max(int(row["lanczos_matvecs"]), 1) for row in rows],
            color=style.CONTEXT,
            marker=style.CONTEXT_MARKER,
            markersize=3.5,
            markevery=_markevery(len(rows)),
            linewidth=1.0,
        )
        handles.append(line)
        labels.append(rf"Two-sided, $\sigma = {float(middle['sigma']):g}$")
    for run, color, marker in zip(certified, colors, markers, strict=True):
        rows = histories[run["label"]]
        k = _series(rows, "k", int)
        common = dict(
            color=color,
            marker=marker,
            markersize=3.5,
            markevery=_markevery(len(rows)),
        )
        (line,) = ax_q.plot(k, _series(rows, "Q"), linewidth=1.2, **common)
        handles.append(line)
        labels.append(rf"One-sided, certified, $\sigma = {float(run['sigma']):g}$")
        # An outer iteration accepted without a new pair costs no product;
        # the log axis shows it at one.
        ax_work.plot(
            k,
            [max(int(row["lanczos_matvecs"]), 1) for row in rows],
            linewidth=0.8,
            **common,
        )
    for run, color, marker in zip(guarded, colors, markers, strict=False):
        rows = histories[run["label"]]
        ax_cone.plot(
            _series(rows, "k", int),
            [max(float(row["violation"]), FLOOR) for row in rows],
            linestyle="none",
            color=color,
            marker=marker,
            markersize=2.5,
        )

    for ax, level, text in ((ax_q, delta, r"$\delta$"), (ax_work, n, "$n$")):
        ax.axhline(level, color=style.INK, linewidth=0.6, linestyle=(0, (4, 2)))
        ax.annotate(
            text,
            (1.0, level),
            xycoords=("axes fraction", "data"),
            xytext=(-3, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=style.INK,
        )
    ax_q.set_yscale("log")
    ax_q.set_title("Terminal quantity $Q_k$")
    ax_work.set_yscale("log")
    ax_work.set_title("Products with $M$ per outer iteration")

    rows = histories[middle["label"]]
    k = _series(rows, "k", int)
    every = _markevery(len(rows))
    index = certified.index(middle)
    ax_dim.axhline(n, color=style.INK, linewidth=0.6, linestyle=(0, (4, 2)))
    ax_dim.plot(
        k,
        _series(rows, "subspace_dim", int),
        color=colors[index],
        marker=markers[index],
        markersize=3.5,
        markevery=every,
        linewidth=1.2,
        label="Krylov subspace used",
    )
    ax_dim.plot(
        k,
        _series(rows, "rank_positive", int),
        color=style.CONTEXT,
        marker=style.CONTEXT_MARKER,
        markersize=3.5,
        markevery=every,
        linewidth=0.9,
        label="Positive Ritz values",
    )
    ax_dim.set_yscale("log")
    ax_dim.set_title(
        rf"Projection subspace, $\sigma = {float(middle['sigma']):g}$ (dashed: $n$)"
    )
    ax_dim.set_ylabel("Dimension")
    ax_dim.legend(loc="upper right", handlelength=2.4)
    # Counts read better as plain numbers at 1, 2, 5, 10, ... than as powers
    # of ten.
    for ax in (ax_work, ax_dim):
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())

    ax_cone.axhline(VIOLATION, color=style.INK, linewidth=0.6, linestyle=(0, (4, 2)))
    ax_cone.set_yscale("log")
    ax_cone.set_ylim(FLOOR / 3.0, 10.0)
    ax_cone.set_title("Largest eigenvalue of accepted $B$ (guard only)")
    ax_cone.annotate(
        "Outside the normal cone",
        (0.98, VIOLATION),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
        color=style.INK,
    )
    for ax in (ax_dim, ax_cone):
        ax.set_xlabel("Outer iteration $k$")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(handles), 3),
        bbox_to_anchor=(0.5, -0.08),
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.suptitle(
        f"{first['study']}: {first['family']} SDP, $n = {n}$, $m = {first['m']}$, "
        f"seed {seed}, {block} guard vectors",
        fontsize=9,
        color=style.INK,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return style.save(fig, directory / "one_sided")


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results/sdp"))
    parser.add_argument("--study", action="append", help="study name (default: all)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args(argv)
    directories = sorted(path.parent for path in args.results_dir.glob("*/runs.csv"))
    if args.study:
        directories = [path for path in directories if path.name in args.study]
    if not directories:
        raise SystemExit(f"no studies under {args.results_dir}")
    written = []
    for directory in directories:
        labels = {row["label"] for row in _read(directory / "runs.csv")}
        if any(label.startswith("inexact_") for label in labels):
            written.extend(figure(directory, args.seed, args.block_size))
        if any(label.startswith("certified_") for label in labels):
            written.extend(one_sided_figure(directory, args.seed, args.block_size))
    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main()
