"""Summarize the SDP sanity experiments: one row per study and method.

    uv run python -m sdp.summarize --results-dir results/sdp_v1

Reads every ``<results-dir>/<study>/runs.csv`` written by :mod:`sdp.run`,
writes ``<results-dir>/summary.csv`` (medians over the instances of a study),
and prints the table with the outcome of each sanity check.  The checks, per
run, with their tolerances:

* ``converged``: the terminal test ``Q <= delta`` stopped the run;
* ``inclusions``: the returned ``Z`` is positive semidefinite, ``B`` is
  negative semidefinite, and ``Z B = 0`` (eigenvalues beyond ``1e-10`` in the
  wrong direction, or ``||Z B|| > 1e-10 ||Z|| ||B||``, fail);
* ``proposition``: the bounds of Proposition 5.2 at termination,
  ``||(E(X), X) - (b, Z)|| <= delta`` and
  ``||a + E'b|| <= c delta (||E|| + sqrt(rho))``;
* ``theory``: the one-step decrease and the update bound of Section 3 hold at
  every accepted update up to ``1e-10`` (the decrease needs a planted
  solution; without one only the update bound is checked).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

TOLERANCE = 1e-10

SUMMARY_FIELDS = [
    "study",
    "family",
    "n",
    "m",
    "label",
    "instances",
    "converged",
    "inclusions_ok",
    "proposition_ok",
    "theory_ok",
    "outer_iterations",
    "cg_steps",
    "lanczos_matvecs",
    "matvecs_per_outer_over_n",
    "subspace_dim_over_n",
    "rank_needed_over_n",
    "seconds",
    "worst_coupling_over_delta",
    "worst_stationarity_over_bound",
    "worst_objective_error",
    "worst_decrease_violation",
    "hosts",
]


def _float(row: dict, key: str) -> float:
    value = row[key]
    return float(value) if value not in ("", None) else float("nan")


def checks(row: dict) -> dict:
    """The four sanity checks of one ``runs.csv`` row (module docstring)."""
    delta = _float(row, "delta")
    converged = row["converged"] == "True"
    inclusions = (
        _float(row, "min_eigenvalue_Z") >= -TOLERANCE
        and _float(row, "max_eigenvalue_B") <= TOLERANCE
        and _float(row, "complementarity") <= TOLERANCE
    )
    proposition = (
        converged
        and _float(row, "coupling_residual") <= delta * (1.0 + 1e-9)
        and _float(row, "stationarity_residual") <= _float(row, "stationarity_bound")
    )
    decrease = _float(row, "decrease_violation")
    theory = _float(row, "ratio_violation") <= TOLERANCE and (
        math.isnan(decrease) or decrease <= TOLERANCE
    )
    return dict(
        converged=converged,
        inclusions=inclusions,
        proposition=proposition,
        theory=theory,
    )


def objective_error(row: dict) -> float:
    """``|<C, Z> - optimal value|``: the planted value, else the reference."""
    truth = _float(row, "objective_star")
    if math.isnan(truth):
        truth = _float(row, "objective_reference")
    return abs(_float(row, "objective") - truth)


def read_runs(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*/runs.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["label"])].append(row)
    summary = []
    for (study, label), runs in groups.items():
        outcomes = [checks(run) for run in runs]
        n = int(runs[0]["n"])

        def median(key: str, runs=runs) -> float:
            return statistics.median(_float(run, key) for run in runs)

        def count(name: str, outcomes=outcomes) -> str:
            return f"{sum(o[name] for o in outcomes)}/{len(outcomes)}"

        outer = [max(_float(run, "outer_iterations"), 1.0) for run in runs]
        per_outer = statistics.median(
            _float(run, "lanczos_matvecs") / k / n
            for run, k in zip(runs, outer, strict=True)
        )
        errors = [objective_error(run) for run in runs]
        decreases = [_float(run, "decrease_violation") for run in runs]
        decreases = [d for d in decreases if not math.isnan(d)]
        summary.append(
            dict(
                study=study,
                family=runs[0]["family"],
                n=n,
                m=int(runs[0]["m"]),
                label=label,
                instances=len(runs),
                converged=count("converged"),
                inclusions_ok=count("inclusions"),
                proposition_ok=count("proposition"),
                theory_ok=count("theory"),
                outer_iterations=median("outer_iterations"),
                cg_steps=median("cg_steps"),
                lanczos_matvecs=median("lanczos_matvecs"),
                matvecs_per_outer_over_n=per_outer,
                subspace_dim_over_n=median("subspace_dim_median") / n,
                rank_needed_over_n=median("rank_needed_median") / n,
                seconds=median("seconds"),
                worst_coupling_over_delta=max(
                    _float(run, "coupling_residual") / _float(run, "delta")
                    for run in runs
                ),
                worst_stationarity_over_bound=max(
                    _float(run, "stationarity_residual")
                    / _float(run, "stationarity_bound")
                    for run in runs
                ),
                worst_objective_error=max(errors),
                worst_decrease_violation=max(decreases) if decreases else float("nan"),
                hosts=";".join(sorted({run["hostname"] for run in runs})),
            )
        )
    return summary


def render(summary: list[dict]) -> str:
    """The summary as a fixed-width table, one block per study."""
    columns = [
        ("label", "method", "{}"),
        ("converged", "conv", "{}"),
        ("inclusions_ok", "incl", "{}"),
        ("proposition_ok", "prop", "{}"),
        ("theory_ok", "thry", "{}"),
        ("outer_iterations", "outer", "{:.0f}"),
        ("cg_steps", "CG", "{:.0f}"),
        ("lanczos_matvecs", "matvecs", "{:.0f}"),
        ("subspace_dim_over_n", "dim/n", "{:.2f}"),
        ("rank_needed_over_n", "need/n", "{:.2f}"),
        ("seconds", "sec", "{:.2f}"),
        ("worst_objective_error", "obj err", "{:.1e}"),
    ]
    lines = []
    for study in dict.fromkeys(row["study"] for row in summary):
        rows = [row for row in summary if row["study"] == study]
        cells = [[form.format(row[key]) for key, _, form in columns] for row in rows]
        widths = [
            max(len(title), *(len(cell[i]) for cell in cells))
            for i, (_, title, _) in enumerate(columns)
        ]
        first = rows[0]
        lines.append(
            f"{study} ({first['family']}, n = {first['n']}, m = {first['m']}, "
            f"{first['instances']} instances; medians)"
        )
        lines.append(
            "  ".join(
                title.ljust(width) if i == 0 else title.rjust(width)
                for i, ((_, title, _), width) in enumerate(
                    zip(columns, widths, strict=True)
                )
            )
        )
        for cell in cells:
            lines.append(
                "  ".join(
                    text.ljust(width) if i == 0 else text.rjust(width)
                    for i, (text, width) in enumerate(zip(cell, widths, strict=True))
                )
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results/sdp"))
    args = parser.parse_args(argv)
    rows = read_runs(args.results_dir)
    if not rows:
        raise SystemExit(f"no runs.csv under {args.results_dir}")
    summary = summarize(rows)
    with (args.results_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    print(render(summary))
    failed = [
        f"{row['study']} {row['label']} seed {row['seed']}: {name}"
        for row in rows
        for name, passed in checks(row).items()
        if not passed
    ]
    if failed:
        print(f"{len(failed)} failed checks:")
        print("\n".join(f"  {line}" for line in failed))
    else:
        print(f"all checks passed on {len(rows)} runs")
    return summary


if __name__ == "__main__":
    main()
