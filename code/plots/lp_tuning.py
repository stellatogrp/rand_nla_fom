"""Figures for the large-LP tuning study."""

from __future__ import annotations

from pathlib import Path

from experiments._paths import study_dir

from ._data import load_histories, select
from ._style import plot_convergence

STUDY = "lp_tuning"
YLABEL = "relative objective error + feasibility"
PANELS = (
    (
        "proxy",
        "preconditioner",
        "proxy_preconditioners.pdf",
        "LP proxy: data-scaled preconditioner sweep",
    ),
    (
        "proxy",
        "sigma_theta",
        "proxy_sigma_theta.pdf",
        "LP proxy: tolerance and relaxation sweep",
    ),
    ("large", "transfer", "large_lp_convergence.pdf", "Large LP: tuned transfer"),
)


def make(results_dir: Path | None = None) -> list[Path]:
    histories = load_histories(STUDY, results_dir)
    output = study_dir(STUDY, results_dir)
    written = []
    for problem, stage, filename, title in PANELS:
        rows = select(histories, problem=problem, stage=stage)
        if not rows:
            continue
        traces: dict[str, list[dict]] = {}
        for row in rows:
            traces.setdefault(row["label"], []).append(row)
        written.append(
            plot_convergence(
                [
                    (label, sorted(trace, key=lambda row: row["iteration"]))
                    for label, trace in traces.items()
                ],
                title,
                YLABEL,
                output / filename,
            )
        )
    return written
