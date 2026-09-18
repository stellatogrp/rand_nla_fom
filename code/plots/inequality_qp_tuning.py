"""Figures for the slack-form inequality-QP tuning study."""

from __future__ import annotations

from pathlib import Path

from experiments._paths import study_dir

from ._data import column, load_histories, select
from ._style import plot_convergence, plt, save_figure

STUDY = "inequality_qp_tuning"
YLABEL = "relative objective error + inequality violation"
PANELS = (
    (
        "proxy",
        "preconditioner",
        "proxy_preconditioners.pdf",
        "Inequality QP proxy: preconditioner sweep",
    ),
    (
        "proxy",
        "sigma_theta",
        "proxy_sigma_theta.pdf",
        "Inequality QP proxy: sigma/theta sweep",
    ),
    ("large", "transfer", "large_convergence.pdf", "Large inequality QP"),
)


def _traces(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    return [
        (label, sorted(trace, key=lambda row: row["iteration"]))
        for label, trace in grouped.items()
    ]


def plot_adaptation(histories: list[dict], output: Path) -> Path:
    """Show how the adaptive policy moved ``theta_k`` and ``sigma_k``."""
    rows = select(histories, stage="transfer", adaptive=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    for label, trace in _traces(rows):
        iterations = column(trace, "iteration")
        axes[0].plot(iterations, column(trace, "theta_k"), label=label)
        axes[1].plot(iterations, column(trace, "sigma_k"), label=label)
    axes[0].set_ylabel(r"relaxation $\theta_k$")
    axes[1].set_ylabel(r"inner tolerance $\sigma_k$")
    for axis in axes:
        axis.set_xlabel("outer iteration")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Progress-based adaptive parameters")
    return save_figure(fig, output)


def make(results_dir: Path | None = None) -> list[Path]:
    histories = load_histories(STUDY, results_dir)
    output = study_dir(STUDY, results_dir)
    written = []
    for problem, stage, filename, title in PANELS:
        rows = select(histories, problem=problem, stage=stage)
        if not rows:
            continue
        written.append(
            plot_convergence(_traces(rows), title, YLABEL, output / filename)
        )
    if select(histories, stage="transfer", adaptive=True):
        written.append(plot_adaptation(histories, output / "adaptive_parameters.pdf"))
    return written
