"""Command-line entry point for every experiment study.

Each study is also runnable on its own -- ``python -m experiments.theta_sweep
--trials 5`` -- and this dispatcher simply groups them behind one command so
that ``all`` can run the broad set unattended.  Studies write CSVs only; use
``python -m plots.make_plots`` to draw the figures, or pass ``--plot`` here to
do both in one go.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

# study name -> (experiment module, run function, plot module, plot function)
STUDIES = {
    "standard": ("standard", "run_study", "standard", "make"),
    "lp": ("lp_tuning", "run_study", "lp_tuning", "make"),
    "inequality-qp": (
        "inequality_qp_tuning",
        "run_study",
        "inequality_qp_tuning",
        "make",
    ),
    "gamma-sweep": ("gamma_sweep", "run_study", "gamma_sweep", "make"),
    "sigma-sweep": ("sigma_sweep", "run_study", "sigma_sweep", "make"),
    "theta-sweep": ("theta_sweep", "run_study", "theta_sweep", "make"),
    "solver-sweep": ("solver_sweep", "run_study", "solver_sweep", "make"),
    "solver-sweep-randomized": (
        "solver_sweep",
        "run_study",
        "solver_sweep",
        "make_randomized",
    ),
    "inequality-gamma-sweep": (
        "gamma_sweep",
        "run_inequality_study",
        "gamma_sweep",
        "make",
    ),
}
# Focused reruns of a slice another study already covers, or studies long
# enough that they should be asked for by name.
FOCUSED_STUDIES = {
    "solver-sweep",
    "solver-sweep-randomized",
    "inequality-gamma-sweep",
}


def study_parser(name: str) -> argparse.ArgumentParser:
    """Return the argument parser a study defines for itself."""
    module_name, _, _, _ = STUDIES[name]
    return import_module(f"experiments.{module_name}").build_parser()


def run(name: str, argv: list[str] | None = None, plot: bool = False) -> None:
    """Run one study, then optionally draw its figures from the CSVs."""
    module_name, function_name, plot_module, plot_function = STUDIES[name]
    args = study_parser(name).parse_args(argv or [])
    if name == "solver-sweep-randomized":
        args.randomize_spectrum = True
    getattr(import_module(f"experiments.{module_name}"), function_name)(args)
    if plot:
        written = getattr(import_module(f"plots.{plot_module}"), plot_function)(
            args.results_dir
        )
        for path in written:
            print(f"plotted {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Flags after the study name are passed to it; "
            "run `python -m experiments.<module> --help` to see them. "
            "Only --results-dir is understood alongside `all`."
        ),
    )
    parser.add_argument(
        "study", nargs="?", default="standard", choices=(*STUDIES, "all")
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="draw the study's figures from the CSVs it just wrote",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="root directory for generated CSVs, passed on to every study",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    selected, rest = build_parser().parse_known_args(argv)
    # parse_known_args leaves a repeated flag in `rest`; the study parsers
    # define their own copies, so drop the duplicates here.
    rest = [flag for flag in rest if flag != "--plot"]
    shared = (
        ["--results-dir", str(selected.results_dir)]
        if selected.results_dir is not None
        else []
    )
    if selected.study == "all":
        if rest:
            raise SystemExit(
                f"only --results-dir may accompany `all`; got {' '.join(rest)}"
            )
        for name in STUDIES:
            if name not in FOCUSED_STUDIES:
                run(name, shared, plot=selected.plot)
    else:
        run(selected.study, shared + rest, plot=selected.plot)


if __name__ == "__main__":
    main()
