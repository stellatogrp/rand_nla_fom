"""Draw every figure the paper uses from the CSVs under ``results/``.

Nothing here recomputes anything: a study whose CSVs are missing is reported
and skipped, so this is safe to run at any point while results accumulate.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from experiments._paths import DEFAULT_RESULTS_DIR

from ._data import MissingResults

# figure set -> (module, factory)
FIGURES = {
    "standard": ("standard", "make"),
    "lp": ("lp_tuning", "make"),
    "inequality-qp": ("inequality_qp_tuning", "make"),
    "gamma-sweep": ("gamma_sweep", "make"),
    "sigma-sweep": ("sigma_sweep", "make"),
    "theta-sweep": ("theta_sweep", "make"),
    "solver-sweep": ("solver_sweep", "make"),
    "solver-sweep-randomized": ("solver_sweep", "make_randomized"),
}


def draw(name: str, results_dir: Path) -> list[Path]:
    module_name, function_name = FIGURES[name]
    factory = getattr(import_module(f"plots.{module_name}"), function_name)
    return factory(results_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "figures",
        nargs="*",
        default=list(FIGURES),
        choices=(*FIGURES, []),
        metavar="FIGURE_SET",
        help="figure sets to draw",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of skipping a figure set whose CSVs are missing",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    drawn = 0
    for name in args.figures:
        try:
            written = draw(name, args.results_dir)
        except MissingResults as error:
            if args.strict:
                raise SystemExit(str(error)) from error
            print(f"skipped {name}: {error}")
            continue
        drawn += len(written)
        for path in written:
            print(f"wrote {path}")
    print(f"{drawn} figures written")


if __name__ == "__main__":
    main()
