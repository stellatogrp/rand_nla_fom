"""Command-line entry point for all experiment studies."""

from __future__ import annotations

import argparse
from importlib import import_module


STUDIES = {
    "standard": ("scripts.standard", "run_study"),
    "lp": ("scripts.lp_tuning", "run_study"),
    "inequality-qp": ("scripts.inequality_qp_tuning", "run_study"),
    "gamma-sweep": ("scripts.gamma_sweep", "run_study"),
    "sigma-sweep": ("scripts.sigma_sweep", "run_study"),
    "theta-sweep": ("scripts.theta_sweep", "run_study"),
    "solver-sweep": ("scripts.solver_sweep", "run_study"),
    "solver-sweep-randomized": ("scripts.solver_sweep", "run_randomized_study"),
    "lp-sigma-sweep": ("scripts.sigma_sweep", "run_lp_study"),
    "inequality-gamma-sweep": ("scripts.gamma_sweep", "run_inequality_study"),
}
FOCUSED_STUDIES = {
    "lp-sigma-sweep",
    "inequality-gamma-sweep",
    "solver-sweep",
    "solver-sweep-randomized",
}


def run_study(name: str) -> None:
    """Import and run one study."""
    module_name, function_name = STUDIES[name]
    getattr(import_module(module_name), function_name)()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        nargs="?",
        default="standard",
        choices=(*STUDIES, "all"),
    )
    selected = parser.parse_args().study
    if selected == "all":
        for name in STUDIES:
            if name not in FOCUSED_STUDIES:
                run_study(name)
    else:
        run_study(selected)


if __name__ == "__main__":
    main()
