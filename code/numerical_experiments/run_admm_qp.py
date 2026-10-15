"""Inexact versus exact ADMM on an equality-constrained convex QP.

    python -m numerical_experiments.run_admm_qp --instances 100

See :mod:`numerical_experiments.study` for the flags and the output tables.
"""

from __future__ import annotations

from .study import build_parser, run_study

STUDY = "admm_qp"


def main() -> None:
    args = build_parser(STUDY).parse_args()
    run_study(STUDY, args)


if __name__ == "__main__":
    main()
