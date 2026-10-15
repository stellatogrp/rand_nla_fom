"""Inexact proximal point on Huber-loss fitting (convex, not strongly convex).

    python -m numerical_experiments.run_ppm_huber --instances 100

See :mod:`numerical_experiments.study` for the flags and the output tables.
"""

from __future__ import annotations

from .study import build_parser, run_study

STUDY = "ppm_huber"


def main() -> None:
    args = build_parser(STUDY).parse_args()
    run_study(STUDY, args)


if __name__ == "__main__":
    main()
