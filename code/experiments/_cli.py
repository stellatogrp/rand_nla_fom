"""Argument-parser pieces shared by the experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._paths import DEFAULT_RESULTS_DIR


def base_parser(
    description: str, *, resumable: bool = False
) -> argparse.ArgumentParser:
    """Return a parser carrying the flags every study understands."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="root directory for generated CSVs",
    )
    if resumable:
        parser.add_argument(
            "--force",
            action="store_true",
            help="recompute settings already present in results.csv",
        )
    return parser


def kind_argument(parser: argparse.ArgumentParser, kinds: tuple[str, ...]) -> None:
    """Add the flag restricting a sweep to some of its problem families."""
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=kinds,
        default=list(kinds),
        metavar="KIND",
        help="problem families to run",
    )
