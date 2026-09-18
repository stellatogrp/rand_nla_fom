"""Where generated results live.

Results are written outside ``code/`` so that the package directory holds only
source.  The location follows the repository layout by default and can be
redirected with ``RAND_NLA_FOM_RESULTS`` or a ``--results-dir`` flag.
"""

from __future__ import annotations

import os
from pathlib import Path

# code/experiments/_paths.py -> code/experiments -> code -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = Path(
    os.environ.get("RAND_NLA_FOM_RESULTS", REPO_ROOT / "results")
)


def study_dir(study: str, results_dir: Path | str | None = None) -> Path:
    """Return the directory holding one study's CSVs and figures."""
    root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    return root / study


RESULTS = "results.csv"
HISTORIES = "histories.csv"
SUMMARY = "summary.csv"
