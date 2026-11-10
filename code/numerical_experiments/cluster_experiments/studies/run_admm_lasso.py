"""Inexact versus exact ADMM on the split LASSO.

    python -m studies.run_admm_lasso --instances 100 --dataset random_normal

See :mod:`studies.study` for the stages and :file:`studies/README.md` for the
flags and the output tables.
"""

from __future__ import annotations

from .study import main

STUDY = "admm_lasso"

if __name__ == "__main__":
    main(STUDY)
