"""Inexact versus exact ADMM on the periodic-grid equality QP.

    python -m studies.run_admm_qp --instances 100 --side 40 --inner gmres

See :mod:`studies.study` for the stages and :file:`studies/README.md` for the
flags and the output tables.
"""

from __future__ import annotations

from .study import main

STUDY = "admm_qp"

if __name__ == "__main__":
    main(STUDY)
