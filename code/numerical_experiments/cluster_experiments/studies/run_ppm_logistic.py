"""Inexact proximal point on ridge logistic regression versus gradient descent.

    python -m studies.run_ppm_logistic --instances 100 --dataset random_normal

See :mod:`studies.study` for the stages and :file:`studies/README.md` for the
flags and the output tables.
"""

from __future__ import annotations

from .study import main

STUDY = "ppm_logistic"

if __name__ == "__main__":
    main(STUDY)
