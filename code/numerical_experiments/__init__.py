"""Numerical experiments for ``tex/General_inexactness.tex``.

The package tests the inner-work theorems of the inexact preconditioned
proximal point framework on two problem families: inexact ADMM on an
equality-constrained convex QP and inexact PPM on Huber-loss fitting.

Run the studies with ``python -m numerical_experiments.run_admm_qp`` and
``python -m numerical_experiments.run_ppm_huber``; draw the figures with
``python -m numerical_experiments.plots``.
"""
