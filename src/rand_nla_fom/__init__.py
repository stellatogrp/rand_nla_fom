"""Preconditioned primal-dual Douglas--Rachford methods."""

from .drs import DRSResult, primal_dual_drs, quadratic_prox

__all__ = ["DRSResult", "primal_dual_drs", "quadratic_prox"]
