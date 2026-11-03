"""Comparable measured work units for publication and diagnostic reports."""

import math

SCOPE = "problem_operators_after_build_including_calibration"


def numeric(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def primary_work(row, config):
    """Count A and A.T vector products for regression; Hessian products for QP.

    A forward product and a transpose product each count once. All counts
    include calibration/setup within the measured post-build scope. Native
    backend counts and historical estimates are not reconstructed here.
    """
    regression = config["family"] in ("lasso", "logistic")
    unit = (
        "design_forward_plus_transpose_vector_products"
        if regression
        else "quadratic_operator_vector_products"
    )
    if (
        str(row.get("work_count_complete")) != "True"
        or row.get("work_count_scope") != SCOPE
        or config["inner"] in ("osqp", "scs")
    ):
        return dict(value=float("nan"), unit=unit, scope="unknown", evidence="unknown")
    counts = [
        numeric(row, name)
        for name in (
            ("design_forward_products", "design_transpose_products")
            if regression
            else ("hessian_products_total",)
        )
    ]
    if not all(math.isfinite(v) and v >= 0 for v in counts):
        return dict(value=float("nan"), unit=unit, scope="unknown", evidence="unknown")
    return dict(value=sum(counts), unit=unit, scope=SCOPE, evidence="measured")
