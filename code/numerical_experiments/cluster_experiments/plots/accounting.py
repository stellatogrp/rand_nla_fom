"""Prevent pilot/shared-data or incomplete-accounting timings from ranking main runs."""

import math

from .corrections import clean_seconds


def main_eligible(row):
    # accounting_v3 marked a run incomplete after any earlier hard kill, yet its
    # final attempt restarted from scratch and is itself a clean measurement.
    complete = str(row.get("setup_accounting_complete", True)) == "True" or (
        math.isfinite(clean_seconds(row))
        and row.get("attempt_worker_wall_seconds") not in (None, "")
    )
    return (
        str(row.get("primary_eligible", True)) == "True"
        or row.get("timing_scope") == "independent_attempts_including_retries"
    ) and (row.get("setup_policy", "independent") == "independent" and complete)


def finite_time(row):
    value = clean_seconds(row)
    return math.isfinite(value) and value > 0
