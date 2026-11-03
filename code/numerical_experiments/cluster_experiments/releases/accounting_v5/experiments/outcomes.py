"""Separate outcomes of a method from infrastructure events that need a rerun.

A worker exception is classified by its type. Operating-system errors (files,
disks, connections) and truncated reads are infrastructure events: a rerun can
succeed. Memory exhaustion is the method exceeding its resource limit. Every
other exception is a deterministic solver failure, such as OSQP reporting
``OSQP_NONCVX_ERROR`` (code 4) or CG losing positive curvature; rerunning it
reproduces the same failure, so it must not block parameter selection.
"""

# Outcomes of the method itself. Anything else (interruption, a dead worker,
# a missing clean timing) is an infrastructure event and must be rerun.
METHOD_OUTCOMES = {
    "converged",
    "time_limit",
    "max_inner",
    "max_outer",
    "inner_iteration_limit",
    "inner_time_limit",
    "tolerance_not_met",
    "solver_failed",
    "solver_error",
    "resource_limit",
    "representation_memory_limit",
    "worker_timeout",
}


def _names(base):
    found = {base.__name__}
    for child in base.__subclasses__():
        found |= _names(child)
    return found


INFRASTRUCTURE_ERRORS = frozenset(_names(OSError) | {"EOFError"})


def error_status(name):
    """Result status for a worker exception, by exception class name."""
    if name in INFRASTRUCTURE_ERRORS:
        return "infrastructure_error"
    if name == "MemoryError":
        return "resource_limit"
    return "solver_error"


def error_type(row):
    """Exception class name, recorded directly or as the ``error`` prefix."""
    if row.get("error_type"):
        return row["error_type"]
    text = row.get("error") or ""
    return text.split(":", 1)[0].strip() if ":" in text else ""


def is_method_outcome(row):
    status = row.get("status")
    if status in METHOD_OUTCOMES:
        return True
    if status == "error":
        # Releases before v5 recorded every exception as "error". Classify the
        # recorded type; an unknown type stays an infrastructure event.
        name = error_type(row)
        return bool(name) and error_status(name) != "infrastructure_error"
    return False
