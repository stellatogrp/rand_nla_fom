"""Recognize explicit solver/resource failures; leave unknown errors unresolved."""

METHOD_OUTCOMES = {
    "converged",
    "time_limit",
    "max_inner",
    "max_outer",
    "inner_iteration_limit",
    "inner_time_limit",
    "tolerance_not_met",
    "solver_failed",
    "resource_limit",
    "representation_memory_limit",
    "worker_timeout",
}


class SolverFailure(ArithmeticError):
    """An explicitly identified numerical failure inside an iterative solver."""


def _names(base):
    found = {base.__name__}
    for child in base.__subclasses__():
        found |= _names(child)
    return found


# Failures of the environment (files, disks, connections, truncated reads):
# a rerun can succeed.
INFRASTRUCTURE_ERRORS = frozenset(_names(OSError) | {"EOFError"})


def error_status(name, message=""):
    """Only recognized numerical/resource failures may lose parameter selection.

    Type names alone cannot distinguish input or programming errors from
    numerical failures. Environment failures need a rerun; any other
    unrecognized exception needs investigation, because a rerun of a bug or of
    bad input reproduces it.
    """
    if name == "MemoryError":
        return "resource_limit"
    if name == "SolverFailure":
        return "solver_error"
    if name == "OSQPException" and message.strip() == "4":
        return "solver_error"  # Explicit OSQP_NONCVX_ERROR backend outcome.
    if name == "OSQPException" and message.strip() == "5":
        return "resource_limit"  # OSQP_MEM_ALLOC_ERROR: the factorization did not fit.
    if name == "ArithmeticError" and message.strip() == "CG lost positive curvature":
        return "solver_error"  # Recognize this exact diagnostic in older releases.
    if name in INFRASTRUCTURE_ERRORS:
        return "infrastructure_error"
    return "unresolved_error"


def error_type(row):
    """Exception class name, recorded directly or as the error prefix."""
    if row.get("error_type"):
        return row["error_type"]
    text = row.get("error") or ""
    return text.split(":", 1)[0].strip() if ":" in text else ""


def classify(row):
    """Method outcome, or the action an unrankable result needs."""
    status = row.get("status")
    if status in ("error", "solver_error"):
        # Recheck v5 solver_error rows too: v5 classified arbitrary exceptions
        # as solver errors, including checksum failures and programming bugs.
        text = row.get("error") or ""
        message = text.split(":", 1)[1].strip() if ":" in text else ""
        status = error_status(error_type(row), message)
    # A recorded solver_error counts only after the recheck above confirms it.
    if status in METHOD_OUTCOMES or status == "solver_error":
        return "method"
    # Interruptions, dead workers, and missing records are environment events.
    return "investigate" if status == "unresolved_error" else "rerun"


def is_method_outcome(row):
    return classify(row) == "method"


def blocked(rows_by_id):
    """Message listing which unrankable runs to rerun and which to investigate."""
    groups = {"rerun": set(), "investigate": set()}
    for run_id, row in rows_by_id.items():
        action = classify(row)
        if action != "method":
            groups[action].add(run_id)
    parts = []
    if groups["rerun"]:
        parts.append(
            f"rerun after infrastructure events (experiments.run --force): "
            f"{sorted(groups['rerun'])[:20]}"
        )
    if groups["investigate"]:
        parts.append(
            f"investigate unresolved errors, then rerun: "
            f"{sorted(groups['investigate'])[:20]}"
        )
    count = sum(len(ids) for ids in groups.values())
    return f"{count} runs cannot be ranked yet; " + "; ".join(parts)
