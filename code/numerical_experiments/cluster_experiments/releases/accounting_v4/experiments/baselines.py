"""OSQP, indirect SCS, and a reference Chambolle-Pock/Condat PDHG."""

import json
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from . import solvers
from .dense import DenseDesign
from .problems import norm_bound


def qp_data(p, design=None):
    """Residual lifting avoids forming A.T A for the LASSO QP."""
    if p.family == "qp":
        m = len(p.indices)
        A = sp.csc_matrix((np.ones(m), (np.arange(m), p.indices)), shape=(m, p.n))
        return p.Q.tocsc(), p.q, A, p.target, p.target, p.n
    if p.family != "lasso":
        raise ValueError("logistic regression is not a quadratic program")
    m, d = p.A.shape
    design = p.A if design is None else design
    P = sp.diags(np.r_[np.zeros(2 * d), np.ones(m)], format="csc")
    q = np.r_[np.full(2 * d, p.regularization), np.zeros(m)]
    eq = sp.hstack([design, -design, -sp.eye(m)], format="csc")
    positive = sp.hstack([sp.eye(2 * d), sp.csc_matrix((2 * d, m))], format="csc")
    A = sp.vstack([eq, positive], format="csc")
    return (
        P,
        q,
        A,
        np.r_[p.target, np.zeros(2 * d)],
        np.r_[p.target, np.full(2 * d, np.inf)],
        2 * d,
    )


def ladder(config, tol):
    """Native tolerances tried in order, each warm-started from the last.

    A single native tolerance either stops early (failing the common test) or
    overshoots it. Tightening until the common test passes, with every rung
    charged, mirrors how the inner methods stop on the common criterion.
    """
    factor = config.get("baseline_eps_factor", 0.1)
    rungs = int(config.get("baseline_tolerance_rungs", 3))
    if rungs < 1:
        raise ValueError("baseline_tolerance_rungs must be positive")
    return [tol * factor * 0.1**k for k in range(rungs)]


def external(p, config):
    started = perf_counter()
    method = config["inner"]
    tol = config.get("tolerance", 1e-6)
    tolerances = ladder(config, tol)
    limit = config.get("time_limit", 1800.0)
    iterations = config.get("max_inner", 1_000_000)
    if method == "osqp" and p.family == "logistic":
        return dict(
            status="not_applicable",
            converged=False,
            reason="OSQP cannot solve the logistic objective exactly",
        ), []
    design = p.A
    if isinstance(design, DenseDesign):
        try:
            design = design.to_sparse(
                int(config.get("max_explicit_matrix_gb", 2.0) * 1024**3)
            )
        except MemoryError as error:
            return dict(
                status="representation_memory_limit",
                converged=False,
                reason=str(error),
                solver_setup_seconds=perf_counter() - started,
                solve_seconds=0.0,
            ), []
    history = []
    value, status, count, metrics, used = None, "", 0, None, []

    def record(x, rung, backend_status, backend_iterations):
        nonlocal value, status, metrics
        value, status = x, backend_status
        used.append(rung)
        if value is None or not np.all(np.isfinite(value)):
            return False
        metrics = p.metrics(value)
        history.append(
            dict(
                iteration=len(history),
                inner=backend_iterations,
                checks=len(history) + 1,
                elapsed=perf_counter() - started,
                backend_tolerance=rung,
                **metrics,
            )
        )
        return max(metrics["stationarity"], metrics["feasibility"]) <= tol

    if method == "scs" and p.family == "logistic":
        import cvxpy as cp

        x = cp.Variable(p.n)
        objective = cp.sum(cp.logistic(-cp.multiply(p.target, design @ x))) / p.A.shape[
            0
        ] + 0.5 * p.regularization * cp.sum_squares(x)
        model = cp.Problem(cp.Minimize(objective))
        # Canonicalization is explicitly charged to setup.
        model.get_problem_data(cp.SCS)
        setup = perf_counter() - started
        for k, rung in enumerate(tolerances):
            remaining = limit - (perf_counter() - started - setup)
            if remaining <= 0 or solvers.INTERRUPTED:
                break
            model.solve(
                solver=cp.SCS,
                use_indirect=True,
                eps_abs=rung,
                eps_rel=rung,
                max_iters=iterations,
                time_limit_secs=remaining,
                warm_start=k > 0,
                verbose=False,
            )
            count += model.solver_stats.num_iters or 0
            if record(x.value, rung, model.status, count):
                break
    else:
        P, q, A, lower, upper, original_n = qp_data(p, design)
        if method == "osqp":
            import osqp

            model = osqp.OSQP()
            tick = perf_counter()
            model.setup(
                P=sp.triu(P, format="csc"),
                q=q,
                A=A,
                l=lower,
                u=upper,
                eps_abs=tolerances[0],
                eps_rel=tolerances[0],
                max_iter=iterations,
                time_limit=limit,
                polishing=False,
                verbose=False,
                rho=config.get("baseline_step", 0.1),
            )
            osqp_setup = perf_counter() - tick
            setup = perf_counter() - started
            for k, rung in enumerate(tolerances):
                remaining = limit - (perf_counter() - started - setup)
                if remaining <= 0 or solvers.INTERRUPTED:
                    break
                # OSQP 1.1.3 counts its own setup against the first solve's
                # time limit (verified); add it back so the solve receives the
                # same budget as every other method.
                model.update_settings(
                    eps_abs=rung,
                    eps_rel=rung,
                    time_limit=remaining + (osqp_setup if k == 0 else 0.0),
                )
                answer = model.solve(raise_error=False)
                count += answer.info.iter
                if record(
                    answer.x[:original_n] if answer.x is not None else None,
                    rung,
                    answer.info.status,
                    count,
                ):
                    break
        else:
            import scs

            equality = np.flatnonzero(np.isfinite(upper) & (lower == upper))
            positive = np.flatnonzero(np.isinf(upper))
            Ac = sp.vstack([A[equality], -A[positive]], format="csc")
            bc = np.r_[lower[equality], -lower[positive]]
            data = dict(P=sp.triu(P, format="csc"), A=Ac, b=bc, c=q)
            cone = dict(z=len(equality), l=len(positive))
            setup = None
            warm = {}
            for rung in tolerances:
                spent = 0.0 if setup is None else perf_counter() - started - setup
                remaining = limit - spent
                if remaining <= 0 or solvers.INTERRUPTED:
                    break
                options = dict(
                    eps_abs=rung,
                    eps_rel=rung,
                    max_iters=iterations,
                    time_limit_secs=remaining,
                    verbose=False,
                    scale=config.get("baseline_step", 0.1),
                )
                if hasattr(scs, "LinearSolver"):
                    options["linear_solver"] = scs.LinearSolver.CPU_INDIRECT
                else:
                    options["use_indirect"] = True
                # SCS fixes tolerances at construction; later rungs rebuild the
                # solver and that rebuild is charged to the solve, not setup.
                model = scs.SCS(data, cone, **options)
                if setup is None:
                    setup = perf_counter() - started
                answer = model.solve(warm_start=bool(warm), **warm)
                count += answer["info"]["iter"]
                warm = dict(x=answer["x"], y=answer["y"], s=answer["s"])
                if record(
                    answer["x"][:original_n], rung, answer["info"]["status"], count
                ):
                    break
    if value is None or not np.all(np.isfinite(value)):
        return dict(
            status="solver_failed",
            backend_status=status,
            converged=False,
            native_tolerance_ladder=json.dumps(used),
        ), history
    converged = bool(max(metrics["stationarity"], metrics["feasibility"]) <= tol)
    return dict(
        status="converged" if converged else "tolerance_not_met",
        backend_status=status,
        backend_tolerance=used[-1],
        native_tolerance_ladder=json.dumps(used),
        converged=converged,
        **metrics,
        solver_setup_seconds=setup,
        solve_seconds=perf_counter() - started - setup,
        total_inner=count,
    ), history


def pdhg(p, config):
    """Reference first-order baseline, no factorization and no hidden inner solve.

    QP: smooth Condat step with tau*(L/2+eta*||A||^2)<1.
    LASSO/logistic: Chambolle-Pock with tau*eta*||A||^2<1.
    """
    started = perf_counter()
    factor = config.get("baseline_step", 1.0)
    tol = config.get("tolerance", 1e-6)
    if p.family == "qp":
        tau, eta = 0.99 / (p.L / 2 + factor), factor
        x, y = np.zeros(p.n), np.zeros(len(p.indices))
    else:
        bound = np.sqrt(norm_bound(p.A))
        # Logistic loss is averaged over m samples: its dual variables have
        # size 1/m. Balance primal/dual steps in that same scaling.
        if p.family == "logistic":
            factor *= p.A.shape[0]
        tau, eta = 0.99 * factor / bound, 0.99 / (factor * bound)
        x, y = np.zeros(p.A.shape[1]), np.zeros(p.A.shape[0])
    bar = x.copy()
    history = []
    status = "max_inner"
    for k in range(config.get("max_inner", 1_000_000)):
        if p.family == "qp":
            old = x.copy()
            grad = p.gradient(x)
            grad[p.indices] += y
            x -= tau * grad
            y += eta * ((2 * x - old)[p.indices] - p.target)
            candidate = x
        elif p.family == "lasso":
            y = (y + eta * (p.A @ bar - p.target)) / (1 + eta)
            old = x.copy()
            v = x - tau * (p.A.T @ y)
            x = np.sign(v) * np.maximum(abs(v) - tau * p.regularization, 0)
            bar = 2 * x - old
            candidate = np.r_[np.maximum(x, 0), np.maximum(-x, 0)]
        else:
            v = y + eta * p.target * (p.A @ bar)
            m = p.A.shape[0]
            lo, hi = np.full(m, -1 / m + 1e-15), np.full(m, -1e-15)
            for _ in range(40):
                mid = (lo + hi) / 2
                a = np.clip(-m * mid, 1e-15, 1 - 1e-15)
                f = mid - v - eta * np.log(a / (1 - a))
                lo = np.where(f < 0, mid, lo)
                hi = np.where(f >= 0, mid, hi)
            y = (lo + hi) / 2
            old = x.copy()
            x = (x - tau * (p.A.T @ (p.target * y))) / (1 + tau * p.regularization)
            bar = 2 * x - old
            candidate = x
        if k % 10 == 0 or k + 1 == config.get("max_inner", 1_000_000):
            metrics = p.metrics(candidate)
            history.append(
                dict(
                    iteration=k,
                    inner=k + 1,
                    checks=len(history) + 1,
                    elapsed=perf_counter() - started,
                    **metrics,
                )
            )
            if max(metrics["stationarity"], metrics["feasibility"]) <= tol:
                status = "converged"
                break
        if solvers.INTERRUPTED:
            status = "interrupted"
            break
        if perf_counter() - started >= config.get("time_limit", 1800.0):
            status = "time_limit"
            break
    return dict(
        status=status,
        converged=status == "converged",
        **p.metrics(candidate),
        total_inner=k + 1,
        solver_setup_seconds=0.0,
        solve_seconds=perf_counter() - started,
    ), history
