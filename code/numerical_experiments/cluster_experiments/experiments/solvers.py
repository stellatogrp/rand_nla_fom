"""Bounded-memory versions of the current ADMM and proximal-point tests."""

import signal
from time import perf_counter

import numpy as np

from .initialization import settings, starting_point
from .inner import Inner
from .limits import budgets

INTERRUPTED = False


def interrupted(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True


def install_signals():
    signal.signal(signal.SIGUSR1, interrupted)
    signal.signal(signal.SIGTERM, interrupted)


def solve(p, config, callback=None, state=None, prepared_inner=None):
    """Algorithm 3 for QP/LASSO, Algorithm 1 for smooth logistic regression.

    Checkpoints contain O(n) vectors at outer boundaries. No iterate history.
    Inner work and checks are charged separately, including terminal checks.
    """
    global INTERRUPTED
    INTERRUPTED = False
    logistic = p.family == "logistic"
    ratio = config.get("ratio", 10.0)
    gamma, c = ratio / p.L, p.L / ratio
    sigma, stride = config["sigma"], config["stride"]
    if not 0 <= sigma <= 1 or stride < 1 or ratio <= 0:
        raise ValueError("invalid sigma, stride, or ratio")
    limits = budgets(config)
    time_limit = limits["time_limit"]
    max_inner = limits["max_inner"]
    max_outer = limits["max_outer"]
    tol = config.get("tolerance", 1e-6)
    initialization = settings(config)
    if prepared_inner is not None and (
        logistic
        or prepared_inner.p is not p
        or prepared_inner.c != c
        or prepared_inner.method != config["inner"]
    ):
        raise ValueError(
            "prepared inner solver does not match the problem/configuration"
        )
    rng = (
        prepared_inner.rng
        if prepared_inner is not None
        else np.random.default_rng(config["seed"] + 100_003)
    )
    inner = (
        None
        if logistic
        else prepared_inner
        if prepared_inner is not None
        else Inner(
            p,
            c,
            config["inner"],
            rng,
            config.get("block_size", 32),
            config.get("rank", 64),
            gd_factor=config.get("gd_factor", 1.0),
        )
    )
    state = {} if state is None else state
    if "rng" in state:
        rng.bit_generator.state = state["rng"]
    x = state.get("x", np.zeros(p.n)).copy()
    z = state.get("z", np.zeros(p.n)).copy()
    dual = state.get("dual", np.zeros(p.n)).copy()
    w = state.get("w", np.zeros(p.n)).copy()
    total_inner = int(state.get("total_inner", 0))
    checks = int(state.get("checks", 0))
    offset = float(state.get("elapsed", 0.0))
    started = perf_counter()
    setup = (0.0 if inner is None else inner.setup_seconds) + state.get("setup", 0.0)
    status = "max_outer"
    history = []
    outer = int(state.get("outer", 0))
    last_checkpoint = perf_counter()
    initialization_seconds = float(state.get("initialization_seconds", 0.0))
    while outer < max_outer:
        inner_started = perf_counter()
        inner_updates = 0
        tick = perf_counter()
        x = starting_point(x, outer, config)
        initialization_seconds += perf_counter() - tick
        initial_point_norm = float(np.linalg.norm(x))
        initial_graph_residual = None
        if not logistic:
            w = z - dual / c
            inner.start(x, c * w - p.q)
        g = None
        while True:
            checks += 1
            if g is None:
                g = p.gradient(x)
            metrics = None
            if logistic:
                e = x + gamma * g - w
                displacement = x - w
                terminal = np.linalg.norm(g) <= tol
                z_new = x
            else:
                e = x + g / c - w
                z_new = p.project(x - g / c)
                displacement = z_new + g / c - w
                terminal = np.linalg.norm(x - z_new) <= tol
            if initial_graph_residual is None:
                initial_graph_residual = float(np.linalg.norm(e))
            # A small graph residual alone does not guarantee a KKT solution.
            if terminal:
                metrics = p.metrics(x, gradient=g)
                if max(metrics["stationarity"], metrics["feasibility"]) <= tol:
                    status = "converged"
                    break
            if INTERRUPTED:
                status = "interrupted"
                break
            if offset + perf_counter() - started >= time_limit:
                status = "time_limit"
                break
            if total_inner >= max_inner:
                status = "max_inner"
                break
            if inner_updates >= limits["max_inner_per_outer"]:
                status = "inner_iteration_limit"
                break
            if perf_counter() - inner_started >= limits["inner_time_limit"]:
                status = "inner_time_limit"
                break
            if config.get("near_exact", False) and np.linalg.norm(e) <= config.get(
                "exact_tolerance", 1e-10
            ):
                break
            if not config.get("near_exact", False) and np.linalg.norm(
                e
            ) <= sigma * np.linalg.norm(displacement):
                break
            for _ in range(
                min(
                    stride,
                    max_inner - total_inner,
                    limits["max_inner_per_outer"] - inner_updates,
                )
            ):
                if (
                    INTERRUPTED
                    or offset + perf_counter() - started >= time_limit
                    or perf_counter() - inner_started >= limits["inner_time_limit"]
                ):
                    break
                if logistic:
                    # Minimax constant GD step for spectrum [mu+1/gamma,L+1/gamma].
                    eta = config.get("gd_factor", 1.0) * 2 / (p.L + p.mu + 2 / gamma)
                    if g is None:
                        g = p.gradient(x)
                    x -= eta * (g + (x - w) / gamma)
                else:
                    inner.step(
                        q_x=g - p.q
                        if config["inner"] == "gd" and g is not None
                        else None
                    )
                    x = inner.x
                g = None
                total_inner += 1
                inner_updates += 1
        elapsed = offset + perf_counter() - started
        if metrics is None:
            metrics = p.metrics(x, gradient=g)
        history.append(
            dict(
                iteration=outer,
                inner=total_inner,
                checks=checks,
                elapsed=elapsed,
                initial_point_norm=initial_point_norm,
                initial_graph_residual=initial_graph_residual,
                inner_initialization=initialization["inner_initialization"],
                **metrics,
            )
        )
        if status in (
            "converged",
            "interrupted",
            "time_limit",
            "max_inner",
            "inner_iteration_limit",
            "inner_time_limit",
        ):
            break
        if logistic:
            w = w - gamma * g
        else:
            dual += c * (x - z)
            z = z_new
        outer += 1
        if callback and perf_counter() - last_checkpoint >= config.get(
            "checkpoint_seconds", 60
        ):
            callback(
                dict(
                    x=x,
                    z=z,
                    dual=dual,
                    w=w,
                    total_inner=total_inner,
                    checks=checks,
                    outer=outer,
                    elapsed=elapsed,
                    setup=setup,
                    rng=rng.bit_generator.state,
                    initialization_seconds=initialization_seconds,
                ),
                history,
            )
            history.clear()
            last_checkpoint = perf_counter()
    if not history:
        metrics = p.metrics(x)
    # Only store a restart state for a completed outer boundary. On a signal
    # inside an inner loop, the last atomic periodic checkpoint is retained.
    return dict(
        status=status,
        **limits,
        converged=status == "converged",
        **initialization,
        initialization_seconds=initialization_seconds,
        **metrics,
        outer=outer,
        total_inner=total_inner,
        checks=checks,
        solve_seconds=offset + perf_counter() - started,
        solver_setup_seconds=setup,
        inner_hessian_products=0 if inner is None else inner.products,
        row_updates=0 if inner is None else inner.row_updates,
        block_factorizations=0 if inner is None else inner.block_factorizations,
        c=c if not logistic else float("nan"),
        gamma=gamma,
    ), history
