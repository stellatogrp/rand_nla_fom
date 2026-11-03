"""Use the existing paper's cost model only where constants are justified."""

import math
from copy import deepcopy
from time import perf_counter

import numpy as np

from . import theory
from .inner import Inner


def constants(p, c, method, block_size=32, preconditioner_condition=1.0, gd_factor=1.0):
    condition = (p.L + c) / (p.mu + c)
    if method == "gd":
        eta = gd_factor * 2 / (p.L + p.mu + 2 * c)
        contraction = max(abs(1 - eta * (p.mu + c)), abs(1 - eta * (p.L + c)))
        return contraction**2, 1.0, True
    if method in ("cg", "nystrom_pcg"):
        condition *= preconditioner_condition
        rho = (math.sqrt(condition) - 1) / (math.sqrt(condition) + 1)
        # CG energy contraction converted to Euclidean residual norm.
        return rho**2, 4 * condition, True
    if method == "block_kaczmarz":
        blocks = math.ceil(p.n / block_size)
        # Uniform partition-block sampling. ||H_block|| <= ||H|| and
        # sum ||H_block e||^2 = ||H e||^2 give this conservative bound.
        return 1 - 1 / (blocks * condition**2), condition**2, False
    return None  # no unverified AGD/cyclic/restarted-GMRES bound


def resolve(p, config, *, return_inner=False):
    """Calibrate per problem, inner solver, block size/rank, and penalty."""
    config = dict(config)
    logistic = p.family == "logistic"
    c = p.L / config["ratio"]
    started = perf_counter()
    recommendation_kind = (
        "bound_cost_model"
        if config.get("inner_initialization", "warm") == "warm"
        else "warm_start_reference_model"
    )
    rng = np.random.default_rng(config["seed"] + 100003)
    if logistic:
        inner = None
        preconditioner_condition = 1.0
    else:
        inner = Inner(
            p,
            c,
            config["inner"],
            rng,
            config.get("block_size", 32),
            config.get("rank", 64),
            gd_factor=config.get("gd_factor", 1.0),
        )
        preconditioner_condition = getattr(inner, "condition_bound", 1.0)
    initial_rng = deepcopy(rng.bit_generator.state)
    initial_cache = None if inner is None else inner.cache.copy()
    initial_counts = (
        None
        if inner is None
        else (inner.products, inner.row_updates, inner.block_factorizations)
    )

    def finish(metadata):
        if not return_inner:
            return config, metadata
        if inner is not None:
            # Restore the exact post-construction random stream/cache so that
            # calibration does not warm the solve or alter block sampling.
            rng.bit_generator.state = initial_rng
            inner.cache = initial_cache
            inner.products, inner.row_updates, inner.block_factorizations = (
                initial_counts
            )
            metadata["tuning_seconds"] -= inner.setup_seconds
        metadata["setup_reused_within_run"] = inner is not None
        return config, metadata, inner

    bound = constants(
        p,
        c,
        config["inner"],
        config.get("block_size", 32),
        preconditioner_condition,
        config.get("gd_factor", 1.0),
    )
    if bound is None:
        if config["sigma"] == "optimal" or config["stride"] == "optimal":
            raise ValueError("no verified contraction model for this solver")
        return finish(
            dict(
                tuning_seconds=perf_counter() - started,
                recommendation_kind="empirical_grid_only",
            )
        )
    alpha, c0, deterministic = bound
    alpha = min(max(alpha, 1e-15), 1 - 1e-15)
    x = rng.normal(size=p.n) / np.sqrt(p.n)
    if inner:
        inner.start(x, -p.q)
    tick = perf_counter()
    for _ in range(3):
        if inner:
            inner.step()
        else:
            x -= (
                config.get("gd_factor", 1.0)
                * 2
                / (p.L + p.mu + 2 * c)
                * (p.gradient(x) + c * x)
            )
    c_in = max((perf_counter() - tick) / 3, 1e-12)
    tick = perf_counter()
    for _ in range(3):
        g = p.gradient(x)
        z = p.project(x - g / c)
        np.linalg.norm(z + g / c)
        np.linalg.norm(x + g / c)
    c_test = max((perf_counter() - tick) / 3, 1e-12)
    stride = 1 if config["stride"] == "optimal" else int(config["stride"])
    sigma_star = theory.optimal_sigma(alpha, c0, stride, deterministic)
    if config["sigma"] == "optimal":
        config["sigma"] = sigma_star
    if config["stride"] == "optimal":
        config["stride"] = theory.optimal_b_exact(
            c_test, c_in, float(config["sigma"]), c0, alpha
        )
    return finish(
        dict(
            alpha=alpha,
            c0=c0,
            deterministic=deterministic,
            sigma_star=sigma_star,
            stride_star=config["stride"],
            c_in_seconds=c_in,
            c_test_seconds=c_test,
            tuning_seconds=perf_counter() - started,
            recommendation_kind=recommendation_kind,
        )
    )
