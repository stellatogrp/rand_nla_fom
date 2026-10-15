"""Constants and bounds from ``tex/General_inexactness.tex``.

Names follow the paper.  ``ratio`` is ``gamma * L`` for the proximal point
method (Proposition 5.1) and ``L / c`` for ADMM (Proposition 5.2); both give
the same inner-solver constants ``alpha`` and ``C_0`` for gradient descent on
the proximal subproblem.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.optimize import brentq


def inner_constants(ratio: float) -> tuple[float, float]:
    """``alpha = (r/(r+2))^2`` and ``C_0 = (1+r)^2`` of Propositions 5.1 and 5.2."""
    alpha = (ratio / (ratio + 2.0)) ** 2
    c0 = (1.0 + ratio) ** 2
    return alpha, c0


def log_d(sigma: float, c0: float) -> float:
    """``log D(sigma)`` with ``D = (1+sigma)^4 C_0 / (sigma^2 (1-sigma)^2)``."""
    return (
        math.log(c0)
        + 4.0 * math.log1p(sigma)
        - 2.0 * math.log(sigma)
        - 2.0 * math.log1p(-sigma)
    )


def check_constant(alpha: float, b: int, deterministic: bool = True) -> float:
    """The constant ``c`` of Proposition 5.1 for a check interval ``b``.

    ``b log(1/alpha)`` for the deterministic bound (Theorem 4.3) and
    ``b (2 - alpha^b) log(1/alpha) / (1 - alpha^b)`` for the expected one
    (Theorem 4.2).
    """
    log_inverse = math.log(1.0 / alpha)
    if deterministic:
        return b * log_inverse
    power = alpha**b
    return b * (2.0 - power) * log_inverse / (1.0 - power)


def g_coefficient(sigma: float, c: float, c0: float) -> float:
    """``g(sigma) = (1+sigma)/(1-sigma) (c + log D(sigma))`` of Proposition 5.1."""
    return (1.0 + sigma) / (1.0 - sigma) * (c + log_d(sigma, c0))


def optimal_sigma(alpha: float, c0: float, b: int, deterministic: bool = True) -> float:
    """The unique root in ``(0, 1)`` of ``1/sigma = 3 + c + log D(sigma)``."""
    c = check_constant(alpha, b, deterministic)

    def h(sigma: float) -> float:
        return 1.0 / sigma - 3.0 - c - log_d(sigma, c0)

    return float(brentq(h, 1e-12, 1.0 - 1e-12, xtol=1e-15, rtol=1e-14))


def sigma_upper_bound(
    alpha: float, c0: float, b: int, deterministic: bool = True
) -> float:
    """``1 / (3 + c + log C_0)``, the bound on ``sigma*`` in Proposition 5.1."""
    return 1.0 / (3.0 + check_constant(alpha, b, deterministic) + math.log(c0))


def log_plus(value: float) -> float:
    """``max(0, log(value))``; zero for non-positive arguments."""
    return max(0.0, math.log(value)) if value > 0 else 0.0


def beta(
    sigma: float, c0: float, e0_norm: float, residual_norm: float, delta: float
) -> float:
    """``beta_k = (1+sigma)^2 C_0 ||e^0||^2 / (sigma^2 max(||R(w_k)||^2, delta^2))``."""
    denominator = sigma**2 * max(residual_norm**2, delta**2)
    return (1.0 + sigma) ** 2 * c0 * e0_norm**2 / denominator


def per_step_bound(beta_value: float, alpha: float, b: int) -> float:
    """Theorem 4.3: ``N_k <= b ceil(log+ beta_k / (b log(1/alpha)))``."""
    return b * math.ceil(log_plus(beta_value) / (b * math.log(1.0 / alpha)))


def expected_per_step_bound(beta_value: float, alpha: float, b: int) -> float:
    """Theorem 4.1: the per-step bound plus the tail term ``b / (1 - alpha^b)``."""
    return per_step_bound(beta_value, alpha, b) + b / (1.0 - alpha**b)


def initial_term(r0_norm: float, delta: float, alpha: float) -> float:
    """``log(max(||R(w_0)||^2, delta^2) / delta^2) / log(1/alpha)``."""
    return math.log(max(r0_norm**2, delta**2) / delta**2) / math.log(1.0 / alpha)


def total_bound_deterministic(
    beta0: float,
    k_delta: int,
    sigma: float,
    c0: float,
    alpha: float,
    b: int,
    r0_norm: float,
    delta: float,
) -> float:
    """The first inequality of (deterministic-total) in Theorem 4.3."""
    log_inverse = math.log(1.0 / alpha)
    return (
        per_step_bound(beta0, alpha, b)
        + k_delta * (b + log_d(sigma, c0) / log_inverse)
        + initial_term(r0_norm, delta, alpha)
    )


def total_bound_expected(
    beta0: float,
    k_delta: float,
    sigma: float,
    c0: float,
    alpha: float,
    b: int,
    r0_norm: float,
    delta: float,
) -> float:
    """Theorem 4.2 with ``E[K_delta]`` replaced by the observed ``K_delta``."""
    log_inverse = math.log(1.0 / alpha)
    power = alpha**b
    c1 = b * (2.0 - power) / (1.0 - power)
    return (
        (k_delta + 1) * c1
        + (k_delta * log_d(sigma, c0) + log_plus(beta0)) / log_inverse
        + initial_term(r0_norm, delta, alpha)
    )


def k_delta_plus(sigma: float, dist0: float, gamma: float, delta: float) -> float:
    """Theorem 3.1: ``floor((1+sigma) dist^2 / (gamma^2 delta^2 (1-sigma)))``."""
    return math.floor((1.0 + sigma) * dist0**2 / (gamma**2 * delta**2 * (1.0 - sigma)))


def n_prime(
    k_up: float,
    sigma: float,
    c0: float,
    alpha: float,
    logp_beta0: float,
    r0_norm: float,
    delta: float,
) -> float:
    """``N'`` of Lemma 4.2."""
    log_inverse = math.log(1.0 / alpha)
    return (k_up * log_d(sigma, c0) + logp_beta0) / log_inverse + initial_term(
        r0_norm, delta, alpha
    )


def b_star(
    c_test: float, c_in: float, n_prime_value: float, k_up: float, alpha: float
) -> float:
    """``b* = sqrt(c_test N' / (c_in (K_up + 1)(1 + 1/(1-alpha))))`` of Lemma 4.2."""
    return math.sqrt(
        c_test * n_prime_value / (c_in * (k_up + 1.0) * (1.0 + 1.0 / (1.0 - alpha)))
    )


def b_star_asymptotic(
    c_test: float, c_in: float, sigma: float, c0: float, alpha: float
) -> float:
    """The ``K_up -> infinity`` limit of ``b*``.

    There ``N' / (K_up + 1) -> log D / log(1/alpha)``, so no bound on
    ``K_delta`` is needed.
    """
    per_outer = log_d(sigma, c0) / math.log(1.0 / alpha)
    return math.sqrt(c_test * per_outer / (c_in * (1.0 + 1.0 / (1.0 - alpha))))


def best_integer_b(b_star_value: float, cost: Callable[[int], float]) -> int:
    """The integer ``b >= 1`` among ``floor(b*)`` and ``ceil(b*)`` with lower cost."""
    candidates = {max(1, math.floor(b_star_value)), max(1, math.ceil(b_star_value))}
    return min(candidates, key=cost)


def tail_factor(alpha: float, b: int) -> float:
    """``(2 - alpha^b) / (1 - alpha^b) = 1 + 1/(1 - alpha^b)``: Theorem 4.2's tail."""
    power = alpha**b
    return (2.0 - power) / (1.0 - power)


def b_cost_per_outer(
    b: int,
    c_test: float,
    c_in: float,
    sigma: float,
    c0: float,
    alpha: float,
    relaxed: bool = True,
) -> float:
    """Lemma 4.2's cost per outer iteration as a function of ``b``.

    With ``F = 1 + 1/(1-alpha)`` (the relaxation discussed in Remark 4.4) or
    ``F = 1 + 1/(1-alpha^b)`` (the exact tail factor of Theorem 4.2), the
    terms of the cost bound that depend on ``b`` are, per outer iteration,
    ``c_in F b + c_test F + c_test (log D / log(1/alpha)) / b``.
    """
    factor = 1.0 + 1.0 / (1.0 - alpha) if relaxed else tail_factor(alpha, b)
    per_outer = log_d(sigma, c0) / math.log(1.0 / alpha)
    return c_in * factor * b + c_test * factor + c_test * per_outer / b


def optimal_b_exact(
    c_test: float, c_in: float, sigma: float, c0: float, alpha: float, b_max: int = 1000
) -> int:
    """Remark 4.4: the integer minimizer of the unrelaxed cost, found by scanning."""
    costs = [
        b_cost_per_outer(b, c_test, c_in, sigma, c0, alpha, relaxed=False)
        for b in range(1, b_max + 1)
    ]
    return 1 + min(range(b_max), key=costs.__getitem__)
