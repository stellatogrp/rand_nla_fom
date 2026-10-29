"""Constants and bounds from ``tex/General_inexactness.tex``.

Names follow the paper, except that the stride ``s`` of the paper is ``b``
here and in every result table.  ``ratio`` is ``gamma * L`` for the proximal point
method (Proposition 5.1) and ``L / c`` for ADMM (Proposition 5.2).  The
inner-solver constants ``alpha`` and ``C_0`` are those of gradient descent on
the proximal subproblem (``inner_constants``, the PPM study) or of GMRES on
the linear system of the ADMM ``x``-subproblem (``gmres_constants``).

Every tolerance argument of the bounds is ``gamma * delta``: the terminal
test ``||z|| <= delta`` enters the bounds through ``||gamma z|| <= gamma delta``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.optimize import brentq

# The breakpoints of the warm-start constant ``G_sigma`` of Lemma 4.1.
SIGMA_1 = (math.sqrt(13.0) - 1.0) / 4.0
SIGMA_2 = (math.sqrt(7.0) - 1.0) / 2.0


def inner_constants(ratio: float) -> tuple[float, float]:
    """``alpha = (r/(r+2))^2`` and ``C_0 = (1+r)^2`` of Propositions 5.1 and 5.2."""
    alpha = (ratio / (ratio + 2.0)) ** 2
    c0 = (1.0 + ratio) ** 2
    return alpha, c0


def gmres_constants(ratio: float) -> tuple[float, float]:
    """``sqrt(alpha) = (sqrt(kappa) - 1)/(sqrt(kappa) + 1)`` and ``C_0 = 4`` for GMRES.

    ``Q`` is positive semidefinite with ``||Q|| = L``, so the spectrum of
    ``Q + cI`` lies in ``[c, L + c]`` and its condition number is at most
    ``kappa = 1 + L/c = 1 + ratio``.  GMRES minimizes the residual over the
    Krylov space, so the Chebyshev polynomial bounds it:
    ``||r_l|| <= 2 alpha^{l/2} ||r_0||`` on ``(Q + cI) x = rhs``.
    The graph residual of the ADMM ``x``-subproblem is ``r / c``, so the
    inner-contraction assumption holds deterministically with these constants.
    """
    root = math.sqrt(1.0 + ratio)
    root_alpha = (root - 1.0) / (root + 1.0)
    return root_alpha**2, 4.0


def warm_start_constant(sigma: float) -> float:
    """``G_sigma`` of Lemma 4.1: the warm start is within ``G_sigma ||R(w_k)||``.

    Piecewise in ``sigma``, continuous and increasing, infinite at ``sigma = 1``.
    """
    if sigma >= 1.0:
        return math.inf
    if sigma <= SIGMA_1:
        return 1.0 + 2.0 * sigma
    if sigma <= SIGMA_2:
        return 2.0 * math.sqrt(3.0 * (1.0 - sigma**2) / (3.0 - 4.0 * sigma**2))
    return (2.0 * sigma - 1.0) / (1.0 - sigma)


def log_d(sigma: float, c0: float) -> float:
    """``log D(sigma)`` with ``D = (1+sigma)^2 C_0 G_sigma^2 / sigma^2`` (Cor. 4.1.1).

    ``D`` is infinite at ``sigma = 1``, where the bounds built on it do not
    apply; ``inf`` is returned there rather than raising.
    """
    if sigma >= 1.0:
        return math.inf
    return (
        math.log(c0)
        + 2.0 * math.log1p(sigma)
        + 2.0 * math.log(warm_start_constant(sigma))
        - 2.0 * math.log(sigma)
    )


def check_constant(alpha: float, b: int, deterministic: bool = True) -> float:
    """The constant ``chi`` of Proposition 5.1 for a stride ``b``.

    ``b log(1/alpha)`` for the deterministic bound (Theorem 4.3) and
    ``b log(1/alpha) / (1 - alpha^b)`` for the expected one (Theorem 4.2).
    """
    log_inverse = math.log(1.0 / alpha)
    if deterministic:
        return b * log_inverse
    return b * log_inverse / (1.0 - alpha**b)


def g_coefficient(sigma: float, chi: float, c0: float) -> float:
    """``g(sigma) = (1+sigma)/(1-sigma) (chi + log D(sigma))`` of Proposition 5.1."""
    return (1.0 + sigma) / (1.0 - sigma) * (chi + log_d(sigma, c0))


def optimal_sigma(alpha: float, c0: float, b: int, deterministic: bool = True) -> float:
    """The unique minimizer ``sigma*`` of ``g`` (Proposition 5.1).

    It is the root in ``(0, sigma_1)`` of equation (optimal-sigma),
    ``1/sigma = 1 + psi(sigma) + chi + log((1+sigma)^2 (1+2 sigma)^2 C_0 / sigma^2)``
    with ``psi(sigma) = 2(1-sigma^2)/(1+2 sigma)``; the left-hand side minus the
    right is strictly decreasing on ``(0, 1)``, so the root is bracketed there.
    """
    chi = check_constant(alpha, b, deterministic)

    def h(sigma: float) -> float:
        psi = 2.0 * (1.0 - sigma**2) / (1.0 + 2.0 * sigma)
        log_d_tilde = (
            math.log(c0)
            + 2.0 * math.log1p(sigma)
            + 2.0 * math.log1p(2.0 * sigma)
            - 2.0 * math.log(sigma)
        )
        return 1.0 / sigma - 1.0 - psi - chi - log_d_tilde

    return float(brentq(h, 1e-12, 1.0 - 1e-12, xtol=1e-15, rtol=1e-14))


def sigma_upper_bound(
    alpha: float, c0: float, b: int, deterministic: bool = True
) -> float:
    """``1 / (3/2 + chi + log C_0)``, the bound on ``sigma*`` in Proposition 5.1."""
    return 1.0 / (1.5 + check_constant(alpha, b, deterministic) + math.log(c0))


def log_plus(value: float) -> float:
    """``max(0, log(value))``; zero for non-positive arguments."""
    return max(0.0, math.log(value)) if value > 0 else 0.0


def beta(
    sigma: float, c0: float, e0_norm: float, residual_norm: float, gamma_delta: float
) -> float:
    """``(1+sigma)^2 C_0 ||e^0||^2 / (sigma^2 max(||R(w_k)||^2, (gamma delta)^2))``.

    ``beta_k`` of Theorems 4.1 and 4.3.
    """
    denominator = sigma**2 * max(residual_norm**2, gamma_delta**2)
    return (1.0 + sigma) ** 2 * c0 * e0_norm**2 / denominator


def per_step_bound(beta_value: float, alpha: float, b: int) -> float:
    """Theorem 4.3: ``N_k <= b ceil(log+ beta_k / (b log(1/alpha)))``."""
    return b * math.ceil(log_plus(beta_value) / (b * math.log(1.0 / alpha)))


def expected_per_step_bound(beta_value: float, alpha: float, b: int) -> float:
    """Theorem 4.1: ``E[N_k] <= log+ beta_k / log(1/alpha) + b / (1 - alpha^b)``."""
    return log_plus(beta_value) / math.log(1.0 / alpha) + b / (1.0 - alpha**b)


def initial_term(r0_norm: float, gamma_delta: float, alpha: float) -> float:
    """``log(max(||R(w_0)||^2, (gamma delta)^2) / (gamma delta)^2) / log(1/alpha)``."""
    return math.log(max(r0_norm**2, gamma_delta**2) / gamma_delta**2) / math.log(
        1.0 / alpha
    )


def total_bound_deterministic(
    beta0: float,
    k_delta: int,
    sigma: float,
    c0: float,
    alpha: float,
    b: int,
    r0_norm: float,
    gamma_delta: float,
) -> float:
    """Equation (deterministic-total) of Theorem 4.3."""
    log_inverse = math.log(1.0 / alpha)
    return (
        per_step_bound(beta0, alpha, b)
        + k_delta * (b + log_d(sigma, c0) / log_inverse)
        + initial_term(r0_norm, gamma_delta, alpha)
    )


def total_bound_expected(
    beta0: float,
    k_delta: float,
    sigma: float,
    c0: float,
    alpha: float,
    b: int,
    r0_norm: float,
    gamma_delta: float,
) -> float:
    """Theorem 4.2 with ``E[K_delta]`` replaced by the observed ``K_delta``.

    ``(K_delta + 1) b/(1-alpha^b) + (K_delta log D + log+ beta_0)/log(1/alpha)``
    plus the initial term.
    """
    log_inverse = math.log(1.0 / alpha)
    return (
        (k_delta + 1) * b / (1.0 - alpha**b)
        + (k_delta * log_d(sigma, c0) + log_plus(beta0)) / log_inverse
        + initial_term(r0_norm, gamma_delta, alpha)
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
    gamma_delta: float,
) -> float:
    """``N'`` of Lemma 4.2."""
    log_inverse = math.log(1.0 / alpha)
    return (k_up * log_d(sigma, c0) + logp_beta0) / log_inverse + initial_term(
        r0_norm, gamma_delta, alpha
    )


def b_star(
    c_test: float, c_in: float, n_prime_value: float, k_up: float, alpha: float
) -> float:
    """``sqrt(c_test (N' + (K_up+1)/log(1/alpha)) / (c_in (K_up + 1)))``: Lemma 4.2."""
    return math.sqrt(
        c_test
        * (n_prime_value + (k_up + 1.0) / math.log(1.0 / alpha))
        / (c_in * (k_up + 1.0))
    )


def b_star_asymptotic(
    c_test: float, c_in: float, sigma: float, c0: float, alpha: float
) -> float:
    """The ``K_up -> infinity`` limit of ``b*``.

    There ``N' / (K_up + 1) -> log D / log(1/alpha)``, so
    ``b* -> sqrt(c_test (1 + log D) / (c_in log(1/alpha)))`` and no bound on
    ``K_delta`` is needed.
    """
    log_inverse = math.log(1.0 / alpha)
    return math.sqrt(c_test * (1.0 + log_d(sigma, c0)) / (c_in * log_inverse))


def best_integer_b(b_star_value: float, cost: Callable[[int], float]) -> int:
    """The integer ``b >= 1`` among ``floor(b*)`` and ``ceil(b*)`` with lower cost."""
    candidates = {max(1, math.floor(b_star_value)), max(1, math.ceil(b_star_value))}
    return min(candidates, key=cost)


def per_outer_overhead(alpha: float, b: int, relaxed: bool = True) -> float:
    """The ``b``-dependent inner iterations per outer iteration of Theorem 4.2.

    ``b / (1 - alpha^b)``, or its relaxation ``b + 1/log(1/alpha)`` used in
    Lemma 4.2 (the two differ by at most ``b/2``, Remark 4.4).
    """
    if relaxed:
        return b + 1.0 / math.log(1.0 / alpha)
    return b / (1.0 - alpha**b)


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

    With ``W(b)`` the inner work per outer iteration, ``per_outer_overhead``
    plus ``log D / log(1/alpha)``, the cost of the inner iterations and of the
    ``1 + W/b`` checks is ``c_in W + c_test (1 + W/b)``; the terms that do not
    depend on ``b`` (``c_out``, ``c_init``) are left out.
    """
    work = per_outer_overhead(alpha, b, relaxed) + log_d(sigma, c0) / math.log(
        1.0 / alpha
    )
    return c_in * work + c_test * (1.0 + work / b)


def optimal_b_exact(
    c_test: float, c_in: float, sigma: float, c0: float, alpha: float, b_max: int = 1000
) -> int:
    """Remark 4.4: the integer minimizer of the unrelaxed cost, found by scanning."""
    costs = [
        b_cost_per_outer(b, c_test, c_in, sigma, c0, alpha, relaxed=False)
        for b in range(1, b_max + 1)
    ]
    return 1 + min(range(b_max), key=costs.__getitem__)
