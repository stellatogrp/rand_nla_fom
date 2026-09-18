"""Small deterministic numerical fixtures."""

import numpy as np


def equality_qp(seed: int = 4, constraints: int = 4, variables: int = 11):
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((constraints, variables))
    factor = rng.standard_normal((variables, variables))
    hessian = factor.T @ factor + np.eye(variables)
    linear_term = rng.standard_normal(variables)
    rhs = rng.standard_normal(constraints)
    kkt = np.block(
        [
            [hessian, matrix.T],
            [matrix, np.zeros((constraints, constraints))],
        ]
    )
    solution = np.linalg.solve(kkt, np.r_[-linear_term, rhs])[:variables]
    return matrix, rhs, hessian, linear_term, solution
