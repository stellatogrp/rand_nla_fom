# Randomized sketch descent

Experiments for the preconditioned primal-dual Douglas--Rachford method in
Section 5 of `Rand_LA_FOM-4`, including equality-constrained quadratic
programs, standard-form linear programs, and slack-form inequality QPs.

## Layout

```text
src/       Installable solver package
scripts/   Reproducible experiment entry points and helpers
tests/     Numerical regression tests for the active solver
figures/   Generated plots and CSV results (git-ignored)
```

See [`scripts/README.md`](scripts/README.md) for the method summary, study
descriptions, and output locations.

The reusable API is independent of the experiment scripts:

```python
from randomized_sketch_descent import (
    DRSResult,
    primal_dual_drs,
    quadratic_prox,
)
```

Equation (58) is solved inexactly by one of three warm-started inner solvers,
selected with `linear_solver`:

- `schur_cg` (default), conjugate gradients on the positive-definite Schur
  complement `I + gamma_x*gamma_lambda*A*A.T`;
- `schur_block_kaczmarz`, block Kaczmarz on the rows of the same system, using
  a cached random row partition controlled by `block_size` and `random_seed`;
- `coupled_gmres`, unrestarted GMRES on the full primal--dual system.

Each is stopped by the relative-error test `||epsilon||_M <= sigma ||s||_M`
checked after every inner iteration.

## Environment

Dependencies are declared in `pyproject.toml`. Create the environment with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra experiments
```

The base package depends only on NumPy; the `experiments` extra adds SciPy and
Matplotlib.

## Usage

Run an individual study from the repository root:

```bash
.venv/bin/python -m scripts.run_experiments standard
.venv/bin/python -m scripts.run_experiments lp
.venv/bin/python -m scripts.run_experiments inequality-qp
.venv/bin/python -m scripts.run_experiments gamma-sweep
.venv/bin/python -m scripts.run_experiments sigma-sweep
.venv/bin/python -m scripts.run_experiments solver-sweep
.venv/bin/python -m scripts.run_experiments lp-sigma-sweep
.venv/bin/python -m scripts.run_experiments inequality-gamma-sweep
```

Use `all` to run every study except the focused LP sigma, inequality gamma, and
solver sweeps. The `sigma-sweep` study is the long one: it checkpoints every
`(problem, conditioning, sigma)` result to `figures/sigma_sweep/results.csv`
and skips completed settings when rerun, so it can be resumed.

The `solver-sweep` study holds `gamma_x = gamma_lambda`, `theta`, and `sigma`
fixed and varies only the inner solver, over 100 independent problem instances
of each family at `m=100, n=500`. It checkpoints per
`(problem, conditioning, instance, solver)` to
`figures/solver_sweep/results.csv` and runs its instances on a process pool, so
it is also resumable.

Run the regression tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
