# Solver, experiments, and plots

```text
src/            Installable solver package (NumPy only), imported as rand_nla_fom
experiments/    Study drivers: compute, write CSVs, never plot
plots/          Figure drivers: read CSVs, never compute
tests/          Regression tests for the solver and the CSV contract
```

The package sits flat in `src/`; `package-dir` in `pyproject.toml` maps that
directory to the import name, so it is `import rand_nla_fom` regardless.

Results land outside this directory, under `../results/<study>/`, because they
are generated and git-ignored.  Redirect them with `--results-dir` or the
`RAND_NLA_FOM_RESULTS` environment variable.

## The method

The package implements equations (58)--(60) and the relaxed relative-error
criterion of Theorem 5 in `Rand_LA_FOM-4`.  The metric is

```text
M = diag(I / gamma_x, I / gamma_lambda).
```

`rand_nla_fom.drs` (`src/drs.py`) provides the public algorithm.  Its private
`_linear_solvers` module contains GMRES for the coupled primal--dual system
and CG, block Kaczmarz, and randomized Kaczmarz for the positive-definite
Schur complement `I + gamma_x*gamma_lambda*A*A.T`.  All are warm started and
checked after every inner iteration against `||epsilon||_M <= sigma ||s||_M`,
which Theorem 5 admits for `0 <= sigma < (2 - theta) / 2`.

```python
from rand_nla_fom import DRSResult, primal_dual_drs, quadratic_prox

result = primal_dual_drs(prox_f, A, b, linear_solver="schur_cg", sigma=0.2)
```

`linear_solver` selects the inner method: `schur_cg` (default),
`schur_block_kaczmarz` (a cached random row partition, sized by `block_size`
and `random_seed`), `schur_randomized_kaczmarz` (single rows drawn with
probability proportional to their squared norms), or `coupled_gmres`
(unrestarted GMRES on the full primal--dual system).

## Environment

```bash
uv sync                # package + scipy, matplotlib, pytest, ruff
uv sync --no-dev       # the NumPy-only solver alone
```

## Running a study

Every study is a module under `experiments/` with its own flags, reachable
either directly or through the dispatcher:

```bash
uv run python -m experiments.run_experiments theta-sweep --trials 5
uv run python -m experiments.theta_sweep --trials 5        # the same thing
uv run python -m experiments.theta_sweep --help            # its flags
```

`--plot` draws the study's figures from the CSVs it just wrote; `all` runs
the broad studies in sequence.  Plotting is otherwise a separate step, so a
figure can be redrawn without recomputing anything:

```bash
uv run python -m plots.make_plots                 # every figure it can draw
uv run python -m plots.make_plots theta-sweep     # just one
```

`make_plots` skips a study whose CSVs are missing and says so; `--strict`
turns that into an error.

## What each study writes

Every study directory holds at most three tables.

| File | Contents |
| --- | --- |
| `results.csv` | one row per run: the settings, and the scalars measured from it |
| `histories.csv` | one row per run and outer iteration, for convergence curves |
| `summary.csv` | aggregates over `results.csv`, where a study has any |

The identifying columns of `histories.csv` match those of `results.csv`, so
the two join.  Reading coerces every field back to `bool`, `int`, `float`, or
`str`; an empty field reads as `nan`, because a missing measurement is
numeric rather than categorical.

The long studies (`sigma-sweep`, `theta-sweep`, `solver-sweep`) append to
`results.csv` as each setting finishes and skip settings already in it, so
they resume after an interruption.  `--force` recomputes instead.

## The studies

`standard` compares the metric, the inner tolerance, and the relaxation on a
small equality QP and a small standard-form LP, then compares CG against
GMRES on both.

`lp` tunes dimensionless preconditioner scales and `(sigma, theta)` on a proxy
LP, transfers the best settings using the singular values of the new
constraint matrix and the RMS objective coefficient, and compares them with a
fixed baseline.

`inequality-qp` solves `min x'Qx` subject to `Ax + s = b, s >= 0`, tunes on a
proxy, transfers to a larger instance, and compares fixed parameters with
progress-based adaptation of `sigma` and `theta`.  The adaptive policy uses
only step norms, feasibility, recent contraction, and inner iteration counts;
it never sees the reference optimum.

`gamma-sweep` imposes `gamma_x = gamma_lambda = gamma` and compares outer and
total inner iterations across equality QPs, LPs, and slack-form inequality QPs
at two condition numbers.  `inequality-gamma-sweep` reruns only the large
inequality QP and additionally records the convergence traces of its
`gamma = 1` runs.

`sigma-sweep` sweeps the admissible inner tolerances at
`theta = gamma_x = gamma_lambda = 1`, recording the outer and cumulative inner
block-Kaczmarz iterations at the first iterate reaching the target accuracy.
Restrict it to one family with `--kinds`.

`theta-sweep` fixes `gamma_x = gamma_lambda = 1` and sweeps the relaxation,
solving equation (58) by randomized Kaczmarz.  Because Theorem 5 couples the
parameters through `sigma < (2 - theta) / 2`, it runs two tolerance policies:
a `fixed` `sigma` admissible at every `theta`, and a `frontier` `sigma` that
tracks the admissibility boundary.  Each setting is repeated over `--trials`
trials that redraw the singular vectors, the interior singular values, and the
Kaczmarz row stream; the extreme singular values stay pinned, so every trial
has exactly the stated `kappa(A)`.  The figure shows the median with whiskers
spanning the observed range.

`solver-sweep` holds the parameters fixed and varies only the inner solver,
over many independent instances of each family, so the spread over instances
can be shown with error bars.  Instances run on a process pool with one BLAS
thread each, so the recorded times are comparable across solvers but measured
under load.  By default the instances of one family share a fixed geometric
singular-value ladder and differ only in their singular vectors and solution
draws; `solver-sweep-randomized` pins the extremes, leaving `kappa(A)`
unchanged, and redraws the interior log-uniformly per instance.  Its results
go to a separate directory and carry a `randomize_spectrum` column.

## Adding a study

1. Write `experiments/<name>.py` with `build_parser()`, `run_study(args)`,
   and a `main()`; use `_results.write_rows` for its tables and
   `_metrics.history_rows` if its figures need convergence curves.
2. Write `plots/<name>.py` with `make(results_dir)` returning the paths it
   wrote, reading only through `plots._data`.
3. Register both in `experiments/run_experiments.STUDIES` and
   `plots/make_plots.FIGURES`.
