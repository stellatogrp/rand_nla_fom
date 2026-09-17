# Preconditioned primal--dual DRS experiments

The core package implements equations (58)--(60) and the relaxed relative-error
criterion in Theorem 5 of `Rand_LA_FOM-4`. The metric/preconditioner is

```text
M = diag(I / gamma_x, I / gamma_lambda).
```

`randomized_sketch_descent.drs` provides the public algorithm. Its private
`_linear_solvers` module contains GMRES for the coupled primal--dual system and
CG and block Kaczmarz for the positive-definite Schur complement. All are warm
started and checked after every inner iteration using
`||epsilon||_M <= sigma ||s||_M`, with
`0 <= sigma < (2 - theta) / 2`.

`problems.py` contains shared problem and reference-solution primitives.
`standard.py` constructs equality-constrained QP and standard-form LP studies.
`sweep_problems.py` builds the controlled-spectrum problem families used by the
scalar `gamma` and `sigma` sweeps, at both the small sizes and the large sizes
whose exact primal--dual certificates remove the need for a reference solver.
`run_experiments.py` only handles command dispatch.

All experiments use one entry point. From the repository root:

```bash
.venv/bin/python -m scripts.run_experiments standard
.venv/bin/python -m scripts.run_experiments lp
.venv/bin/python -m scripts.run_experiments inequality-qp
.venv/bin/python -m scripts.run_experiments gamma-sweep
.venv/bin/python -m scripts.run_experiments sigma-sweep
.venv/bin/python -m scripts.run_experiments lp-sigma-sweep
.venv/bin/python -m scripts.run_experiments inequality-gamma-sweep
```

The standard study writes plots and its raw summary table under
`figures/standard/`.

The `lp` study tunes dimensionless preconditioner scales and `(sigma, theta)`
on a proxy LP, transfers the best settings using the singular values of the new
constraint matrix and the RMS objective coefficient, and compares them with
the original fixed baseline. Outputs go to `figures/lp_tuning/`.

The `inequality-qp` study solves `min x.T Q x` subject to
`A x + s = b, s >= 0`. It tunes on `n=100, m=20`, transfers parameters to
`n=500, m=100`, and writes convergence plots plus raw results under
`figures/inequality_qp_tuning/`. It also compares fixed parameters
with progress-based adaptation of `sigma` and `theta`. The adaptive policy
uses only step norms, feasibility, recent contraction, and inner iteration
counts; it does not use the reference optimum.

The `gamma-sweep` study imposes `gamma_x = gamma_lambda = gamma` and compares
outer and total inner iterations across equality QPs, LPs, and slack-form
inequality QPs with moderately conditioned (`kappa(A) = 10`) and
ill-conditioned (`kappa(A) = 100`) constraint matrices. Outputs go
to `figures/gamma_sweep/`. The sweep uses 25 logarithmically spaced
values of `gamma`; its focused inequality-QP plot shows both outer iterations
and total inner CG iterations for the two condition numbers. The focused
`inequality-gamma-sweep` runs this comparison at `n=500, m=100` without
rerunning the other problem classes. Its focused figure also shows normalized
objective error plus feasibility versus outer and cumulative inner iterations
for the common `gamma=1` runs.

The `sigma-sweep` study compares 21 admissible values from `0` through `0.49`
with `theta = gamma_x = gamma_lambda = 1`. For each setting, it records the
outer and cumulative inner block-Kaczmarz iterations at the first iterate whose
normalized objective error plus feasibility is at most `1e-7`. Its plot and CSV table are
written to `figures/sigma_sweep/`. Kaczmarz blocks contain approximately 5% of
the Schur-complement rows. All three problem families use 500 constraints and
1,000 primal variables; the inequality-QP solver augments its `500 x 1000`
constraint matrix to `500 x 1500` with slack variables. Each result is
checkpointed and includes solver time, dimensions, seed, block size, and
termination data. The focused `lp-sigma-sweep` command reruns only the two LP
cases and integrates them into the shared outputs.

The `solver-sweep` study isolates the inner solver. It fixes
`gamma_x = gamma_lambda = 1`, `theta = 1`, and `sigma = 0.2`, and for each of
the three problem families at both condition numbers it draws 100 independent
instances with `m = 100` constraints and `n = 500` primal variables (the
inequality QP augments its `100 x 500` matrix to `100 x 600` with slacks).
Every instance is solved four times, once per inner solver: CG and block
Kaczmarz and randomized Kaczmarz on the Schur complement, and GMRES on the
coupled system. Each run gets the same inner budget of `1000 m` iterations and
the same target, the first iterate whose normalized objective error plus
feasibility is at most `1e-7`; the recorded outer and cumulative inner
iteration counts are then aggregated over instances. `solver_comparison.pdf`
shows means with one-standard-deviation error bars,
`solver_distributions.pdf` the underlying per-instance distributions,
`summary.csv` the aggregate table, and `results.csv` every instance. Instances
run on a process pool with one BLAS thread each, so the recorded times are
comparable across solvers but measured under load.
