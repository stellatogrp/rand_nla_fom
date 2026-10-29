# Numerical experiments for `tex/General_inexactness.tex`

Experiments that test the theorems of the inexact preconditioned proximal
point framework (Algorithm 1 of the paper) on two problem families:

| Study | Problem | Method | Section of the paper |
| --- | --- | --- | --- |
| `admm_qp` | `min 1/2 x'Qx + q'x  s.t.  Ax = b`, `n = 2000`, `m = 600`, `kappa(Q) = 10^4` | inexact ADMM (Algorithm 3) with GMRES inside, against exact ADMM | Proposition 5.2, with the GMRES constants below |
| `ppm_huber` | Huber-loss fitting, `n = 2000` features, `m = 3000` observations, 10% gross outliers | inexact proximal point with gradient descent inside, against plain gradient descent | Proposition 5.1 |

In words, the ADMM study solves a linearly constrained quadratic program:
it minimizes the strongly convex quadratic `1/2 x'Qx + q'x` over the affine
set `{x : Ax = b}` of 600 equality constraints on 2000 variables, and the
inexactness enters through the linear solve of the `x`-subproblem, the
system `(Q + cI) x = c w_k - q`, which GMRES solves.  The
PPM study solves linear regression with the Huber loss: it fits `x` to
3000 noisy observations `b ~ Ax`, 10% of them gross outliers, by
minimizing `sum_i huber_M((Ax - b)_i)` with threshold `M = 0.5`, an
objective that is smooth and convex but not strongly convex, and the
inexactness enters through the gradient-descent solve of each proximal
subproblem.  These are the two instances of Algorithm 1 worked out in
Section 5 of the paper.  The PPM study also runs plain gradient descent on
the same objective, with stepsize `1/L` and the same stopping test, as the
classical baseline.

Both use the stopping tolerance `delta = 1e-9` and 100 independent instances
by default.  Every ground truth is computed by CVXPY with Clarabel
(`problems.py`); its achieved tolerance, KKT residual, and wall-clock time
are recorded.  The QP plants a KKT point, so the CVXPY solution can be
checked against the known minimizer (`truth_planted_error` in
`parameters.csv`).

## Running

```bash
cd code
uv run python -m numerical_experiments.run_admm_qp --instances 100
uv run python -m numerical_experiments.run_ppm_huber --instances 100
uv run python -m numerical_experiments.plots              # figures for all
uv run python -m numerical_experiments.plots admm_qp      # one study
uv run python -m numerical_experiments.trajectory_2d      # the 2D paths figure
```

`--help` lists the flags: dimensions, the ratio `L/c` (ADMM) or `gamma L`
(PPM) that fixes the penalty or stepsize, the `sigma` and stride grids (the paper's stride `s` is `b` in the code and
the tables), the `b` of the `sigma` sweep and the `sigma` of the `b` sweep, the near-exact
tolerance, the stepsize of the gradient descent baseline in units of `1/L`
(`--gd-step`, PPM), workers, and the results directory.  Instances run on a process
pool with one BLAS thread each and are checkpointed, so a study resumes
after an interruption; `--force` recomputes.  Results land in
`results/numerical_experiments/<study>/` (git-ignored; override with
`--results-dir` or `RAND_NLA_FOM_RESULTS`).

Every inexact run has a hard stop, because the theory guarantees termination
only for `sigma < 1` and the sweep includes `sigma = 1`: `--max-total-inner`
inner steps in total (gradient steps for PPM, GMRES iterations for ADMM;
default 1,000,000) or `--time-limit` seconds of
wall-clock (default 300; not applied to the near-exact baseline, which
always terminates), checked at every inner check, besides `--max-outer`
outer iterations (default 50,000).  Both defaults are about ten times the
heaviest converging run observed, the near-exact baseline of the Huber study.
A run that hits a stop is recorded with `converged = False` and the reason in
the `stopped_by` column of `results.csv` (and of `runs.csv` for the 2D
instance); `summary.csv` counts such runs per configuration
(`n_max_total_inner`, `n_time_limit`, `n_max_outer`), and the figure legends
flag the configuration with `[converged/runs converged]`.

Tests: `uv run pytest numerical_experiments/tests` (also picked up by the
repository's `uv run pytest`).

## How the problem data is constructed

Instance `i` of a study is drawn from `numpy.random.default_rng(seed + i)`
with `seed = 0` by default (`--seed`), so the 100 instances have seeds
`0, ..., 99` and any instance can be regenerated on its own.  The
constructors are `make_qp` and `make_huber` in `problems.py`; the
dimensions below are the defaults and can be changed with `--n`, `--m`,
`--kappa` (QP), and `--threshold` (Huber).

**Linearly constrained QP (`admm_qp`).**  `Q = V diag(lambda) V'` with `V`
the orthogonal factor of the QR decomposition of an `n x n` standard
Gaussian matrix and `lambda` log-spaced in `[1/kappa, 1]`, so that
`Q` is positive definite with `kappa(Q) = 10^4` and `L = ||Q||_2 = 1`
(`Q` is symmetrized once more after the product to remove rounding
asymmetry).  `A` is `m x n` standard Gaussian scaled by `1/sqrt(n)`.  The
data `q` and `b` are not drawn directly: a primal-dual pair `x*` (`n`
Gaussian entries scaled by `1/sqrt(n)`) and `nu*` (`m` Gaussian entries
scaled by `1/sqrt(m)`) is drawn first, and then `q = -Qx* - A'nu*` and
`b = Ax*`, which makes `(x*, nu*)` satisfy the KKT conditions exactly.
Because `Q` is positive definite, `x*` is the unique minimizer, and the
distance between it and the CVXPY solution is recorded as
`truth_planted_error`.

**Huber regression (`ppm_huber`).**  The design `A` is `m x n` standard
Gaussian scaled by `1/sqrt(m)`, and a true coefficient vector `x_true` is
standard Gaussian.  The observations are `b = A x_true + 0.1 w` with `w`
standard Gaussian; then `10%` of the entries of `b`, chosen uniformly
without replacement, are shifted by a gross outlier of magnitude uniform
in `[3, 8]` with a random sign.  The loss is the standard Huber function
with threshold `M = 0.5`, quadratic `r^2/2` for `|r| <= M` and linear
`M(|r| - M/2)` outside, summed over the `m` residuals of `Ax - b`.  Its
gradient is `A' clip(Ax - b, -M, M)`, so the objective is convex and
`L`-smooth with `L = ||A||_2^2` (computed by Lanczos on `A'A` at setup),
but the flat parts of the loss on the outliers mean it is not strongly
convex, and the minimizer is not known in closed form.  The noise level,
outlier fraction, and outlier range are fixed in `make_huber`; only `n`,
`m`, and `M` are exposed as flags.

The stepsize or penalty is set from the smoothness constant in both
studies: `c = L / 10` for ADMM and `gamma = 10 / L` for PPM (`--ratio`).

## The inner solvers

**PPM: gradient descent.**  Gradient descent with stepsize
`2 gamma / (gamma L + 2)` on the proximal subproblem, warm-started at the
previous iterate, with the constants `alpha = (r/(r+2))^2` and
`C_0 = (1+r)^2`, `r = gamma L`, of Proposition 5.1 (`theory.inner_constants`).

**ADMM: GMRES.**  The `x`-subproblem is the symmetric positive definite
system `(Q + cI) x = c w_k - q`, and its graph residual
`e = x + grad f(x)/c - w_k` is the residual of that system divided by `c`.
Each outer iteration starts a GMRES cycle (`admm.Gmres`: classical
Gram-Schmidt with one reorthogonalization, Givens rotations) at the
previously accepted `x`, whose residual is already known, and the pair is
checked every `b` (the stride) GMRES iterations; a check forms the iterate and evaluates
the gradient there, one product with `Q` on top of the `b` of the Arnoldi
steps.  The spectrum of `Q + cI` lies in `[c, L + c]`, so its condition
number is at most `1 + L/c`, and since GMRES minimizes the residual over the
Krylov space, the Chebyshev bound gives `||e^l|| <= 2 alpha^{l/2} ||e^0||` with
`sqrt(alpha) = (sqrt(1 + L/c) - 1)/(sqrt(1 + L/c) + 1)`: the inner-contraction
assumption holds deterministically with this `alpha` and `C_0 = 4`
(`theory.gmres_constants`; `alpha = 0.288` at `L/c = 10`).  These replace
the gradient descent constants of item 3 of Proposition 5.2 in `sigma*`,
`b*`, and every bound below.  A cycle restarts at Krylov dimension 200
(`admm.RESTART`), where the bound would no longer apply; the largest
`max_inner_per_outer` of the study is far below it.

## What runs per instance

1. Draw the instance and solve it with CVXPY (timed: `truth_seconds`).
2. Setup: the smoothness constant `L` by Lanczos; for ADMM also the
   Cholesky factor of `AA'` for the projection onto `{Az = b}` and, for the
   exact method only, the Cholesky factor of `Q + cI` (all timed).
3. The paper's parameters (timed together as `parameter_seconds`):
   `sigma*` from equation (optimal-sigma) of Proposition 5.1 by bisection,
   using the deterministic `c = b log(1/alpha)` of Theorem 4.3 since both
   inner solvers contract deterministically (the Theorem 4.2 variant,
   `c = b log(1/alpha) / (1 - alpha^b)`, is recorded as
   `sigma_star_expected`); the constant `D = (1+sigma)^2 C_0 G_sigma^2 / sigma^2`
   of Corollary 4.1.1 with the piecewise warm-start constant `G_sigma` of
   Lemma 4.1 (`theory.warm_start_constant`) enters every bound; the costs `c_in` of one inner
   iteration and `c_test` of one check, measured once per study by timing
   them on a representative instance in the parent process before the pool
   starts (`calibration_seconds`), so that the concurrent workers do not
   distort the ratio (for ADMM, one Arnoldi step and one check, which forms
   the GMRES iterate, evaluates the gradient there, and projects, both at
   Krylov dimension 10); `b*` of Lemma 4.2 in its `K_up -> infinity` form,
   `sqrt(c_test (1 + log D) / (c_in log(1/alpha)))`, rounded to the cheaper
   of its two neighboring integers (`b_best`); and the unrelaxed
   recommendation of Remark 4.4 (`b_star_exact`): the integer minimizer,
   found by scanning, of the same cost model with the per-outer inner work
   `b / (1 - alpha^b)` of Theorem 4.2 in place of its relaxation
   `b + 1/log(1/alpha)`.  The cost model per outer iteration is
   `c_in W + c_test (1 + W/b)` with `W` the inner work per outer iteration,
   `b + 1/log(1/alpha)` (or `b/(1 - alpha^b)`) plus `log D / log(1/alpha)`
   (`theory.b_cost_per_outer`).  The oracle `b*` with `K_up = K_delta^+` is
   recorded too; it agrees to many digits because `K_delta^+` is
   astronomically large.
4. The runs, in this order:
   * `exact` (ADMM only): the `x`-subproblem solved by the Cholesky factor;
   * `gd` (PPM only): plain gradient descent on `f` with stepsize `1/L`,
     stopped by the same test `||grad f(x)|| <= delta`; one of its
     iterations is one gradient evaluation, so it is compared with the
     proximal method on inner work and wall-clock time;
   * `near_exact`: the inner solver run until the graph residual is at
     most `1e-12` in absolute value, a stand-in for an exact prox;
   * the `sigma` sweep at `b = 1`: `sigma` in `{0.01, 0.05, 0.2, 0.5, 0.9, 1}`
     plus `sigma*`;
   * the `b` sweep at `sigma = 0.1`: `b` in `{1, 2, 5, 10, 20}`; `b_best`
     and `b_star_exact` are flagged (`b_is_optimal`, `b_exact_is_optimal`),
     and each is added as its own run when it is not on the grid.
5. For each run, the per-iteration metrics and the theorem checks below.

Within a run only the algorithm is timed (`runtime_seconds`); the
post-processing that evaluates the metrics is done afterward from the
stored iterates.

## Theorem checks (columns of `results.csv`, aggregated in `summary.csv`)

| Column | Statement tested | Expected |
| --- | --- | --- |
| `thm31_ratio` | Theorem 3.1: `sum_k ||gamma z_k||^2 <= (1+sigma)/(1-sigma) dist(w_0, zer B)^2` | `<= 1` |
| `lemma32_min_ratio` | Lemma 3.2: `||w_k - s||^2 - ||w_{k+1} - s||^2 >= (1-sigma)/(1+sigma) ||gamma z_k||^2` | `>= 1` |
| `per_step_violations` | Theorem 4.3, per step: `N_k <= b ceil(log+ beta_k / (b log(1/alpha)))` with `||R(w_k)||` replaced by its lower bound `(1-sigma)||y_k - w_k||` of Lemma 3.1 | `0` |
| `per_step_violations_exact` | the same with the exact `||R(w_k)||` (ADMM only, where the exact step is a Cholesky solve) | `0` |
| `thm43_ratio` | Theorem 4.3, total: measured `sum_k N_k` over the bound with the observed `K_delta` | `<= 1` |
| `thm42_ratio` | Theorem 4.2 with `E[K_delta]` replaced by the observed `K_delta`: `(K_delta + 1) b/(1 - alpha^b) + (K_delta log D + log+ beta_0)/log(1/alpha)` plus the initial term | `<= 1` |
| `kappa_empirical`, `rate_bound`, `rate_observed_max` | Theorem 3.2: with `kappa = max_k dist(w_k)/||R(w_k)||`, every step contracts `dist^2` by at least `1 - (1-sigma)/(kappa^2(1+sigma))` (ADMM only) | `rate_observed_max <= rate_bound` |

The tolerance in `beta_k` and in the initial term is `gamma delta`, as in
the paper (the terminal test `||z|| <= delta` is `||gamma z|| <= gamma delta`
on the scale of `R`).  `dist(w_k, zer B)` uses the CVXPY solution (`w* = x*` for PPM,
`w* = x* + grad f(x*)/c` for ADMM).  The distance-based checks are
evaluated only while `||gamma z_k|| >= 1e-3`, so the accuracy of the CVXPY
solution (about `1e-6` in `x` for the Huber problem, `1e-14` for the QP)
cannot bias them.  `||R(w_0)||`, needed for `beta_0`, is computed exactly
(ADMM) or by gradient descent to near machine precision (PPM).  The checks
apply to the runs of Algorithm 1; the gradient descent baseline has no
theorem to test and leaves these columns empty.

## Tables

| File | Contents |
| --- | --- |
| `parameters.csv` | one row per instance: `L`, `alpha`, `C_0`, `sigma*` (for every `b` of the grid), `b*`, `b_best`, calibration costs, ground-truth accuracy and time, setup times |
| `results.csv` | one row per run: counts, times, final accuracy, the theorem checks |
| `histories.csv` | one row per run and outer iteration (for gradient descent: each of the first 200 iterations, then iterations at a 2% geometric spacing, and the last one, with `inner` and `checks` counting the steps since the previous row): suboptimality, feasibility or stationarity, `||z_k||`, inner iterations, checks, elapsed time, distance to the solution, the per-step bounds |
| `curves.csv` | percentiles over instances of the history metrics against outer iterations, cumulative inner iterations, and time (final values carried forward after stopping) |
| `summary.csv` | one row per configuration: moments of the scalar measurements and the outcome of every theorem check |

## Figures

Drawn by `plots.py` from the tables alone, as PNG and PDF, following the
`dataviz` conventions with the colorblind-safe Okabe-Ito palette (validated
with the skill's all-pairs check): the five swept values are blue, sky
blue, green, orange, and purple in a fixed order, the paper's recommended
value is vermillion, the classical baseline (exact ADMM, or gradient
descent for PPM) is black, and the near-exact baseline is a muted gray.
Every series also has its own marker shape, so
identity never rests on hue alone.  Curves are medians over instances with
the interquartile band and sparse 10--90 percentile whiskers.

| Figure | Shows |
| --- | --- |
| `convergence_sigma`, `convergence_b` | suboptimality and feasibility (ADMM) or stationarity (PPM) against outer iterations and against cumulative inner work (gradient evaluations for PPM, which count the gradient at the start plus one per gradient step and so run one ahead of `cumulative_inner`; GMRES iterations for ADMM); the gradient descent baseline appears on the inner-work panels only, since it has no outer iterations, just as exact ADMM appears on the outer-iteration panels only |
| `convergence_sigma_zoom`, `convergence_b_zoom` | the outer-iteration panels on two windows, the early transient and, per metric, the final approach to its plateau, as median lines with markers only, so that curves that coincide at full scale can be told apart |
| `work_sigma`, `work_b` | accepted updates, total inner iterations against the Theorem 4.3 bound, checks, and algorithm wall-clock, against the swept parameter, with the baselines as horizontal lines (gradient descent on every panel but the accepted-updates one) |
| `per_step` | inner iterations per outer iteration against the per-step bound, one panel per `sigma` |
| `walltime` | setup, parameter computation, and algorithm time per configuration, baselines included, with whiskers for the 10--90 percentile spread of the algorithm time over instances and with the CVXPY ground truth for scale |
| `walltime_sgm` | the same chart with the shifted geometric mean over instances of each time in place of the median (shift 1 s; `--walltime-shift`), with the same whiskers |
| `bstar_model` | the cost model behind `b*`: the per-outer cost `c_in W + c_test (1 + W/b)` with the relaxed inner work of Lemma 4.2 and the unrelaxed one of Remark 4.4 against `b`, with their minimizers, next to the measured time per outer iteration |

Wall-clock times are measured with `--workers` (12 by default) single-threaded workers running
concurrently, so they are comparable across configurations but were taken
under load.

Titles, axis labels, and tick labels are solid black; legend entries and
in-plot annotations use a secondary gray.

## Paths of the iterates in two dimensions

`trajectory_2d.py` draws one Huber instance with the same generator as
the main study but `n = 2` features and few observations, so that the
loss is anisotropic and kinked and the proximal steps are visible at full
scale.  The default, `m = 7` observations with seed 6 (`--m`, `--seed`),
is the instance with the most visible steps (steps longer than 2% of the
distance from the start to the optimum) among those whose near-exact run
takes fewer than 50 outer iterations, in the ranking over seeds 0 to 39
and several observation counts that `--scan 40` prints; with hundreds of
outer iterations the individual steps are no longer distinguishable.
Two close alternatives, `--m 6 --seed 31` (a hook at the optimum and a
wide gradient descent arc) and `--m 8 --seed 9` (a strongly curved
valley, 81 outer iterations), are kept side by side with the default under
the names `ppm_huber_2d_m6_s31` and `ppm_huber_2d_m8_s9` (`--name` sets
the results subdirectory).  With `gamma L = 10`
and `b = 1` as in the main study it runs, from the origin, the near-exact
proximal point method, the inexact one at every `sigma` of the sweep and
at `sigma*`, and plain gradient descent, each to `delta = 1e-9`, and
solves the instance with CVXPY.  The tables land in
`results/numerical_experiments/ppm_huber_2d/`: `instance.csv` (constants,
optimum, start), `runs.csv` (outer iterations, gradient steps and evaluations, time per
run), `trajectories.csv` (one row per run and accepted iterate, with, for
the proximal runs, the center `w_k` of the subproblem the iterate came
from, its exact solution `J(w_k)` computed to machine precision, and the
distance between the two), and `contours.csv` (`f - f*` on a grid over
the full view and over the zoom window).

The figure `trajectory_2d` shows the accepted iterates of every run over
the contours of `f - f*` (powers of ten and their triples), with the
optimum starred and the start circled, in the same colors and markers as
the `sigma` sweep of the main study; the right panel zooms on the optimum
(`--zoom` sets the window, `--plot-only` redraws from the tables).  The
legend states each run's outer iterations and gradient evaluations.  The figure
`trajectory_2d_exact` keeps only the near-exact path and the `sigma = 0.9`
run and adds, at each iteration of the latter, the exact solution of the
subproblem it approximated as a hollow marker joined to the accepted
iterate by a dotted segment.  Both are numbered by the iteration: with a
loose `sigma` the warm start can pass the acceptance test of several
consecutive subproblems without an inner step, so one accepted point can
carry several numbers and be joined to several exact solutions, one per
subproblem.

## The torsion operator

`torsion_operator.py` is the same experiment on an operator that is not a
gradient: `B(x) = J (x - x*)` with `J` the rotation by a quarter turn, so
`B` is monotone (`J + J^T = 0`); `x*` sits at `--distance` (default 5) from
the origin in a direction drawn from `--seed`, and the stepsize is `--gamma`
(default 2).  On a rotation the exact proximal point step satisfies
`||w - x*||^2 - ||J(w) - x*||^2 = ||w - J(w)||^2` with equality, so the
one-step decrease lemma of the paper (Lemma 3.2) is tight there, whereas
on the Huber instance the decrease is tens of times the guaranteed amount;
the factor `(1 - sigma)/(1 + sigma)` that inexactness puts on the
guaranteed decrease therefore acts on all of it.  The inner solver is
Richardson iteration `x <- x - tau e` on the resolvent residual
`e = gamma B(x) + x - w`, which is what gradient descent on the proximal
subproblem reduces to for a gradient; `tau = 1/(1 + gamma^2)` minimizes the
contraction factor `rho = ||I - tau (I + gamma J)|| = gamma / sqrt(1 + gamma^2)`,
so the inner-contraction assumption holds with `alpha = rho^2` and
`C_0 = 1`, from which `sigma*` follows as in the main studies.  There is no
classical baseline: the runs are the near-exact method, the `sigma` sweep,
and `sigma*`, all deterministic.  Every proximal run obeys the hard stop of
the main studies.  The tables mirror those of the Huber instance, with
`distance` (`||x - x*||`) in place of `f - f*`, and the extra `field.csv`
holds the direction of `-B` on a coarse grid (`--field` arrows per axis).

The figures `trajectory_2d` and `trajectory_2d_exact` (`--exact-sigma`
picks the run the latter singles out) are as for the Huber instance, over
contours of `||x - x*||` with the direction of `-B` drawn as arrows.  A
third figure, `fejer_ratio`, drawn for both instances, plots against
`sigma` the one-step decrease of `||w_k - x*||^2` divided by
`||gamma z_k||^2` over the accepted updates (median, 10--90% whiskers),
together with the guarantee `(1 - sigma)/(1 + sigma)` of the one-step
decrease lemma, the level 1 guaranteed for exact steps, and the level the
near-exact run attains.  A fourth, `convergence_sigma`, also drawn for both
instances, is the sweep's convergence in the layout of the main studies'
figure of the same name: the merit metric (`f - f*` or `||x - x*||`) and,
for the Huber instance, the residual `||z_k||`, against outer iterations
and against cumulative inner work, one curve per run since there is a
single instance.  The torsion figure has the distance row only: `J` is an
isometry, so `||B(x_k)|| = ||x_k - x*||`.

## Known issue: Cholesky through SciPy on Apple Accelerate

On macOS 27.0 and 27.1, Accelerate's `dpotrf` and `spotrf` with `uplo='U'`
write into the strictly lower triangle (from order 33 for `dpotrf`, 65 for
`spotrf`) and write out of bounds from order 1449 (`dpotrf`) or 1481
(`spotrf`); `uplo='L'` is unaffected.  SciPy's `cholesky` and `cho_factor`
reach the `'U'` path for `lower=True` on C-ordered input, so at the QP size
of `n = 2000` they corrupt the heap and the process crashes later in
unrelated code.  `admm.py` and `problems.py` therefore factor with
`numpy.linalg.cholesky` and solve with `scipy.linalg.cho_solve`, which are
unaffected.

Apple tracks this as FB24671487 and fixed it in macOS 27.2 (verified in beta
26B5101f); see scipy/scipy#26145 and #26388.  On 27.2 or later the workaround
can be dropped.
