# The cluster problems as the original studies

`studies/` runs the cluster's problems and inner solvers in the format of the
original studies of `numerical_experiments` (`../README.md`): the same
configurations per instance, the same stopping rule, the same tables with the
same columns, the same theorem checks, and the same figures.  The problems,
matrix sources, and inner solvers come unchanged from the sealed numerical
release `releases/regression_v9` (`studies.ENGINE`); no sealed release, root
`experiments/` module, manifest, or earlier result is modified.

| Study | Problem | Method | Original study |
| --- | --- | --- | --- |
| `admm_qp` | the periodic-grid equality QP: `n = side^3` variables, `n/4` coordinate equalities, `kappa(Q) = 10^4` | inexact ADMM with any cluster inner solver, against exact ADMM by FFT | `admm_qp` |
| `admm_lasso` | the split LASSO `min_{u,v>=0} 1/2 ||A(u-v)-b||^2 + lambda 1'(u+v)` on any matrix source | inexact ADMM, against exact ADMM (Cholesky or CG) | `admm_qp` |
| `ppm_logistic` | ridge logistic regression on any matrix source | inexact proximal point with gradient descent inside, against plain gradient descent | `ppm_huber` |

## Running

On a workstation (`--allow-local`; never on a login node), exactly like the
original `run_study`: calibrate in the parent, run the instances on a pool,
collect, and aggregate.

```bash
uv run python -m studies.run_admm_qp --allow-local --instances 2 --side 6
uv run python -m studies.run_admm_lasso --allow-local --instances 2 --rows 600 --cols 200
uv run python -m studies.run_ppm_logistic --allow-local --instances 2 --rows 600 --cols 200
uv run python -m plots.studies admm_qp_side6_gmres      # the original figures
```

`--help` lists the flags.  The original ones keep their names, meanings, and
defaults: `--instances`, `--seed`, `--ratio` (`L/c` for ADMM, `gamma L` for PPM),
`--delta`, `--sigmas`, `--bs` (the paper's stride `s`), `--sigma-sweep-b`,
`--b-sweep-sigma`, `--near-exact-tolerance`, `--max-outer`, `--max-total-inner`,
`--time-limit`, `--gd-step`, `--gd-max-iterations`, `--workers`,
`--results-dir`, `--force`, `--no-aggregate`, and `--max-restarts`.  The
problem flags are the cluster's: `--side` and `--kappa` for the QP;
`--dataset-spec`, `--dataset`, `--orientation`, `--data-dir`, `--rows` and
`--cols` (to shrink a `random_sparse` source), and `--lambda-ratio` for the
regressions.  The inner solver is `--inner` (with `--rank`, `--block-size`,
`--gd-factor`, `--restart`), and `--inner-initialization` resets the inner
start every outer iteration (`fixed` or `gaussian`, as in the cluster's
ablation; `warm`, the default, is the original).  Flags can be read from a
file: `@studies_v1/inputs/admm_qp_side40_gmres.args`.

Results land in `<results-dir>/<name>/` (default `results/studies/`, or
`$RAND_NLA_FOM_RESULTS/studies/`), with `<name>` from `--name` or from the
settings, e.g. `admm_qp_side40_gmres` or `admm_lasso_random_normal_cg`.
`settings.json` holds the settings of the study; a stage with other settings
refuses to write into it.

### On Della: one task per configuration

A campaign (`studies.campaign`) runs studies on Della with **every
configuration of every study its own Slurm array task**, running that
configuration for each instance in turn and storing each run as it finishes, so
a configuration that stalls holds one slot until its time limit and delays no
other result.  (One task per run is finer still, but a campaign of that kind
exceeds Della's per-user limit of 1,000 queued jobs many times over.)  A
campaign specification lists the studies and assigns each to a lane; `prepare`
writes their argument files and the manifests, and seals them:

```bash
uv run python -m studies.campaign prepare studies_v1/campaign.json
uv run python -m studies.campaign commands studies_v1   # the submissions
uv run python -m studies.campaign status studies_v1     # after a sync down
```

Each lane is its own chain of arrays of `slurm/studies_campaign.slurm` (export
`MANIFEST`): a **calibrate** array (one task per study: it times `c_in` and
`c_test`, fixes the configuration list in `configurations.csv`, and sets up
every instance in `parts/<seed>/setup/`), the **run** array (one task per study
and configuration, `afterany` on the calibrate array), and a **collect** array
(one task per study, `afterany` on the runs, collecting with
`--allow-partial`).  Lanes keep the caps of `AGENTS.md` apart (eight QP tasks,
one Kaczmarz task, two regression tasks), so the slow Kaczmarz and regression
runs never wait for, or hold up, the QP runs.  The run manifests are
configuration-major, so the baselines of every study come first.  Each study
reserves room for its recommended runs before calibration fixes them; a
reserved index beyond its list ends at once.  A run manifest longer than
Della's array limit (2,501 tasks) is split into parts that run side by side
with the lane's cap divided among them.  Every task checks the campaign seal
first: code or inputs changed after the launch stop it, and a changed campaign
gets a new name.

`collect` can run at any time: it merges the finished runs, lists the rest in
`missing.csv`, and with `--allow-partial` writes the tables from what is
there.  A missing run is rerun by resubmitting its manifest index; the
instances already finished are skipped.  Every run
records its host, CPU, Slurm ids, package versions, and source hashes in
`parts/<seed>/runs/<n>.provenance.json`; inside an allocation each task also
verifies the sealed engine release.  `studies_v1/README.md` has the exact
submission commands of the first campaign.

The single-study stages remain for small studies: `--stage calibrate`, then
`--stage run` per run or `--stage task` per instance (every configuration in
one process, resuming after the last finished one), then `--stage collect`.

On a workstation (Apple M5 Max, one BLAS thread), one `admm_qp` instance with
GMRES and the default grids took 11 s at side 40 (64,000 variables) and 153 s,
with a peak resident set of 1.6 GB, at side 100 (1,000,000 variables); one
`admm_lasso` or `ppm_logistic` instance took 7 s on a 6,000 by 2,000 design.
These are not cluster measurements.

## What runs per instance

As in the original (`../README.md`, "What runs per instance"), with the
cluster's problems:

1. Draw the instance (`generation_seconds`): the engine's `build`, seeded by the
   instance seed (for a native-label dataset the seed changes nothing, so use
   one instance).
2. Setup: the smoothness constant `L` by Lanczos (`smoothness_seconds`), as in
   the original; for ADMM also the exact `x`-update (`factor_seconds`) and the
   inner solver's setup, such as the Nystrom sketch (`inner_setup_seconds`),
   shared by the instance's configurations as the original shares its setup.
3. The ground truth (`truth_*`), then the paper's parameters
   (`parameter_seconds`): `sigma*` of Proposition 5.1, `s*` of Lemma 4.2
   (`b_star`, `b_best`), the unrelaxed `s*` of Remark 4.4 (`b_star_exact`), the
   oracle `b_star_oracle`, and `||R(w_0)||` (`r0_norm`), exactly as the original.
4. The runs, in the original order: `exact` (ADMM) or `gd` (PPM); `near_exact`
   (inner residual `<= 1e-12`); the `sigma` sweep at `s = 1` with `sigma*`;
   the `s` sweep at `sigma = 0.1` with `b_best` and `b_star_exact` flagged or
   added.
5. For each run, the per-iteration metrics and the theorem checks below.

Every inexact run stops at `||z_k|| <= delta = 1e-9` (`||x - z_{k+1}||` for
ADMM, `||grad f(x)||` for PPM) or at the original hard stops (`10^6` inner
steps or 300 s, not the time limit for the near-exact baseline; 50,000 outer
iterations); `stopped_by` records why.  The time limit counts the algorithm
alone, not the metrics evaluated with the clock paused, so size the Slurm
`--time` of a task from a pilot, not from `--time-limit`.  A configuration
that outlasts its task is rerun, and outlasts it again, at every resubmission;
`--near-exact-time-limit` caps the near-exact baseline, which the original
leaves uncapped and which for Kaczmarz checks after every block update.

## Inner solvers and their constants

| `--inner` | Studies | Implementation | `(alpha, C_0)` |
| --- | --- | --- | --- |
| `gmres` (default, ADMM) | `admm_qp`, `admm_lasso` | the original's GMRES (restart 200), basis grown on demand | `theory.gmres_constants`, as the original |
| `gd` | all three | ADMM: the engine's minimax-step gradient descent; PPM: the original's step `2 gamma/(gamma L + 2)` | `(rho^2, 1)` for ADMM; for PPM `theory.inner_constants`, as the original, or the contraction of the step scaled by `--gd-factor` |
| `cg` | `admm_qp`, `admm_lasso` | the engine's CG | `(rho^2, 4 kappa)` |
| `block_kaczmarz` | `admm_qp` | the engine's randomized block projections | expected-value bound: `sigma*` uses Theorem 4.2's `chi`, and Theorem 4.3's deterministic checks are `nan` |
| `agd`, `nystrom_pcg`, `cyclic_kaczmarz` | QP (and LASSO for the first two) | the engine's solvers | none verified |

The engine's solvers are started from the residual `-c e`, which the original
GMRES also uses instead of one more product.  For a solver without verified
constants (`contraction = none` in `parameters.csv`), `sigma*` and `s*` are not
defined: their runs are left out, and the columns that need `alpha` and `C_0`
(`thm43_*`, `thm42_*`, `per_step_*`, the bound columns of `histories.csv`) are
`nan`; Theorem 3.1, Lemma 3.2, and Theorem 3.2 do not need them and are still
checked.  For block Kaczmarz, whose contraction holds only in expectation,
`sigma*` and `s*` are defined, Theorem 4.2's bound (`thm42_*`) is checked, and
the deterministic bounds of Theorem 4.3 (`thm43_*`, `per_step_*`, the bound
columns of `histories.csv`) are `nan`, since the theory does not claim them
there.  Its `alpha` is within `1e-5` of one, so Remark 4.4's integer search for
`s*` extends past the original's 1000 to four times Lemma 4.2's `s*`, where
the minimizer lies (about 1.4 times it).  The default stride grid suits Krylov
and gradient solvers; for Kaczmarz, where a check costs a full product and an
update a block of rows, pass a block-scaled grid with `--bs` and
`--sigma-sweep-b` (for example the cluster's fractions of one pass over the
blocks).

## Ground truth

| Study | Reference solver (`truth_solver`) | `truth_stationarity` |
| --- | --- | --- |
| `admm_qp` | CG on the reduced KKT system, checked against the planted KKT point (`truth_planted_error`) | reduced gradient norm |
| `admm_lasso` | FISTA with adaptive restart and a polish on the support (`fista_polish`), or CVXPY with Clarabel (`--truth-solver clarabel`, small designs) | natural residual `||b - soft(b - grad, lambda)||` |
| `ppm_logistic` | Newton-CG with an Armijo line search (`newton_cg`), or CVXPY with Clarabel | gradient norm |

The original computes every ground truth with CVXPY and Clarabel, which does
not scale to these sizes.  On small designs the iterative references match
Clarabel's objective value to a relative `1e-8` (the tests' tolerance; about 12
digits were observed) and reach a stationarity of about `1e-14`, below
Clarabel's.  `--truth-tolerance`, `--truth-time-limit`, and
`--truth-max-iterations` bound them; `truth_status` is `optimal_inaccurate`
when the tolerance was not reached, and `truth_tolerance` then holds the
attained relative stationarity.

## Tables and figures

`parameters.csv`, `results.csv`, `histories.csv`, `curves.csv`, and
`summary.csv` have exactly the original columns, in the original order, and the
same meanings (`../README.md`, "Tables" and "Theorem checks").  `parameters.csv`
appends what identifies a cluster study: `family`, `algorithm`, `inner`,
`rank`, `block_size`, `restart`, `gd_factor`, `inner_initialization`,
`contraction`, `mu`, `smoothness_certified` (the engine's bound),
`inner_setup_seconds`, `exact_solver`, `truth_solver`, `source`, and
`orientation`.  `curves.csv` and `summary.csv` come from a snapshot of the
original `aggregate.py`.  `plots.studies` is a snapshot of the original
`plots.py` that draws the same ten figures, labeled for each study, with the
reference solve in place of CVXPY in the wall-clock figures.  Without
contraction constants it marks no recommended `sigma*` or `s*` and skips the
two figures of the bounds (`per_step`, `bstar_model`); for block Kaczmarz it
skips `per_step`.  `collect` treats an instance whose run files do not match
its record as unfinished, so a lost file cannot shorten the tables silently.

The feasibility column of an ADMM study is `||x_{k,I} - t||` for the QP's
coordinate equalities `x_I = t` (the original QP records `||A x_k - b||`) and
`||min(x_k, 0)||` for the split LASSO; it is `nan` for logistic regression, as
for the original Huber study.

## Differences from the original studies

* The iterates do not fit in memory at these sizes, so the metrics that the
  original evaluates from stored iterates after a run (`suboptimality`,
  `suboptimality_z`, `feasibility`, `dist`, `residual_exact`) are evaluated
  after every outer iteration with the clock paused; `elapsed` and
  `runtime_seconds` still time the algorithm alone.  `--no-exact-residuals`
  skips `||R(w_k)||` (one exact `x`-update per outer iteration, by CG for a
  large LASSO design), and with it the columns that need it.
* The exact `x`-update is an FFT (QP) or a Cholesky factor of the smaller Gram
  matrix, or CG to `rtol = 1e-13` beyond `--exact-direct-max` (LASSO), in place
  of the Cholesky factor of `Q + cI`; `exact_solver` records which.
* `L` is computed by Lanczos, as in the original, not taken from the engine's
  certified bound (`smoothness_certified`), so `c = L/ratio` and
  `gamma = ratio/L` match the original's
  definitions; Lanczos starts from a fixed vector, so `L` is reproducible.
  On the random designs tried the certified bound was 5 to 11 times the
  Lanczos value, so these runs use a smaller `c` and a larger `gamma` than the
  earlier cluster runs at the same `ratio`.
* The ground truth is an iterative reference solve (above), not CVXPY.
* The PPM inner step is the original's `2 gamma/(gamma L + 2)`, not the engine's
  logistic step, which also counts the ridge `mu`: with the original's step the
  constants of Proposition 5.1 hold exactly for every `mu >= 0`.
* `gradient_evaluations` counts the products with the Hessian (ADMM) or the
  gradient evaluations (PPM) actually performed, as the original does except
  at a GMRES breakdown, where the original counts the steps it skipped.
* Instances are checkpointed per configuration, not per instance.
