# Accounting release v7

A separate source tree preserving all earlier releases and queued-job code.

V7 fixes the work-cap boundary found by the 64,000-variable validation pilot:
block-size-one Kaczmarz previously took 64,001 updates for a one-pass budget.
Budgets now use integer update limits; reported fractional work is derived from
the update count rather than accumulated floating-point additions. Setup,
calibration, certificates, and work-counter definitions are unchanged.
This release includes the v5 accounting and planner fixes and corrects the
exception-classification and reporting regressions introduced in its review.
Batch arrays start a new interpreter for
every task, so **any edit to a tree in use reaches tasks that have not started
yet**. New jobs select this tree with `EXPERIMENT_RELEASE=accounting_v7`;
`slurm/common.sh` needs no change. Manifests default to
`implementation=accounting_v7`, so run IDs never collide with older results.

## Timing

Every attempt starts from scratch. The final attempt is therefore a clean
measurement and is the only time used for ranking.

| Field | Meaning |
| --- | --- |
| `total_seconds`, `clean_attempt_seconds` | Parent-measured wall time of the final attempt: startup, loading, setup, calibration, solve, and reporting. |
| `all_attempts_seconds` | Sum over recorded attempts (campaign cost); NaN if an earlier attempt was hard-killed. Never used for ranking. |
| `prior_unclean_attempts`, `campaign_accounting_complete` | Earlier hard kills. They no longer disqualify the clean final attempt. |
| `superseded_attempts` | `--force` moves earlier attempts here, so a forced rerun is measured alone. |

No checkpoints are written during the timed solve: attempts never resume, so
checkpoint I/O would only charge ADMM for overhead the baselines never pay.
Signal handlers are installed before loading, and an interruption during the
build or calibration produces an `interrupted` record (`interrupted_stage`).

## Work counts

Counts are measured at the problem operators, not reconstructed from iteration
counts. `work_count_scope=problem_operators_after_build_including_calibration`
includes inner setup, all calibration calls (warm-up included), the solve,
stopping checks, diagnostics, and the final true convergence certificate.
Matrix generation/loading and artifact preparation are outside this **work-count**
scope; their measured times remain included in the existing timing fields.

- `hessian_products_total`: quadratic-operator vector equivalents for QP and
  LASSO, including every column of a sketch. A fused LASSO loss/gradient or a
  full metric gradient counts as one equivalent. Logistic regression reports
  NaN here because it does not have a constant Hessian.
- `design_forward_products`, `design_transpose_products`: vector equivalents
  of multiplication by the regression design and its transpose. These include
  objective-only passes and metric evaluation; fused streaming operations count
  their algebraic forward/transpose work, not physical disk passes.
- `gradient_evaluations`: full objective gradients, distinct from Hessian
  products. Do not add it to the other counters; the categories overlap.
- `setup_*` and `calibration_*`: stage counts. Calibration counts are saved
  before resetting the solver cache/RNG and are included exactly once in totals.
  `row_updates` and `block_factorizations` also include calibration; their
  `solve_and_setup_*` fields retain the corresponding noncalibration counts.
- `inner_hessian_products` and `check_products` remain narrower diagnostic
  counters. They are not a formula for the full total. `certificate_products`
  identifies explicit true-gradient certification calls.
- `work_units` is the inner-update budget only, excluding checks/calibration.
  Kaczmarz row work is reported separately from full operator products.
- OSQP/SCS do not expose comparable internal work counts: their totals are NaN
  and `work_count_complete=False`. Wall-clock setup and solve costs remain
  measured. `work_count_complete=True` means complete within the stated operator
  scope, not a count of all floating-point operations or preprocessing work.

`plots.corrections` never modifies original result CSVs. Complete measured
operator totals are preserved, including PDHG and LASSO: iteration counts never
replace a measurement with another unit. PDHG has no calibration stage, so its
measured total is also its solve-phase total. The nine-iteration QP regression
case correctly reports 12 measured products.

Historical PDHG totals remain NaN because the recorded iteration count does not
identify every diagnostic evaluation. Legacy recomputation-mode inner/check
counts can be exported as explicitly partial estimates excluding final
diagnostics and calibration. A v4 recorded total is also labeled partial;
recurrence checks must not be counted as true gradient evaluations. There is
no claim that omitted work is bounded by one product. Compare measured totals
within the same scope and unit, consulting `work_count_basis` and
`solve_phase_basis`; historical estimates are not exact measurements.

LASSO PDHG multiplies by A and Aᵀ separately, so its measured Hessian count
holds only its metric evaluations; its basis is flagged
`..._pdhg_lasso_metrics_only_compare_design_products`. Compare LASSO and
logistic work across methods with `design_forward_products` and
`design_transpose_products` (an ADMM Hessian product is one of each); the
report summary includes their medians. On a small LASSO test, PDHG's Hessian
count was 19 against CG's 264, while its design products were 190 + 190
against 351 + 264.

## Changes that alter the algorithm or the protocol

Each is a recorded configuration field and therefore part of the run hash.

- `check_source=recurrence` (default for new manifests): CG and Nyström-PCG
  read the stopping test from their maintained residual, using
  `Q x + q = c (w - x) - r`. The common convergence certificate still
  recomputes the true gradient. GD already reused its check gradient; this
  removes the extra product per check that only the Krylov methods paid.
  GMRES and Kaczmarz do not maintain the residual vector and still recompute.
- `budget_unit=work` (default): budgets count Hessian-product equivalents.
  A block Kaczmarz update costs `block_size / n`, so a one-row block is no
  longer stopped after a small fraction of one pass over the rows.
- The legacy planner filters unsupported model prescriptions (including
  Nyström `optimal` arms). Both screening and expansion regenerate Kaczmarz
  stride grids from each instance and block size. Screening tries every such
  stride; expansion retains all seeds and initialization modes for successful
  method/block pairs. Infrastructure events require a rerun before expansion.
- Kaczmarz stride grids scale with the number of blocks (1 and 1%, 5%, 10%,
  50%, 100% of one pass); the stride search for model recommendations extends
  to the number of blocks, capped at 100,000.
- Nyström-PCG is no longer a model solver. The v3 bound multiplied `kappa(H)`
  by the spread of the sketch eigenvalues, which bounds the preconditioned
  system worse than plain CG. TODO: a usable bound needs the sketch residual
  norm; verify the statement before adding one.
- Calibration times one warm-up call and then takes the median of 7 timings
  for both the inner step and the check, matching the configured check source.

## Baselines

- OSQP 1.1.3 counts its own setup against the first solve's `time_limit`
  (verified: a limit of half the setup time stops at iteration 0). The setup
  time is added back so the solve receives the same budget as other methods.
- OSQP and SCS use a warm-started native tolerance ladder
  (`baseline_tolerance_rungs`, default 3: 0.1, 0.01, 0.001 times the common
  tolerance). Each rung is checked against the common criterion, and every
  rung is charged. SCS rebuilds its solver for a new tolerance; the rebuild is
  charged to the solve. `native_tolerance_ladder` lists the rungs used.

## Selection and evaluation

- Ranking uses clean single-attempt time. accounting_v3 final attempts are
  also clean; legacy rows that resumed from a checkpoint have no clean time.
- Selection groups include budgets, check source, and budget unit, so runs
  with different budgets (for example the 120 s Kaczmarz screen) never pool.
- Worker exceptions use conservative classification (`experiments/outcomes.py`).
  Method outcomes: `MemoryError` and OSQP's memory-allocation error (code 5)
  are `resource_limit`; the typed `SolverFailure`, legacy CG's exact
  lost-curvature diagnostic, and OSQP's explicit nonconvex error (code 4) are
  `solver_error`. Operating-system errors and truncated reads are
  `infrastructure_error`: rerun them. Every other exception (invalid input,
  checksum failures, programming errors, other OSQP codes) is
  `unresolved_error`: investigate it, since a rerun reproduces a bug. Blocked
  selection and Kaczmarz expansion list the two groups separately. Older v5
  `solver_error` rows are reclassified from their recorded exception type and
  message rather than trusted blindly.
- `--from-results` and `--from-manifest` accept several directories, such as
  the QP fast lane and the expanded Kaczmarz grid.
- The `evaluate` phase reruns every selected configuration, baselines
  included, on held-out seeds (default 3, 4, 5); seeds used for selection
  are rejected.

The speed_v2 Kaczmarz screen converged in no arm, but block sizes 1 and 8 ended
on `max_inner` (100,000 steps, about 1.6 passes at n = 64,000), not on time.
Rerun the screen under v5 work budgets before concluding anything:

```bash
cd releases/accounting_v7
uv run python -m experiments.plan \
  --from-manifest manifests/qp_full_tune_20261030.csv \
  --output-dir manifests/accounting_v7/qp_plan
# After the fast lane completes (QP steps from its winners):
uv run python -m experiments.manifest steps \
  --from-results results/qp_fast_v5 \
  --from-manifest manifests/accounting_v7/qp_plan/fast.csv \
  --output manifests/accounting_v7/qp_steps.csv
# After the steps phase completes:
uv run python -m experiments.manifest evaluate \
  --from-results results/qp_steps_v5 \
  --from-manifest manifests/accounting_v7/qp_steps.csv \
  --output manifests/accounting_v7/qp_evaluate.csv
```

Submit with `--export EXPERIMENT_RELEASE=accounting_v7` and a `--constraint`
for one CPU family, so timings are comparable across methods.

## Known limits

- Python overhead still dominates small-block Kaczmarz steps: the eight-block
  cache rarely hits when there are many blocks.
- The `L` bounds remain certified but loose, which still slows GD inner steps
  and the PDHG baseline; the ratio sweep only partly compensates.
- `RLIMIT_AS` limits virtual memory, not resident memory.
