# Accounting release v4

A separate source tree, so the running `speed_v2` (root) and `accounting_v3`
jobs keep loading unchanged code. Batch arrays start a new interpreter for
every task, so **any edit to a tree in use reaches tasks that have not started
yet**. New jobs select this tree with `EXPERIMENT_RELEASE=accounting_v4`;
`slurm/common.sh` needs no change. Manifests default to
`implementation=accounting_v4`, so run IDs never collide with older results.

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

`hessian_products_total = inner_hessian_products + check_products`. Earlier
releases omitted the gradient computed by every stopping check; for GD at
stride 1 the inner counter alone missed nearly all products. For logistic
regression, `check_products` counts gradient evaluations.

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
- Interrupted or missing runs stop selection with a list of run IDs to rerun,
  instead of silently disqualifying a configuration.
- `--from-results` and `--from-manifest` accept several directories, such as
  the QP fast lane and the expanded Kaczmarz grid.
- The `evaluate` phase reruns every selected configuration, baselines
  included, on held-out seeds (default 3, 4, 5); seeds used for selection
  are rejected.

The speed_v2 Kaczmarz screen converged in no arm, but block sizes 1 and 8 ended
on `max_inner` (100,000 steps, about 1.6 passes at n = 64,000), not on time.
Rerun the screen under v4 work budgets before concluding anything:

```bash
cd releases/accounting_v4
uv run python -m experiments.plan \
  --from-manifest manifests/qp_full_tune_20261030.csv \
  --output-dir manifests/accounting_v4/qp_plan
# After the fast lane completes (QP steps from its winners):
uv run python -m experiments.manifest steps \
  --from-results results/qp_fast_speed_v2 \
  --from-manifest manifests/speed_v2/fast.csv \
  --output manifests/accounting_v4/qp_steps.csv
# After the steps phase completes:
uv run python -m experiments.manifest evaluate \
  --from-results results/qp_steps_v4 \
  --from-manifest manifests/accounting_v4/qp_steps.csv \
  --output manifests/accounting_v4/qp_evaluate.csv
```

Submit with `--export EXPERIMENT_RELEASE=accounting_v4` and a `--constraint`
for one CPU family, so timings are comparable across methods.

## Known limits

- Python overhead still dominates small-block Kaczmarz steps: the eight-block
  cache rarely hits when there are many blocks.
- The `L` bounds remain certified but loose, which still slows GD inner steps
  and the PDHG baseline; the ratio sweep only partly compensates.
- `RLIMIT_AS` limits virtual memory, not resident memory.
