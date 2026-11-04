# Launch status — November 3, 2026

## Completion update — November 4, 2026, 03:21 UTC

All four recovery pilots finished; none reached the common 1e-6 certificate.
Both LASSO runs hit their 7,200-second solve budgets. Both logistic GD runs
hit the 3,600-second per-inner-solve limit. The latest cluster check showed
no queued or running jobs. All four final CSVs and histories are synced locally.

| Dataset | Family | Stationarity | Feasibility | Stop |
| --- | --- | --- | --- | --- |
| smallNORB | LASSO | 3.2633e-4 | 3.2433e-4 | time_limit |
| smallNORB | Logistic GD | 1.9268e-4 | 0 | inner_time_limit |
| realsim | LASSO | 1.5497e-2 | 7.1214 | time_limit |
| realsim | Logistic GD | 4.7179e-3 | 0 | inner_time_limit |

Regression parameter/initialization sweeps and final evaluation remain pending.
Twenty follow-up candidates are prepared but unsubmitted. Feasibility progress
for realsim LASSO and first-inner-solve cost for logistic GD require further
pilot work before expanding the study.

## Status update — November 3, 2026, 22:27 UTC

Tasks 0 and 1 finished without reaching the common 1e-6 tolerance. The
smallNORB LASSO run reached its 7,200-second solve limit, with stationarity
3.2633e-4 and feasibility 3.2433e-4. The smallNORB logistic GD run reached
an inner time limit after 5,224.9 total solve seconds, with stationarity
1.9268e-4. These are recorded convergence-budget outcomes, not successful
evaluations. Both realsim tasks (2 and 3) are RUNNING. The 20 follow-up
candidates remain unsubmitted; full regression sweeps and final evaluation
are pending pilot convergence.

The original submission record follows.

Submitted array **16759787**, tasks 0–3, with at most two tasks running.
The first scheduler check showed all four pending in the CPU partition.

| Task | Dataset | Family | Inner solver | Ratio | Sigma | Stride |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | smallNORB (wide) | LASSO QP | Nystrom-PCG, rank 128 | 1e6 | 0.2 | 1 |
| 1 | smallNORB (wide) | Logistic regression | Vanilla GD | 1e4 | 0.9 | 1 |
| 2 | realsim (tall) | LASSO QP | Nystrom-PCG, rank 128 | 1e6 | 0.2 | 1 |
| 3 | realsim (tall) | Logistic regression | Vanilla GD | 1e4 | 0.9 | 1 |

Manifest: `regression_v1/inputs/initial.csv`, sealed against the frozen v7
numerical archive and campaign source seal. Output directory:
`results/regression_recovery_v1_20261103`.

Every task: independent setup, warm inner initialization, one core/BLAS thread,
64 GB allocation, 56 GB worker limit, 7,200-second solve budget,
3,600-second inner budget, 900-second setup allowance in the worker hard
wall timeout, 2:30:00 Slurm limit. Slurm test-only validation passed before
submission. The completed realsim pilot 16536565_0 justified reusing the
one-core allocation (1,937 CPU seconds / 1,947 elapsed seconds); its memory
report was not used to lower the existing 64 GB request.

Validation: Ruff passed; all 321 tests passed. New tests check objective and
data preservation, finite budgets, warm/fixed/Gaussian candidate coverage,
manifest identities, content seals, and refusal to overwrite existing inputs.
No frozen numerical or campaign source files were changed.

Twenty follow-up candidates are prepared but unsubmitted. Inspect these four
results and their setup costs/histories before selecting penalty neighbors,
initialization ablations, and longer PDHG runs. Full tuning and held-out
evaluation remain subsequent stages, not completed by these pilots.

## Follow-up submission attempt

The user authorized the 20 prepared follow-up pilots. Slurm test-only validation
rejected both the 20-task array and a reduced two-batch array with
`allocation failure: MaxJobCount limit reached`. No follow-up job ID has been
accepted yet. A local bounded submission process retries that specific global
capacity rejection every five minutes for up to one hour. It stops immediately
on connection errors or any different failure; an ambiguous actual submission
cannot be blindly repeated. Its live state is recorded in
`regression_v1/followup_submission_status.csv` and its helper output in
`regression_v1/followup_submission.log`.

The pending layout has two serial batch tasks, each running ten independent
workers. Every configuration retains its 7,200-second solve limit, 3,600-second
inner limit, 900-second setup allowance, iteration caps, and independent setup
accounting. Each batch requests one core, 64 GB, and 22:45:00, covering ten
8,100-second worker hard limits plus shutdown headroom. Results will be written
to `results/regression_followups_v1_20261104/`. No changes to the sealed manifest
or numerical engine were needed; the manifest was uploaded in the original
regression setup.
