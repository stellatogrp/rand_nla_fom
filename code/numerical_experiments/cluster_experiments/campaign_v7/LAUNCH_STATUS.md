# Initial v7 critical-path launch

Submitted representative pilots:

| Work | Job | Manifest task |
| --- | --- | --- |
| QP: 1,000,000 variables, Nyström rank 128, warm start | 16536093 | 660 |
| Kaczmarz: 64,000 variables, block size 512, stride 1 | 16536098 | 72 |
| Independent wide smallNORB LASSO, Nyström rank 128 | 16536104 | 0 |
| Independent tall realsim LASSO, Nyström rank 128 | 16536565 | 8 |

All jobs passed `sbatch --test-only` before submission. The last observed state
of the first three jobs was pending for resources/priority. No success or
convergence is assumed until their result CSVs and resource statistics exist.

At the initial pilot launch, the following full requests had passed test-only
validation and were awaiting promotion. See the production update below for
the subsequent submissions:

- QP: 2,079 configurations in 260 batches of eight; concurrency eight;
  16 GB allocation / 12 GB worker cap; 5 h 30 min Slurm limit per batch.
- Kaczmarz: 174 screening configurations; concurrency one; 4 GB allocation /
  3 GB worker cap; 15-minute Slurm limit and 120-second solve cap per case.
- Real-data pilots: 16 independent configurations across tall realsim and wide
  smallNORB, LASSO/logistic, two parameter scales, and PDHG references. The first
  tall/wide pair is submitted. Inspect them before releasing the remaining
  pilot cases, keeping total regression concurrency at most two.

Run `stats` on concrete pilot task IDs (`16536093_0`, `16536098_0`,
`16536104_0`, `16536565_0`), inspect setup/solve costs and common residuals, then
release the corresponding arrays through the slurm skill. Retain all failed or
censored results. This is a pilot-gated campaign, not an automatic promotion.

Next phases are prepared in `advance.py`: require complete tuning before
creating the steps/penalty grid, and complete steps before evaluation on
held-out seeds 3, 4, 5. A successful screen is required before Kaczmarz expansion;
independent real-data pilot convergence is required before scaling regression
tuning. See `README.md` for the exact follow-up commands.

The campaign inputs/source are sealed and archived under `snapshots/`.
No existing numerical release, result CSV, or previously launched sweep was
modified. Local validation: 319 tests passed; Ruff and formatting checks passed.

## Production promotion after completed resource pilots

The million-variable Nyström pilot converged: approximately 126 seconds clean
wall time, including approximately 58 seconds of solver setup. Slurm reported
130 elapsed seconds, 125 CPU seconds, and 6,162 MB maximum RSS. The largest-block
Kaczmarz screen pilot reached its 120-second solve cap cleanly; Slurm reported
124 elapsed seconds, 120 CPU seconds, and 148 MB maximum RSS.

After inspecting those measurements, the following were submitted:

- **16543722:** full QP tuning, 260 batches / 2,079 configurations, concurrency 8.
- **16543752:** full Kaczmarz screen, 58 batches of three independent cases /
  174 configurations, concurrency 1. Each batch has a 45-minute hard Slurm limit;
  each case retains its own 120-second solve cap and separately timed setup.
  The original 174-job shape was rejected by the QoS submission limit; serial
  batching reduced job count without changing scientific budgets or sharing setup.
- **16543771:** dependent steps-manifest planner, after any completion of 16543722.
- **16543777:** dependent Kaczmarz-expansion planner, after 16543752.

The two independent real-data pilots (16536104_0 and 16536565_0) were running
at the last check. The full QP array had started producing results. The
Kaczmarz array was pending on the site QoS running-job limit. Planning jobs
remain dependency-gated and refuse incomplete or unresolved results.

Initial production result capture after promotion: 188 of 2,079 corrected QP
configurations had results, all converged. The Kaczmarz directory held the one
censored resource-pilot result; the rest of the screen was queued. The two
independent real-data pilots were still running without final result CSVs.
These are point-in-time counts, not a claim that later jobs have finished.

## November 1 completion check and next-stage launch

The corrected QP tuning finished all 2,079 configurations: 2,073 converged,
three OSQP worker timeouts, and three OSQP solver errors. Screening finished
all 174 cases at the 120-second solve cap with no converged arm. The expansion
planner completed and correctly emitted no expansion. The steps planner
completed and generated 1,770 configurations under
`results/campaign_v7_plans/qp_steps.csv`, with a matching source/manifest seal.

The first tall/wide independent LASSO pilots both reached their 30-minute solve
caps. Neither met the common tolerance. Their completed resource measurements
showed approximately 6 GB and 28 GB maximum RSS and near-full single-core CPU
use; the subsequent pilot requests retain the proven 64 GB allocation and
56 GB worker limit.

Next-stage jobs submitted after inspecting these results:

- **16583285:** QP stepsize pilot, steps manifest task 1500 (one-million-variable
  rank-128 Nyström case, ratio 0.1), before releasing the full steps array.
- **16583289:** remaining 14 independent regression pilots, tasks 1-7 and 9-15,
  concurrency at most two. The original pilot records are preserved.

Evaluation remains pending completion of the steps phase; full regression
tuning remains gated on convergent independent pilots. No Kaczmarz expansion
is scheduled when the screen has no eligible pair.

## Completion of next-stage pilots and full steps submission

All 16 independent real-data pilots finished: one converged, twelve reached
the solve-time limit, and three reached an inner-solve time limit. The sole
converged case was smallNORB logistic regression with PDHG at baseline step 1,
stationarity approximately 9.60e-7, in about 1,281 seconds of independently
measured wall time. None of the LASSO or logistic-GD pilots met the common
tolerance. The generated `regression_readiness.csv` records this explicitly;
full regression tuning remains blocked on convergence for those groups.

The QP steps pilot converged in approximately 397 seconds. Slurm reported
400 elapsed seconds, 396 CPU seconds, and 6,155 MB maximum RSS on one core.
After inspecting these measurements, job **16606062** was submitted for the
full 1,770-case stepsize/penalty sweep (222 batches of eight, concurrency eight,
16 GB allocation / 12 GB worker cap, 5 h 30 min per-batch Slurm limit).
Held-out evaluation remains dependent on successful completion and selection
from that sweep.

The full steps array was observed running eight one-core workers. Dependent
held-out evaluation planner **16606094** is queued after job 16606062 and will
refuse incomplete or unresolved inputs. It generates the evaluation manifest;
held-out solves have not started yet.
