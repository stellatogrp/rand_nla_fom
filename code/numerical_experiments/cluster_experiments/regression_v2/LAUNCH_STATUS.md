# Next regression batch submission

## Completion update

All four tasks in array 16907671 are COMPLETED, and their final result CSVs
and histories are synced locally. All four recorded optimization outcomes are
time_limit at the 7,200-second solve budget; no run reached the common 1e-6
certificate. No jobs for this experiment project remain in the live queue.

| Task | Dataset / family | Stationarity | Feasibility |
| --- | --- | --- | --- |
| 0 | realsim LASSO | 1.2187e-6 | 1.0055e-7 |
| 1 | realsim logistic GD | 2.3179e-3 | 0 |
| 2 | smallNORB LASSO | 1.6287e-5 | 1.6204e-5 |
| 3 | smallNORB logistic GD | 2.9573e-4 | 0 |

True-certificate audits cost 1,636.7 seconds for realsim LASSO (22.7% of solve
time) and 1,991.2 seconds for smallNORB LASSO (27.6%). These costs are included
in the recorded solve and independent worker times. Representative scheduler
CPU efficiencies were about 99.6% (task 0) and 99.5% (task 3), on one core.
There were no crashes or out-of-memory outcomes. The next priority is reducing
certificate overhead while retaining true-gradient checks, then further
regression convergence pilots before broad sweeps and final evaluation.

The original submission record follows.

Array **16907671**, tasks `0-3%2`, was accepted after successful test-only
validation. It runs four pilots from `regression_v2/inputs/pilots.csv` using
the sealed `regression_v9` release. Each task requests one core, 64 GB, and
2:30:00; at most two run concurrently. Results are independent and go to
`results/regression_v2_20261105/`.

The batch audits realsim LASSO stopping at ratio 1e4, tests smallNORB LASSO
at ratio 1e5, and tests logistic GD with ratio 100 and sigma 0.99 on both
datasets. Sigma remains 0.2 for LASSO; stride remains 1 for all four runs.
The normalized original-problem certificate remains 1e-6. Explicit true
certificate checks replace the extra graph gate; all added costs are charged.
Original numerical releases, data/objectives, and historical results are preserved.

Every run retains 7,200 solve seconds, 3,600 seconds per inner solve,
100,000 total inner updates, 10,000 per outer solve, 10,000 outer solves,
and an 8,100-second worker wall cap. Slurm provides the final 2:30:00 stop.
The worker address-space limit is 56 GB.

Validation: Ruff passed; all 347 tests passed, including the graph-gate
counterexample, true-certificate rejection of nonoptimal points, invariant GD
updates, finite certificate cadence, and rejection of uncalibrated optimal
check-cost prescriptions. Immutable numerical and launcher hashes are verified
before each worker starts.
