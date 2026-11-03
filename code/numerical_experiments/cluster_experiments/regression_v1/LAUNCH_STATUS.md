# Launch status — November 3, 2026

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
