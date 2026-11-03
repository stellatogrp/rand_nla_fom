# LASSO and logistic regression recovery pilots

These pilots use the sealed `accounting_v7` numerical engine and existing
Rahimi–Recht features: smallNORB (24,300 by 30,000, wide) and realsim
(72,309 by 50,000, tall). Native labels, feature provenance, scaling, LASSO
lambda, logistic ridge, and the common 1e-6 certificate are unchanged.
LASSO uses the nonnegative split-variable QP formulation.

Run `uv run python -m regression_v1.prepare --output-dir regression_v1/inputs`
once to create immutable, content-sealed manifests accepted by
`campaign_v7.runner`. No change to a sealed numerical release is needed.

The four initial pilots test LASSO Nystrom-PCG (rank 128, ratio 1e6,
sigma 0.2) and logistic vanilla GD (ratio 1e4, sigma 0.9), each with stride 1
and warm inner initialization. Here ratio = L/c = L*gamma. The smaller
LASSO penalty tests whether outer progress improves; the larger allowed
logistic inexactness tests whether the first inner solve can finish without
oversolving. These are empirical choices, not claimed optimal parameters.

Each independent worker charges matrix loading/validation, construction,
calibration, preconditioner setup, and solve costs. Existing feature preparation
provenance is retained, including its incomplete prior-release campaign cost
flag. There is no cross-run setup amortization. Every run has a 7,200-second
solve budget, 3,600-second inner-solve budget, 100,000 total inner updates,
10,000 updates per outer iteration, and 10,000 outer iterations. The worker
has a separate 900-second setup allowance and hard wall timeout, with a
2:30:00 Slurm allocation as the final stop. Memory: 64 GB allocated and a
56 GB worker address-space limit; one core, one BLAS thread, no GPU.
Use at most two concurrent regression tasks, with one task per array element.

`followup_candidates.csv` contains 20 unsubmitted candidates: penalty
neighbors, fixed-zero and Gaussian inner initialization for each method and
dataset, and longer PDHG baselines. The warm pilots are the corresponding
initialization reference. Inspect convergence histories and resource stats
before choosing follow-ups. Then run sigma/stride tuning (including verified
model prescriptions where applicable), step/penalty tuning, initialization
ablations at selected settings, and held-out seed evaluation. Keep every
censored outcome; do not promote an unresolved or failed pilot to a main run.
OSQP/SCS representation limits on full dense features remain documented in
the earlier pilot results; OSQP does not apply to the logistic objective.
