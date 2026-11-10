# Regression certificate audit v9

This release copies the sealed accounting_v7 engine and adds an opt-in
`terminal_policy=common_certificate`. It applies the same original-problem
stationarity/feasibility certificate used by the baselines, without requiring
an additional unnormalized graph residual to pass first. A true-gradient
certificate is checked initially, at accepted outer boundaries, every ten
inner updates by default, and before budget/interrupt exits. For recurrence
solvers, certificate gradients are recomputed independently. Numerical inner
updates, data/objectives, RNG streams, and setup accounting are preserved.

All additional operator work and certificate time are included in measured
worker/solve time. Certificate counts and time are exported explicitly. The
legacy graph-gated policy remains available for controlled auditing. Old
results are never relabeled or overwritten.

The old check-cost calibration omits the new certificate schedule. New-policy
pilots therefore use explicit sigma/stride values; requesting the old 'optimal'
selection raises an error pending a new cost calibration. Inner contraction
constants remain reference quantities, not a validated full stopping-cost model.
