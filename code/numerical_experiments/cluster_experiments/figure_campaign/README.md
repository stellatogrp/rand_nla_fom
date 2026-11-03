# Fresh QP summary-figure campaign

This campaign repeats the held-out QP evaluation from fresh initial states with
scalar inner histories and true-gradient outer-boundary audits. The previous
v7 results are retained. All setup, calibration, trace work, trace CSV I/O,
process startup, and solver execution remain in worker wall time. Never pool
these times with untraced releases or amortize setup across configurations.

The 183 configurations cover n=64,000, 262,144, and 1,000,000; seeds 3, 4, 5;
GD, AGD, CG, GMRES, and Nystrom-PCG ranks 32/128; and warm/fixed/Gaussian
inner initialization. Warm-selected parameters are reused for all three
initialization modes. Any model-selected sigma/stride is resolved from the
recorded warm v7 evaluation, then fixed across modes. Native PDHG/SCS and the
eligible 64,000-variable OSQP baseline are retained once per seed. OSQP at
larger sizes was not promoted by the previous screening; absent evaluations
are labeled rather than presented as timed successes. Kaczmarz has no converged
screening configuration and is not promoted to this held-out evaluation.

`prepare.py` freezes the new `figure_v8` numerical release, creates sealed
manifests, and records selection provenance. `runner.py` verifies the archive,
manifest, launcher, and plotting source hashes before every worker. The
numerical v7 releases and regression campaign are untouched.

Pilot two representative million-variable configurations (CG and rank-128
Nystrom-PCG); inspect completed stats before expanding. The full array uses
one configuration per task, at most eight concurrent QP tasks, one CPU and
16 GB per task, with a 12 GB worker address limit. The solve limit remains
1,800 seconds; the independent worker has an additional 600-second setup
allowance in its hard timeout. The Slurm limit is 45 minutes. The existing
two-task regression lane remains separate. No GPUs, partition override, or
QoS override are requested. All submissions use the Slurm skill helper and
pass test-only validation first.

The final plotting job depends on successful completion of the evaluation
array. It requires every manifest result, preserves convergence/resource
censors, and refuses ambiguous histories or incomplete trace accounting.
Scientific interpretation of theory comparisons uses only the verified inner
GD/CG contraction constants; no Nystrom/AGD/GMRES bound is fabricated.
