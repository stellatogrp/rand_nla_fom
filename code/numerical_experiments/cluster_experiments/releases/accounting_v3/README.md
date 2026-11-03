# Accounting release v3

This separate source tree preserves the numerical implementation used by the
already-running root `speed_v2` QP study. New jobs explicitly set
`EXPERIMENT_RELEASE=accounting_v3`; `slurm/common.sh` then selects this directory.
Data, manifests, results, and Slurm scripts are relative links into the project.
Root data/results stay excluded from upload; the release links are uploaded.

## Independent main runs

Main manifests use `setup_policy=independent`. `experiments.batch` still launches
a new supervised worker for every configuration. Every worker validates and
loads its own input, constructs its own problem, calibrates its inner method,
and constructs its own Nyström preconditioner or block cache. The setup built
for calibration is reused only within that same run and is charged once.

`experiments.run` records each attempt under `RUN/attempts/`. Interrupted main
runs restart independently, preserving prior attempts and measured costs.

| Field | Meaning |
| --- | --- |
| `total_seconds` | Parent-measured worker wall time, including startup, loading, setup, tuning, solving, reporting, and all recorded retries. |
| `attempt_worker_wall_seconds` | Measured wall time of the latest attempt. |
| `recorded_attempt_seconds` | Known measured attempt time; a lower bound if an earlier attempt was hard-killed without a terminal record. |
| `generation_seconds`, `tuning_seconds`, `solver_setup_seconds` | Measured components of the latest attempt. |
| `cumulative_*` | Corresponding recorded setup components across attempts; unknown components remain NaN. |
| `setup_accounting_complete` | False if a prior attempt lacks its final timing record. In that case `total_seconds` is NaN and the run cannot win a main comparison. |
| `data_preprocessing_seconds` | Separately recorded artifact construction cost, with scope and completeness flags. It is not silently amortized. |
| `end_to_end_seconds` | Sum of measured worker time and one artifact-build reference. This is explicitly **not** one contiguous wall-time measurement. NaN if either part is incomplete. |

Do not sum `end_to_end_seconds` across configurations: campaign totals must count
each shared artifact build once. Historical failed preparation jobs from prior
releases are not reconstructed; `preprocessing_campaign_accounting_complete`
explicitly remains false. OS filesystem caches are not flushed. These are
independent-process runs, not a claim of cold operating-system caches.

Both manifest selection and default plots reject shared-pilot or incomplete
main timing records. Implementations, thread counts, and timing scopes are
separated in plot groups and summary CSVs. `--include-pilots` explicitly shows
incremental pilot timings with a distinct axis label.

## Preparation and pilot speedups

`feature_store` commits checksummed feature-column chunks atomically. A resume
verifies input parameters and every reused chunk. Construction and retries are
timed in durable journals, including raw-data loading. SIGTERM/USR1 retain
committed chunks; an unobserved hard kill leaves a visible accounting gap.

Optional job-local temporary space is private and capacity-checked. Final
assembly takes place in bounded RAM, followed by a sequential C-order NPY write.
Unverified legacy outputs are quarantined, never treated as valid cached data.
Committed chunks are removed only after valid artifact metadata is written;
the journals remain. `features_v3.slurm` defaults to 64 GiB and one core.

Each dataset has its own preparation/pilot chain. Two serial lanes enforce the
two-task cap. Failure cancels that dataset's pilot; `afterany` lane links allow
the next independent dataset to proceed. `--kill-on-invalid-dep=yes` prevents
dead dependency jobs from keeping a lane blocked.

`pilot_batch` is restricted to `shared_pilot` manifests in pilot/smoke/profile
phases. Its one-entry problem cache avoids holding two full resident matrices.
It records `shared_setup.csv` and actual `batch_accounting.csv` wall times.
Individual CSVs identify their shared setup event, record incremental solve
time, and have `primary_eligible=False`. A parent watchdog bounds each active
configuration and the whole pilot batch. Main manifests cannot use this cache.

Dense value/gradient and Gram-product kernels accumulate forward and adjoint
work over the same blocks without forming a Gram matrix. Objective values feed
diagnostics without another matrix pass. The same common tolerance and check
stride remain in force; tests compare these kernels to independent formulas.

## Selective cores

`core_policy` reads completed profiling CSVs and actual Slurm stats through the
skill helper. It requires matching hardware, repeated converged solves, at
least 1.2x speedup, and at least 90% **whole-job** CPU efficiency. The current
policy selects two cores for the measured million-variable/rank-128 Nyström QP
case; other cases remain serial. Requests and `EXPERIMENT_THREADS` must match.
Workers reject a CPU model that does not match the policy. A matched one-/two-core
ablation uses independent main workers and charges every setup in both arms.

No completed result CSVs or historical source files are rewritten by this release.
