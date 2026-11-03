# Frozen releases, validation, and publication work counts

The initial reviewed source, `accounting_v6`, was sealed before cluster
validation. Eleven of its twelve cluster cases passed. The Kaczmarz audit
caught 64,001 row updates under a 64,000-update cap, caused by floating-point
accumulation. That failed result is preserved. `accounting_v7` is the separately
sealed correction: budget enforcement uses integer update counts. Existing root/v3/v4/v5 job code is excluded from validation
syncs. No production sweep is promoted automatically by these tools.

## 1. Validate before promoting a release

Run from this standalone project:

```bash
uv run python -m validation_v7.releases verify --release accounting_v7
uv run python -m validation_v7.pilots \
  --output release_history/pilots/accounting_v7_validation_20261031.csv
```

The immutable pilot manifest records the numerical archive hash. Every task
verifies the release and manifest before running. It uses independent setup,
one BLAS thread, a 60-second solve cap, finite inner/outer iteration caps, a
120-second parent setup allowance, and a Slurm hard limit. Small-block Kaczmarz
is deliberately capped at one pass of inner-update work. Its checks and
calibration are still measured and charged.

There are twelve cases: QP with 64,000 variables and condition number 10,000
(CG, GD, Nyström-PCG, and block-size-one Kaczmarz), and GD/PDHG for tall and wide
LASSO and logistic problems. The regression designs are seeded sparse random
matrices, 16,384 by 4,096 before transposition. These are offline validation
cases, not evidence of performance on the NysADMM datasets. The local smoke
variant uses smaller data and is not a cluster performance measurement.

Use the slurm skill for all cluster operations. Before each sync:

```bash
python3 /Users/pranav/.agents/skills/slurm/scripts/slurm_agent.py \
  sync up --config slurm/validation.env --dry-run
```

Review the transfer: it must not modify the code or manifests used by current
jobs. Then sync with the same configuration. Validate the first task with
`submit slurm/validation_v7.slurm --array 3 --cpus 1 --mem 8G --time 00:10:00`,
exporting the manifest path above and a unique `EXPERIMENT`. Inspect its audit,
logs, and `stats` before submitting `--array '0-2,4-11%1'`. Existing lane caps still
apply; validation uses one profiling slot and does not increase sweep caps.
The completed first pilot supports one core; the remaining bounded tasks use
five-minute Slurm limits. Memory reports do not establish a safe lower limit.

Every task writes an audit CSV alongside its original result. A completed job
means both the worker and its audit passed. After syncing results:

```bash
uv run python -m validation_v7.audit \
  --manifest release_history/pilots/accounting_v7_validation_20261031.csv \
  --results-dir results/validation_v7_20261031 \
  --output results/validation_v7_20261031/accounting_audit.csv
```

The audit checks identities, setup policy/timing, finite convergence
certificates, recorded operator work, and iteration caps. It accepts honest
nonconvergence under the cap; it does not certify solver speed or prove a
convergence theorem. Keep the full sweep on hold if validation fails. Do not sync changes to the
shared history index while validation tasks are starting: update archives and
the index together before submission or after the validation array finishes.

## 2. Publication work metrics

For LASSO and logistic regression, the primary work unit is the sum of measured
forward and transpose design-matrix vector products. One `A @ x` and one
`A.T @ y` count as two products. These counts cover the same post-build scope,
including calibration and inner setup, across GD/ADMM and PDHG. Disk passes,
floating-point operations, matrix generation, and native backend internals are
not implied by this unit. Setup/loading/preparation times remain independently
charged in the existing timing fields.

QP uses measured quadratic-operator vector products. Hessian counts are
secondary diagnostics for regression, particularly LASSO PDHG, where they count
only metric evaluations. OSQP/SCS work stays unknown when their internals are
not exposed; their wall-clock comparisons remain available.

```bash
uv run python -m plots.publication \
  --results-dir results/your_evaluation_directory \
  --output results/publication_tables
```

`measured_work.csv` and `measured_summary.csv` contain measured work only.
`historical_estimates_work.csv` and its summary hold historical QP estimates
with explicit, separate scopes. `unknown_work.csv` preserves runs without a
comparable work measurement. Neither estimates nor unknown counts are pooled
with measured values. Pilots/screens are excluded unless `--include-pilots` is
explicit. Censored runs remain in the run tables; work-to-convergence summaries
use converged runs only and report their count. Summaries retain all parameters
except seed and never mix units, evidence, or work scopes. Repeated run IDs in
multiple inputs are rejected rather than silently double counted.

`plots.report` retains the existing colorblind-friendly figures, adds the
primary measured work summary, and keeps Hessian counts marked diagnostic.
Original result CSVs are read, never edited.

## 3. Preserve each reviewed source state

`release_history/history.jsonl` is an append-only, hash-linked capture log. Each
entry points to a read-only content-addressed tar archive. Numerical seals
record every source hash plus shared launch scripts, dependencies, and validation
tools. A refreshed `source_manifest.json` cannot reseal an altered release.
`freeze` is idempotent for identical sources and rejects edits under a sealed
name. Use a new release directory for subsequent numerical changes.

```bash
uv run python -m validation_v7.releases freeze --release accounting_v8 \
  --note 'Concrete changes and validation performed in this review'
uv run python -m release_tools.reporting --name reporting_v2 \
  --note 'Concrete changes to reporting since reporting_v1'
```

Capture a new label before and after each future review. Reporting snapshots
preserve the plotting/export sources separately from frozen numerical code.
Use `tar -tf release_history/snapshots/<sha256>.tar` to inspect an archive;
extract into a new temporary directory to compare versions. Do not extract over
a running project or overwrite a previous snapshot, seal, or history entry.

The initial captures preserve the source versions available now. They do not
reconstruct intermediate review states that were never saved. The small change
log is in `release_history/CHANGELOG.md`; machine-readable notes live in each
history entry. Read-only permissions and hash checks are tamper-evident local
controls, not administrator-proof immutable storage. Commit/archive the history
and snapshots together in durable storage; group scratch is not backed up.
