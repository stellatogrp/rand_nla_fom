# Captured source history

This log describes source states captured during the validation work. Earlier
intermediate review states were not saved and are not reconstructed here.
Machine-readable notes and hash links are in `history.jsonl`; archive identifiers
and per-file hashes are recorded under `seals/` and `report_seals/`.

- **accounting_v6:** froze the Claude-reviewed implementation before pilots.
  Eleven of twelve cluster pilots passed. Small-block Kaczmarz exposed a
  floating-point budget-boundary bug: 64,001 updates under a 64,000-update cap.
  The source, failed audit, and original result remain preserved.
- **accounting_v7:** separate correction using integer update limits for total
  and per-outer budgets, with reported work derived from update counts. The
  targeted cluster Kaczmarz pilot passed at exactly 64,000 updates. All twelve
  v7 cluster pilot audits subsequently passed; original CSVs and generated
  reports are under `results/validation_v7_20261031/`.
- **accounting_v3 / v4 / v5:** archived the currently available source versions
  and current shared launch/validation dependencies for preservation. These
  captures are not claims about exact historical launch environments.
- **reporting_v1:** publication reporting uses measured design forward/transpose
  products as the primary regression work metric, retains Hessian counts as
  diagnostics, and exports measured work, historical estimates, and unknown
  counts into separate tables. Summaries separate scopes and hardware and show
  work to convergence only for successful runs.

For each future review, capture a new source label with a concise description
before and after editing. Do not overwrite an old seal, snapshot, or history
entry. Numerical and reporting capture commands are documented in
`../EXPERIMENT_VALIDATION.md`. Back up this directory together with the code.
