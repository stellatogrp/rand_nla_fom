# Fresh QP figure campaign — November 3, 2026

## Completion update — November 3, 2026, 22:27 UTC

All 183 evaluation runs converged. All four PNG/PDF figures and supporting
CSVs were generated and downloaded to `main_figures/fresh_v8/`. The theory
comparison contains 35,758 GD/CG observations and no exceedances with the
recorded numerical slack; it tests inner contraction, not an outer KKT rate.
Representative full-array million-variable jobs used 24 CPU seconds over
25 elapsed seconds (GD, task 120) and 119 over 121 (rank-128 Nystrom-PCG,
task 165), approximately 96% and 98% CPU efficiency on one core.

The original submission record follows.

- Numerical release: `figure_v8`, archived hash
  `f8be21a4c85ab637` (prefix; full hash in the release seal).
- Plotting snapshot: `reporting_v2`, archived hash
  `ec89f379a0625236a4e8aa863ef2a90fbc38fb09d2dc27dcb6cfa433b21041ed`.
- Pilot array: **16761759**, tasks 0 and 1, both COMPLETED and converged.
  Million-variable CG: 26 scheduler seconds / 23 CPU seconds, 22.85 measured
  worker seconds, 382 MB reported RSS. Rank-128 Nystrom-PCG: 118 scheduler
  seconds / 110 CPU seconds, 110.33 measured worker seconds, 6,105 MB RSS.
  CPU efficiency was about 88% for the short CG process and 93% for Nystrom;
  one core is already the minimum. Memory reports were not used to reduce
  the established 16 GB allocation.
- Both pilot traces were complete; all 395 recorded CG observations satisfied
  its contraction bound with numerical slack. The pilot runs occupy evaluation
  indices 21 and 162 and are reused without retiming or shared setup.
- Remaining evaluation array: **16762036**, tasks
  `0-20,22-161,163-182%8` (181 runs; at most eight concurrent QP tasks).
- Dependent plotting job: **16762080**, `afterok:16762036`.
- Results: `results/main_figures_v8_20261103/`.
- Final four-figure destination: `main_figures/fresh_v8/` (PNG and PDF plus CSVs).
- Historical three-figure summary: `main_figures/previous_v7/`. It has no inner
  theory plot because v7 did not record the required inner residual histories.

All submissions passed test-only validation. Every evaluation uses independent
setup and finite time/work/outer/inner limits. Active regression pilots remain
untouched. Frozen sources are excluded by `slurm/figure_v8.env`.

Local validation: Ruff passed; 330 tests passed. A six-run smoke study generated
all four figure types and checked 4,402 inner observations against the verified
GD/CG bounds without any exceedances. These tiny smoke runs are validation
artifacts under a temporary directory, not scientific main results.
