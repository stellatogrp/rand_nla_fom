# studies_v1 launch status

## 2026-11-06: calibration and pilots submitted

Approved by the user: the three calibrate arrays and one pilot per lane; the
full scale (instances per study) is decided from the pilots. The campaign as
sealed has 10 QP, 3 Kaczmarz, and 5 regression instances per study; a change
of instance counts only regenerates the argument files and the seal (the study
settings, calibrations, and setups do not depend on them).

Before submission: Ruff and all 387 tests passed; `sync up --dry-run` listed
only the `studies` sources, the campaign inputs, tests, sync configurations,
and notes, and nothing sealed or frozen; every submission passed
`sbatch --test-only`. All jobs: one CPU, `--constraint=icelake`, no GPU, no
partition or QoS override.

| Job | What | Array | Memory, time | Dependency |
| --- | --- | --- | --- | --- |
| 16984630 | QP calibrate (18 studies) | `0-17%8` | 16 GB, 0:30 | |
| 16984662 | Kaczmarz calibrate (4 studies) | `0-3%1` | 8 GB, 0:30 | |
| 16984664 | regression calibrate (6 studies) | `0-5%2` | 16 GB, 0:30 | |
| 16984676 | QP pilot: `admm_qp_side100_nystrom_pcg_rank128`, near-exact, 10 instances | `35` of `qp.run.part000.csv` | 16 GB, 1:00 | `afterok:16984630` |
| 16984678 | Kaczmarz pilot: `admm_qp_side40_block_kaczmarz_block32`, near-exact (600 s cap), 3 instances | `4` of `kaczmarz.run.part000.csv` | 8 GB, 1:00 | `afterok:16984662` |
| 16984680 | regression pilot: `ppm_logistic_random_normal_gd`, near-exact, 5 instances | `11` of `regression.run.part000.csv` | 16 GB, 1:00 | `afterok:16984664` |

The pilots request one hour rather than the run arrays' provisional two
because `--test-only` placed two-hour jobs about a day and a half out, and
laptop timings put each pilot near 30 minutes; a pilot that runs out of time
still stores every finished run. Thirty-minute jobs run in QoS `test` (two
running jobs per user), as the queue shows.

Next: inspect the calibrations (`configurations.csv`, setups) and the pilots'
`stats`, size `--time` and `--mem` of the run arrays, choose the instance
counts, and submit the run and collect arrays after the user's approval.

## 2026-11-06: calibration and pilot results

All 28 calibrate tasks completed and set up every instance (10 per QP study,
3 per Kaczmarz study, 5 per regression study); a million-variable setup took
about 2 minutes (Lanczos 52 s, Nystrom sketch 60 s). The 30- and 60-minute
jobs ran in QoS `test`, two at a time (`QOSMaxJobsPerUserLimit`). Calibration
hosts were icelake nodes of two models (Xeon Gold 6342, Platinum 8380). One
Nystrom-PCG rank-128 step at a million variables costs 201 ms (GMRES: 42 ms,
GD: 18 ms), from the preconditioner's rank-128 products.

| Pilot | State | Wall | Per instance | Outcome |
| --- | --- | --- | --- | --- |
| 16984676_35, QP Nystrom-PCG rank 128, side 100, near-exact | FAILED at the one-hour limit (exit 138, the Slurm warning signal) | 0:58 | 1,009-1,300 s | 2 of 10 instances finished and stored, both converged, K = 140 |
| 16984678_4, Kaczmarz block 32, near-exact | COMPLETED | 0:30 | 600 s (the cap) | all 3 stopped at the time cap with K = 0 |
| 16984680_11, logistic GD, near-exact | COMPLETED | 0:47 | 517-597 s | all 5 converged, K about 480, about 70,000 GD steps |

Peak memory reported: 7.2 GB (QP pilot), 93 MB (Kaczmarz), 145 MB
(regression); CPU efficiency 99% on one core for all three. The QP pilot's
unfinished instances are rerun by its task index in the run array, which skips
the two stored ones.

## 2026-11-06: full run submitted (20 QP / 3 Kaczmarz / 10 regression instances)

After the pilots the user chose 20 instances per QP study and 10 per regression
study (Kaczmarz stays at 3) and approved the full submission. Only the
instance counts changed: `prepare --force` rewrote the 24 QP and regression
argument files, `campaign.json`, and the seal; the code, the manifests (261,
58, and 90 run tasks), the calibrations, the configuration lists, the existing
setups, and the pilots' finished runs are unchanged (the instance count is not
part of a study's identity). The calibrate arrays run again to set up the new
instances; they skip the stored calibrations and setups. Time limits scale
with the pilots' measurements: the longest QP task (Nystrom-PCG rank 128,
near-exact, side 100) is about 20 x 1,300 s = 7.2 h, the longest regression
task (logistic near-exact) about 10 x 600 s = 1.7 h; plain gradient descent on
the logistic problem took 12 s per instance locally. Before submission: Ruff
and tests passed, `sync up --dry-run` listed only the notes, the campaign
specification, its argument files, manifests, and seal, and the three new run
shapes passed `sbatch --test-only`.

| Job | What | Array | Memory, time | QoS | Dependency |
| --- | --- | --- | --- | --- | --- |
| 16996211 | QP setup of the new instances | `0-17%8` | 16 GB, 1:00 | test | |
| 16996213 | regression setup of the new instances | `0-5%2` | 4 GB, 0:30 | test | |
| 16996215 | QP runs | `0-260%8` | 16 GB, 9:00 | short | `afterany:16996211` |
| 16996227 | regression runs | `0-89%2` | 4 GB, 3:00 | short | `afterany:16996213` |
| 16996229 | Kaczmarz runs | `0-57%1` | 4 GB, 1:00 | test | `afterany:16996211:16996213` |
| 16996231 | QP collect | `0-17%8` | 4 GB, 0:30 | test | `afterany:16996215` |
| 16996234 | regression collect | `0-5%2` | 4 GB, 0:30 | test | `afterany:16996227` |
| 16996243 | Kaczmarz collect | `0-3%1` | 4 GB, 0:30 | test | `afterany:16996229` |

The Kaczmarz runs wait for both setup arrays because QoS `test` lets a user
run two jobs at once: the setups get both slots first. Collects run
`--allow-partial` and list unfinished runs in each study's `missing.csv`.

## 2026-11-07: complete

All 461 tasks of the eight arrays above ended COMPLETED. The results are synced
down to `results/studies_v1/` (1.7 GB). `campaign status` shows every study
calibrated and every planned run finished: 6,014 runs (5,000 QP, 174 Kaczmarz,
840 regression), every instance complete. No study has a `missing.csv`, no run
recorded an error, and no Slurm log contains a traceback, an error, or a
warning. Solver time summed over the runs is 78.6 h (QP 54.6, Kaczmarz 14.5,
regression 9.5; setup and ground truth excluded).

Outcomes: 5,812 runs converged (`||z_k|| <= 1e-9`) and 202 stopped at their
time limit:

| Runs | Study, configuration | Where they stopped |
| ---: | --- | --- |
| 40 | `admm_qp_side100_nystrom_pcg_rank128`, s sweep at s = 10 and s = 20, all 20 instances | 300 s per inexact run. s = 10 at about 130 of the 141 outer iterations the converged runs need (median `\|\|z\|\|` 4.6e-9, 1,290 inner steps); s = 20 at about 68 (median 3.4e-5, 1,340 inner steps). One Nystrom step costs about 0.2 s at a million variables. |
| 150 | every inexact Kaczmarz run (block and cyclic, blocks of 32 and 128) | 300 s, after 5 to 113 outer iterations, with `\|\|z\|\|` from 1.2e-5 to 0.87 (0.6 to 0.9 million inner steps for blocks of 32) |
| 12 | every near-exact Kaczmarz run | 600 s (`--near-exact-time-limit`), inside the first outer iteration |

Exact ADMM converged on every Kaczmarz instance. The Kaczmarz lane is censored,
as the workstation screen predicted; on Della the runs reach the time limit
before the `10^6` inner steps.

Theorem checks, over every run where they apply: Theorem 3.1 ratio at most 1 in
5,140 of 5,140 runs (largest 0.735); Lemma 3.2 minimum ratio at least 1 in
5,140 of 5,140 (smallest 1.15); Theorem 4.3 ratio at most 1 in 2,580 of 2,580
(largest 0.786); Theorem 4.2 ratio at most 1 in 2,658 of 2,658 (largest
0.785); per-step bound: no violation in 2,800 runs, with the surrogate or the
exact residual (largest ratio 1, as in the original `admm_qp` study).

Format: `results.csv`, `histories.csv`, `curves.csv`, and `summary.csv` have
the headers of the original `admm_qp` and `ppm_huber` tables (47, 19, 19, and
157 columns); `parameters.csv` has the original 45 columns followed by 16
cluster columns (`family` to `orientation`). `plots.studies` drew every study's
figures locally: 10 per study with deterministic contraction constants (GMRES,
GD, CG, logistic GD), 9 for block Kaczmarz (`bstar_model` but no `per_step`,
its bound holding in expectation only), and 8 for AGD, Nystrom-PCG, and cyclic
Kaczmarz (neither).

Sizing (`stats`): the longest QP task (16996215_35, Nystrom-PCG rank 128,
near-exact, side 100) ran 5:40 of 9:00 with 7.2 GB reported and 99% CPU
efficiency; the longest regression task (16996227_11, logistic near-exact) ran
1:02 of 3:00 with 145 MB and 99%. Kaczmarz task 16996229_4 took 4 s, since its
pilot had stored its runs.

Open, for the user to decide: rerunning the 40 censored Nystrom-PCG runs with
a longer per-run time limit changes the study settings, so it would be a new
campaign (`studies_v2`), not a resubmission here.
