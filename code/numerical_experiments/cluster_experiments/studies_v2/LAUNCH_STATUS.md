# studies_v2 launch status

## 2026-11-07: submitted

Requested by the user after `studies_v1`: rerun the Nystrom-PCG studies with a
longer limit and, for fairness, the other inner solvers under the same limit,
leaving out Kaczmarz. The per-run limit is 1,200 s, the value proposed with
the `studies_v1` results (the censored s = 20 runs need about 650 s).

Before submission: `prepare` sealed 24 studies (44 files); the manifests are
`studies_v1`'s QP and regression manifests with the campaign name changed, and
`tests/test_campaign.py` checks that every study's settings differ from
`studies_v1`'s in `time_limit` only. Ruff and all 389 tests passed;
`sync up --dry-run` listed 40 files (the `studies_v2` campaign, its two sync
configurations, the notes of both campaigns, the README, and
`tests/test_campaign.py`) and no `studies` source or sealed `studies_v1` file.
All jobs: one CPU, `--constraint=icelake`, no GPU, no partition or QoS
override; no pilots, since `studies_v1` ran these tasks with the same code.

`sbatch --test-only` placed every job longer than an hour at about
2026-11-13 01:15 and the half-hour jobs at about 2026-11-08 23:19; the QP
calibrate tasks started within a minute of submission anyway (QoS `short`),
with only `JobArrayTaskLimit` and `Priority` as pending reasons, as in
`studies_v1`, whose estimates were also hours off.

| Job | What | Array | Memory, time | QoS | Dependency |
| --- | --- | --- | --- | --- | --- |
| 17044846 | QP calibrate and set up 20 instances (18 studies) | `0-17%8` | 16 GB, 1:30 | short | |
| 17044848 | regression calibrate and set up 10 instances (6 studies) | `0-5%2` | 4 GB, 0:30 | test | |
| 17044895 | QP runs | `0-260%8` | 16 GB, 9:00 | short | `afterany:17044846` |
| 17044897 | regression runs | `0-89%2` | 4 GB, 2:00 | short | `afterany:17044848` |
| 17044899 | QP collect | `0-17%8` | 4 GB, 0:30 | test | `afterany:17044895` |
| 17044902 | regression collect | `0-5%2` | 4 GB, 0:30 | test | `afterany:17044897` |

Expected from `studies_v1` (`stats` and per-run times): the longest QP task,
the side-100 Nystrom-PCG rank-128 near-exact baseline (task 35), about 6.3 h;
the formerly censored s = 20 configuration (task 233), about 4 h and s = 10
(task 215) about 2.3 h; every other QP task at most about 2.6 h, 24 of them
over an hour. The longest regression task took 1:02 in `studies_v1`.

## 2026-11-07: runs pinned to one processor model

Della's `icelake` feature is advertised by three nodes only (an
`sbatch --test-only` that excludes them is rejected): `della-h12n14` and
`della-h12n15` (Xeon Platinum 8380) and `della-h17n4` (Xeon Gold 6342). In
`studies_v1` the exact-ADMM baseline, the same computation in every study of a
problem, took 8.3 s on the Platinum node and 10.7-12.2 s on the Gold one at
side 100 (all five LASSO baselines, on the Gold node, agreed within 1%). Each
configuration is its own task, so different solvers, and different
configurations of one solver, can be timed on processors 1.3-1.45x apart.

The user chose to pin the work that had not started to the Platinum nodes.
Cancelled while pending (nothing had run): QP calibrate tasks 14-17
(side 100: AGD, CG, Nystrom-PCG ranks 32 and 128), the whole QP run array
17044895, regression run tasks 2-89 of 17044897, and both collect arrays.
Resubmitted with `--exclude=della-h17n4` (except the collects, which time
nothing), chained so that the lane caps of `AGENTS.md` (eight QP, two
regression tasks at once) still hold:

| Job | What | Array | Memory, time | Dependency |
| --- | --- | --- | --- | --- |
| 17045201 | QP calibrate, side-100 AGD, CG, Nystrom-PCG (studies 14-17) | `14-17%4` | 16 GB, 1:30 | `afterany:17044846` |
| 17045204 | QP runs | `0-260%8` | 16 GB, 9:00 | `afterany:17045201` |
| 17045206 | regression runs 2-89 | `2-89%2` | 4 GB, 2:00 | `afterany:17044897` |
| 17045208 | QP collect | `0-17%8` | 4 GB, 0:30 | `afterany:17045204` |
| 17045210 | regression collect | `0-5%2` | 4 GB, 0:30 | `afterany:17044897:17045206` |

Left where they were: QP calibrate tasks 0-13 of 17044846 (0-5 done; 6-13,
sides 64 and the side-100 GMRES and GD studies, running on the Gold node) and
regression runs 0 and 1 of 17044897 (the exact-ADMM baselines of the LASSO
GMRES and GD studies, running on the Gold node); the regression calibrate
17044848 had finished. So every inexact and near-exact run is timed on a
Platinum node; the setups (Lanczos, the Nystrom sketch) are timed where their
calibrate task ran, which `calibration.json` records. `plots.solver_comparison`
compares solvers by algorithm time plus the solver's own setup (the sketch,
zero for GMRES), leaving out the projector and Lanczos setup that every solver
of a problem shares, and lists the processors of each row.

## 2026-11-08: runs finished; one task rerun

All QP and regression arrays ended by 08:33 on 2026-11-08. Of the 377 tasks
submitted after the pinning, 376 completed. Run task 35 (side-100
Nystrom-PCG rank-128 near-exact baseline, 17045204_35) was stopped by the
Slurm warning signal at 8:58 of its 9:00 (exit 138) on `della-h12n15` after 17
of its 20 instances; `missing.csv` listed seeds 17-19. With the user's
go-ahead to continue, task 35 was resubmitted (17100291, `--array 35`, 16 GB,
4:00, `--exclude=della-h17n4`; it skips the 17 stored runs) with the collect
of that study after it (17100293, `qp.collect.csv` row 17, `afterany`).

Results synced to `results/studies_v2/` (1.7 GB). Apart from those three
runs, every planned run finished: 5,867 runs, no error, every one converged
(`||z_k|| <= 1e-9`); none reached the 1,200 s limit, and the longest inexact
run took 590 s. Theorem checks over every run where they apply: Theorem 3.1
in 5,020 of 5,020 (largest ratio 0.735), Lemma 3.2 in 5,020 of 5,020
(smallest 1.15), Theorems 4.3 and 4.2 in 2,610 of 2,610 each (largest 0.786
and 0.785), and no per-step violation in 2,830 runs.

Against `studies_v1`, run by run: all 5,797 runs that converged in
`studies_v1` and are planned here reproduce their outer and inner iteration
counts and stopping reason exactly. The 40 runs censored there now converge
(side 100, rank 128, all 20 instances each: s = 10 in 140-141 outer iterations
and a median 309 s, s = 20 in 140-141 and a median 532 s, at most 590 s). The other 30 runs are new `s*` strides: the fresh
calibration put one off the grid for side-40 GD (20 runs) and LASSO GD (10).

Timing. Because those 5,797 runs repeat identical computations, the ratio
of their wall-clock times between the campaigns measures the nodes. On
`della-h12n14` in both campaigns the median ratio is 0.99 (10th to 90th
percentile 0.85-1.15, 1,195 runs). Runs moved from `della-h12n14` to
`della-h12n15` took 1.73 times as long (1.18-2.86, 586 runs), and runs on
`della-h12n15` in both campaigns 1.83 times as long (60 runs); from the Gold
node to `della-h12n14` the ratio is 0.77 (1,380 runs), the processor gap seen
before. So during this campaign `della-h12n15` was loaded enough to slow
memory-bound work by about 1.7x, as much as the processor gap the pinning
removed: same-model pinning does not make shared-node timings comparable.
Almost all the inexact rows that the solver comparison rests on (the fixed,
`sigma*`, and best configuration of every solver) ran on `della-h12n14`;
the exceptions are LASSO GD's best (sigma 0.9), side-100 GD's `sigma*`, and
side-100 Nystrom-PCG rank 32 at sigma 0.1, s = 1. Most near-exact and exact
baselines ran on `della-h12n15` (side-100 exact ADMM: 7.5 s on `della-h12n14`,
8.2-18.8 s on `della-h12n15` depending on when it ran). `plots.solver_comparison` lists the host of every
row; `results/studies_v2/solver_comparison.csv` has the table.

## 2026-11-08: complete

The rerun of task 35 (17100291_35) completed in 40:13 of 4:00 on
`della-h12n14`, finishing seeds 17-19 (about 800 s each, against about
1,900 s each on the loaded `della-h12n15`), and the collect of that study
(17100293_17) completed. `campaign status`: all 5,870 planned runs finished,
all 420 instances complete; no study has a `missing.csv` (the copy left by
the first collect was deleted locally, since `sync down` deletes nothing).
Every run converged, and every theorem check holds as above. The final
cross-check against `studies_v1`: 5,800 runs reproduce exactly, 40 formerly
censored runs converge, and 30 runs are the new `s*` strides. That one
configuration, the side-100 Nystrom-PCG rank-128 near-exact baseline, mixes
hosts: 17 instances on `della-h12n15` and 3 on `della-h12n14`.

Figures: `plots.studies` drew all 24 studies locally (10 figures for the 13
studies with contraction constants, 8 for AGD and Nystrom-PCG).
`plots.solver_comparison` wrote `results/studies_v2/solver_comparison.csv`.
Median cost per run (algorithm time plus the solver's own setup) of each
solver's best configuration, all timed on `della-h12n14` except LASSO GD:

| Problem | CG | GMRES | GD | AGD | Nystrom-PCG r32 | Nystrom-PCG r128 | exact ADMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QP side 40 | 0.7 s | 0.9 s | 0.7 s | 1.0 s | 2.1 s | 9.7 s | 0.3-0.4 s |
| QP side 64 | 4.5 s | 4.8 s | 5.8 s | 5.4 s | 11.2 s | 41.1 s | 1.3-4.4 s |
| QP side 100 | 17.2 s | 22.5 s | 25.7 s | 27.5 s | 43.7 s | 165.2 s | 7.5-18.8 s |
| LASSO | 3.9 s | 5.1 s | 7.2 s (on `h12n15`) | 6.6 s | | 23.8 s | 19.3-25.1 s |

Nystrom-PCG takes as many inner steps as CG in its best configurations (QP
side 100: 239-242 against 242), so its preconditioner does not cut
iterations on these subproblems, while each step costs more and the rank-128
sketch adds 76 s at side 100. The exact-ADMM range spans the studies and so
the two hosts.
