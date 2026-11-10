# bench_v1 launch status

## 2026-11-08: pilot submitted

Requested by the user after `studies_v2`: the one-node benchmark proposed to
compare the solvers' wall-clock times without the node and load differences
of per-task timing.

Design changes from the proposal, before launch: the plan times every
inexact configuration of every solver (not two chosen ones) on the first 5
instances of each problem (not all), because correcting `studies_v2`'s times
for the loaded node changed the two fastest configurations of 13 of 24
studies, and even the four fastest by the raw times missed a corrected top-two
configuration in 10; the estimated length (about 14 h) is the same either way.

Before submission: Ruff and all 398 tests passed (6 new in
`tests/test_bench.py`, among them an end-to-end run on a 64-variable QP
campaign that checks the iteration counts against the campaign's, the
resumption, and the seal); a local smoke test of the first side-40 instance
(74 runs, 44 s on a laptop) reproduced `studies_v2`'s outer and inner
iteration counts in all 74 runs; holding the six side-100 setups at once
peaked at 8.1 GB locally (6.5 GB of it for the rank-128 sketch), so the
side-100 job asks for 24 GB. `sync up --dry-run` listed the 12 new or edited
files (the `bench` package, this campaign's plan, seal, and notes, the Slurm
script, two sync configurations, the `studies_v2` notes, and two tests) and
no sealed file.

| Job | What | Node | Memory, time |
| --- | --- | --- | --- |
| 17114017 | pilot: first instance of side 40 and of LASSO, near-exact baselines included, into `results/bench_v1` (the full runs skip them) | `della-h12n14` | 16 GB, 1:00 |

Next: inspect the pilot (`stats`, iteration counts, the exact-ADMM probe),
then submit one job per problem, chained `afterany` so that only one runs at a
time, all with `--nodelist=della-h12n14`.

## 2026-11-09: pilot results and the full runs

The pilot waited about 14 hours for `della-h12n14` (pending `Priority`, then
`Resources`), ran from 04:37, and was stopped by the Slurm warning signal at
57:40 of its 1:00 (exit 138); 849 MB reported, 99% CPU efficiency. It timed
the whole first side-40 instance (80 runs in 9.8 minutes) and 48 of the 67
runs of the first LASSO instance; the next jobs resume from there. Every run
it finished reproduced the campaign's outer and inner iteration counts.

Two things the pilot measured:

- Each LASSO run spends about 31 s outside its timed algorithm (0.17 s per
  outer iteration: the diagnostics, chiefly the exact-residual solve, which
  the paused clock leaves out of `runtime_seconds`, as in the campaigns); at
  side 40 it is under a second. The timings are unaffected, but LASSO takes
  about three times the wall-clock estimated.
- `della-h12n14` itself was loaded: the pilot's side-40 runs took a median
  1.60 times as long as the same runs on that node in `studies_v2`
  (10th-90th percentile 1.08-2.12), the LASSO runs 1.19 times. The
  interleaving puts every solver under the same fluctuating load, so the
  comparison stays fair, but it is noisier and the absolute times are high.

Remaining work at the pilot's pace: side 40 about 0.7 h, LASSO 4 h, side 64
4 h, side 100 15 h. To wait for the node twice rather than four times, the
rest runs as two jobs, the second `afterany` on the first, so only one runs
at a time; `sbatch --test-only` placed both at about 2026-11-13 21:29, and
estimates for this node have been pessimistic.

| Job | What | Node | Memory, time | Dependency |
| --- | --- | --- | --- | --- |
| 17183540 | side 40, LASSO, side 64 (all instances not yet timed) | `della-h12n14` | 16 GB, 14:00 | |
| 17183542 | side 100 | `della-h12n14` | 24 GB, 22:00 | `afterany:17183540` |

## 2026-11-10: held for Della's monthly maintenance

From 18:44 on 2026-11-09 job 17183540 was pending `ReqNodeNotAvail`, and on
2026-11-10 the login node could not reach `/home` or `/scratch/gpfs`
(`sbatch --test-only` failed to `cd` into the project). Della's routine
maintenance is the second Tuesday of each month, about 6 AM to 2 PM, and
covers `/scratch/gpfs`; the scheduler holds any job whose time limit would
run into it, with that pending reason
(https://researchcomputing.princeton.edu/faq/when-do-systems-undergo-s).
2026-11-10 is that Tuesday. Both jobs stay queued, unchanged; they become
eligible when the maintenance ends.
