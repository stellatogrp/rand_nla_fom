# studies_v3 launch status

## 2026-11-11: submitted

Requested by the user: retry the block Kaczmarz experiments with block sizes
around `sqrt(n)` (`sqrt(n)`, half and twice it, and a few others). The user
chose both variants (randomized and cyclic), up to 8 tasks at once, and a
3,600 s limit per inexact run (`README.md` has the design).

Before submission: `prepare` sealed 10 studies (16 configurations reserved per
randomized study, 13 per cyclic: 145 run tasks); a local calibrate stage of
the two 250-row studies produced the planned configuration lists (randomized:
`sigma* = 0.0702`, `s* = 592` and `833` steps); Ruff and all 400 tests passed
(`tests/test_campaign.py` checks the seal and that every study is at 1/4 to
4 `sqrt(n)` rows with the 3,600 s limit, the `10^7` step cap, and strides that
are fractions of a pass); `sync up --dry-run` listed 21 new or edited files
and no sealed one. No other job of the project was queued.

| Job | What | Array | Memory, time | Placement | Dependency |
| --- | --- | --- | --- | --- | --- |
| 17343875 | calibrate and set up 3 instances (10 studies) | `0-9%1` | 4 GB, 0:30 | `--nodelist=della-h12n14` | |
| 17343877 | runs | `0-144%8` | 4 GB, 4:00 | `--exclude=della-h17n4` | `afterany:17343875` |
| 17343889 | collect | `0-9%8` | 4 GB, 0:30 | any `icelake` node | `afterany:17343877` |

A run task is one configuration on 3 instances, at most 3 x 3,600 s; the
near-exact tasks stop at 3 x 600 s. Estimated 150-180 core-hours, about a day
at 8 tasks at once.

## 2026-11-11: the `studies` package changed locally after this launch

At the user's request the calibration behind `s*` was made cheaper for future
studies (`studies/inner.py` `calibrate`, `studies/study.py`): solvers whose
step costs the same wherever it falls (all but GMRES) time every step instead
of one step after nine untimed ones, the default is 30 repetitions instead of
300, and the study calibrates on its first instance's own setup instead of
building one for the calibration. This campaign was sealed with, and runs on
Della with, the previous code, archived byte for byte in
`releases/studies_r1` (equal to the `studies/*.py` entries of this seal; see
`tests/test_campaign.py`). Until this campaign's jobs have finished, the
changed `studies/` must not reach Della: the sync configurations of the
launched campaigns now exclude `/studies`, and its collect and any rerun use
the code on Della. Verify it locally with
`studies.campaign.verify(manifest, release="studies_r1")`.
