# studies_v2: studies_v1 with a 1,200 s limit, without Kaczmarz

`studies_v1` stopped every inexact run at 300 s, as the original studies do.
That censored 40 Nystrom-PCG runs (side 100, rank 128, strides s = 10 and
s = 20, all 20 instances): in the median, s = 10 stopped 11 and s = 20
stopped 73 outer iterations short of the 141 that the converged runs needed. This campaign reruns every non-Kaczmarz study of `studies_v1`
with `--time-limit 1200`, so that the Nystrom-PCG runs finish and every inner
solver is compared under the same, longer limit. Nothing else changes: the
studies, instances (seeds), grids, stopping rule (`||z_k|| <= 1e-9`, `10^6`
inner steps, 50,000 outer iterations), code, and engine are those of
`studies_v1` (`tests/test_campaign.py` checks that each study's settings
differ from its `studies_v1` counterpart in `time_limit` only). The Kaczmarz
lane is left out: in `studies_v1` every inexact Kaczmarz run ended between
`||z|| = 1.2e-5` and `0.87` at 300 s, far from `1e-9`.

Results go to `results/studies_v2/<study>/`; `studies_v1` and its results are
untouched. Status: complete on 2026-11-08, all 5,870 runs converged; the
record, the timing caveats, and the solver comparison are in `LAUNCH_STATUS.md`.

| Lane | Studies | Instances | Run tasks | At once | Per task |
| --- | --- | ---: | ---: | ---: | --- |
| `qp` | `admm_qp`, sides 40, 64, 100 × GMRES, GD, AGD, CG, Nystrom-PCG ranks 32 and 128 | 20 | 261 | 8 | 16 GB, 9 h (calibrate 1.5 h) |
| `regression` | `admm_lasso` (GMRES, GD, AGD, CG, Nystrom-PCG rank 128) and `ppm_logistic` (GD) on the `random_normal` design | 10 | 90 | 2 | 4 GB, 2 h (calibrate 0.5 h) |

Sizing comes from `studies_v1` (`../studies_v1/LAUNCH_STATUS.md`), which ran
the same tasks with the same code, so there are no pilots. The longest task,
the near-exact Nystrom-PCG rank-128 baseline at side 100 (no time limit, as in
the original), took about 1,130 s per instance: about 6.3 h for 20. A censored
configuration needs about 141 outer iterations of s inner steps at about
0.23 s each, so s = 20 needs about 650 s per run (3.6 h per task), within the
1,200 s limit. Each calibrate task also sets up all of its instances; at
side 100 with Nystrom-PCG rank 128 that took about 53 minutes for 20 in
`studies_v1`.

The calibrate tasks time `c_in` and `c_test` afresh. `sigma*` (Proposition
5.1) does not depend on them, so the sigma sweep is that of `studies_v1`; `s*`
(Lemma 4.2, Remark 4.4) does, so an `s*` stride off the grid may differ from
`studies_v1`'s.

## Commands

From `cluster_experiments/`, through the slurm skill:

```bash
uv run python -m studies.campaign prepare studies_v2/campaign.json  # done; --force before a launch only
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync up --config slurm/studies_v2.env --dry-run
uv run python -m studies.campaign commands studies_v2               # the submissions (skip the pilots)
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync down --config slurm/studies_v2_results.env
uv run python -m studies.campaign status studies_v2
uv run python -m plots.studies admm_qp_side100_nystrom_pcg_rank128 --results-dir results/studies_v2
```

Launch record: `LAUNCH_STATUS.md`.
