# studies_v1: the cluster problems as the original studies

The first campaign of `studies.campaign` (see `../studies/README.md`): the
cluster's three problem families with all of their inner solvers, run in the
format of the original `admm_qp` and `ppm_huber` studies, to the original
stopping rule (`||z_k|| <= 1e-9`; `10^6` inner steps or 300 s per inexact run;
50,000 outer iterations). Results go to `results/studies_v1/<study>/`.

Status: complete on 2026-11-07, all 6,014 runs; outcomes, theorem checks, and
the censored runs are in `LAUNCH_STATUS.md`.

| Lane | Studies | Instances | Run tasks | At once | Per task |
| --- | --- | ---: | ---: | ---: | --- |
| `qp` | `admm_qp`, sides 40, 64, 100 (64,000 to 1,000,000 variables) × GMRES, GD, AGD, CG, Nystrom-PCG ranks 32 and 128 | 20 | 261 | 8 | 16 GB, 9 h |
| `kaczmarz` | `admm_qp`, side 40 × block and cyclic Kaczmarz, blocks of 32 and 128 rows | 3 | 58 | 1 | 4 GB, 1 h |
| `regression` | `admm_lasso` (GMRES, GD, AGD, CG, Nystrom-PCG rank 128) and `ppm_logistic` (GD) on the 100,000 × 50,000 `random_normal` design | 10 | 90 | 2 | 4 GB, 3 h |

The caps are those of `../AGENTS.md`. Every run task runs one configuration of
one study for every instance, storing each run as it finishes, so a
configuration that stalls holds one slot until its time limit and delays no
other configuration, solver, or lane. One task per run would be 6,014 tasks,
beyond Della's per-user limit of 1,000 queued jobs; this layout is 465 tasks in
all, calibrate and collect included. Each lane's calibrate task for a study
times `c_in` and `c_test`, fixes `configurations.csv`, and sets up every
instance (Lanczos `L`, the ground truth, the paper's parameters), which the
run tasks then load.

Kaczmarz uses strides that are fractions 0.01, 0.05, 0.1, 0.5, and 1 of one
pass over its blocks (the cluster's earlier screen), with the sigma sweep at
0.1 pass, and caps the near-exact baseline at 600 s
(`--near-exact-time-limit`). On a workstation every block-Kaczmarz run tried
(block 32, side 40) stopped at the `10^6` inner steps in about 2 minutes
without converging, as in the earlier screen, so this lane records censored
runs. On Della every inexact and near-exact Kaczmarz run stopped at its time
limit; only exact ADMM converged. The regression lane reads the campaign's copy of `datasets/random.json`
(`random.json`), so it does not depend on Della's `datasets/`.

## Inputs and seal

`campaign.json` is the specification; `inputs/` and `seal.json` are generated
from it:

```bash
uv run python -m studies.campaign prepare studies_v1/campaign.json  # --force before a launch only
```

Every task checks `seal.json` (the specification, every input, the `studies`
sources, the Slurm script, and the sealed engine release) before it runs.
After the launch nothing sealed may change; a changed campaign is
`studies_v2`. This README is not sealed.

## Submitting

Through the slurm skill, from `cluster_experiments/`, after
`sync up --config slurm/studies_v1.env --dry-run` and `sync up`. Print the
commands, lane by lane:

```bash
uv run python -m studies.campaign commands studies_v1 \
  --pilot qp admm_qp_side100_nystrom_pcg_rank128 \
  --pilot kaczmarz admm_qp_side40_block_kaczmarz_block32 \
  --pilot regression ppm_logistic_random_normal_gd
```

For each lane, in order:

1. the calibrate array;
2. the pilot, `afterok` on the calibrate array: the near-exact configuration
   of the lane's most expensive study, for every instance;
3. after inspecting the pilot (`stats`, its runs under
   `results/studies_v1/<study>/parts/`), the run array, `afterany` on the
   calibrate array, with `--time` and `--mem` sized from the pilot (the pilot's
   task is skipped there, as it is done);
4. the collect array, `afterany` on the run array.

Each `submit` validates with `sbatch --test-only` and submits only with
`--yes`. Jobs of at most an hour appear to run in Della's `test` QoS, which
allows two running jobs per user; the QP lane needs its measured time limit
above an hour to run eight at once.

## Following and collecting

```bash
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync down --config slurm/studies_v1_results.env
uv run python -m studies.campaign status studies_v1
uv run python -m studies.run_admm_qp @studies_v1/inputs/admm_qp_side40_gmres.args \
  --stage collect --allow-partial                       # tables from what finished
uv run python -m plots.studies admm_qp_side40_gmres --results-dir results/studies_v1
```

`status` counts finished runs per study; `collect --allow-partial` writes the
tables from the finished runs and lists the rest in `missing.csv`. A missing
run is rerun by resubmitting its task index; finished runs are skipped.
