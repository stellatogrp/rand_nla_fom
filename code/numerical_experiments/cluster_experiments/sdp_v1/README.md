# sdp_v1: sanity check of Algorithm 3 on small synthetic SDPs

The user asked (2026-11-11) for the SDP version of Algorithm 3 (doubly inexact
proximal ADMM: an approximate solve of the quadratic `f`-subproblem, then an
approximate projection onto the cone by block Lanczos whose negative Ritz
pairs give the subgradient) and for small synthetic experiments on Della to
sanity check it. The method and its code are in `../sdp/README.md`; this
campaign is the list of studies in `studies.txt`, one Slurm array task each.

| Task | Study | Instance | Purpose |
| --- | --- | --- | --- |
| 0 | `planted_n40` | planted, `n = 40`, `m = 80`, rank 4 | smallest case |
| 1 | `planted_n80` | planted, `n = 80`, `m = 160`, rank 8 | reference case |
| 2 | `planted_n120` | planted, `n = 120`, `m = 240`, rank 12 | largest case |
| 3 | `planted_n80_rho0.1` | as task 1 with `rho = 0.1` | proximal parameter |
| 4 | `planted_n80_stride4` | as task 1 with stride `s = 4` | stride |
| 5 | `planted_n80_dual8` | as task 1 with `rank S* = 8` | no strict complementarity |
| 6 | `maxcut_n40` | max-cut relaxation, `n = 40`, `c = 0.01` | no planted solution |
| 7 | `maxcut_n80` | max-cut relaxation, `n = 80`, `c = 0.01` | no planted solution |

Every study runs three instances (seeds 0, 1, 2) with `delta = 1e-6`, and on
each instance 14 methods: exact proximal ADMM; conjugate gradients alone,
block Lanczos alone, and both (block sizes 4 and 16) at
`sigma = 0.1, 0.5, 0.9`; and the near-exact baseline (`E <= 1e-10`). The
planted studies use `c = 1` and `rho = 1` unless the table says otherwise;
`c = 0.01` for max-cut was the best of a local scan of exact proximal ADMM
over `c` in `{0.003, ..., 10}`. `--reference` also solves every instance with
CVXPY (Clarabel).

The quantities checked per run are the four of `sdp.summarize`: convergence,
the graph inclusions at the returned point, the bounds of Proposition 5.2,
and the one-step decrease and update bound of Section 3. Wall-clock times are
recorded with their host but are not the object here: the tasks avoid the two
nodes on which `studies_v3` is timing its runs.

```bash
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync up --config slurm/sdp_v1.env --dry-run
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py submit slurm/sdp_sanity.slurm \
  --config slurm/sdp_v1.env --cpus 1 --mem 4G --time 01:00:00 --array 0-7%2 \
  --job-name rnla-sdp_v1 --sbatch-arg=--exclude=della-h12n14,della-h12n15 \
  --export STUDIES=sdp_v1/studies.txt --export RESULTS=results/sdp_v1
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync down --config slurm/sdp_v1_results.env
uv run python -m sdp.summarize --results-dir results/sdp_v1
uv run python -m plots.sdp --results-dir results/sdp_v1
```

`slurm/sdp_v1.env` uploads only `sdp/`, `sdp_v1/`, and `slurm/sdp_*`; the code
of the launched campaigns on Della is not touched. Launch record:
`LAUNCH_STATUS.md`.
