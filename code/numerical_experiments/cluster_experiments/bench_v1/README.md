# bench_v1: one-node timing of every inner solver

`studies_v2` ran each configuration as its own Slurm task on shared nodes, so
its wall-clock times mix hosts and loads: identical runs took a median 1.73
times as long on `della-h12n15` as on `della-h12n14` during that campaign
(`../studies_v2/LAUNCH_STATUS.md`). This benchmark re-times the solvers in one
process on one node, `della-h12n14`, with the runs of each instance
interleaved in a seeded random order, so that a change of load falls on every
solver alike. Its code is `bench/one_node.py`; it reads `studies_v2`'s stored
setups (read only) and writes `results/bench_v1/`.

What is timed (`plan.json`, sealed by `seal.json` with the `bench` and
`studies` sources, the Slurm script, and `studies_v2`'s seal):

- the four ADMM problems of `studies_v2`: the QP at sides 40, 64, and 100 and
  the LASSO on the `random_normal` design, each on its first 5 instances;
- for each inner solver (GMRES, GD, AGD, CG, Nystrom-PCG ranks 32 and 128;
  rank 128 only on the LASSO), every inexact configuration of `studies_v2`
  (both sweeps), labelled `fixed` (`sigma = 0.1`, `s = 1`), `sigma_star`, and
  `campaign_best`. All of them, because `studies_v2`'s node-dependent timings
  reorder the near-ties among the fastest configurations (correcting for the
  loaded node changed the top two of 13 of the 24 studies);
- the near-exact baseline of every solver on the first 3 instances at side
  100 and the first 5 elsewhere, and exact ADMM once per instance (it also
  serves as a probe of the node's speed over the job).

That is 1,503 runs, estimated at about 14 hours from `studies_v2`'s times on
`della-h12n14` (side 40 about 0.7 h, LASSO 2.2 h, side 64 2.5 h, side 100
8.9 h). The cost of a run is its algorithm time plus the solver's own setup
(the Nystrom sketch; zero for GMRES), both timed in the benchmark; the
projector and Lanczos setup, the same for every solver, is not timed again.
Every run repeats a `studies_v2` computation, so `summarize` checks that its
outer and inner iteration counts are the campaign's.

## Commands

From `cluster_experiments/`:

```bash
uv run python -m bench.one_node plan --campaign studies_v2 --benchmark bench_v1 \
  --output bench_v1/plan.json --instances 5 --near-exact qp_side100=3 --near-exact-default 5
uv run python -m bench.one_node seal bench_v1          # --force before a launch only
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync up --config slurm/bench_v1.env --dry-run
# one job per problem, chained so that only one runs at a time, all on one node:
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py submit slurm/bench_one_node.slurm \
  --config slurm/bench_v1.env --cpus 1 --constraint icelake --mem 16G --time 02:00:00 \
  --sbatch-arg=--nodelist=della-h12n14 --job-name rnla-bench_v1-qp_side40 \
  --export PLAN=bench_v1/plan.json --export RESULTS=results/bench_v1 --export PROBLEMS=qp_side40
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync down --config slurm/bench_v1_results.env
uv run python -m bench.one_node summarize --plan bench_v1/plan.json --results-dir results/bench_v1
```

A job cut short loses only the run in progress: resubmitting it skips the
runs in `measurements.csv`. Launch record: `LAUNCH_STATUS.md`.
