# sdp_v2: the one-sided, safeguarded projection

`sdp_v1` found that the two-sided pair of Algorithm 3 for SDPs (positive Ritz
pairs for `Z`, negative ones for `B`) needs the whole space at almost every
outer iteration. The user asked (2026-11-12) to try the one-sided variant with
a safeguard: `Z` from the positive Ritz pairs, `B` the rest of `M`, accepted
only when a guard estimate says that `B <= 0` and, in the certified method,
when a Cholesky factorization proves it (`../sdp/README.md`).

Questions, per study:

1. Is every accepted pair on the graph (`violations`, from the eigenvalues of
   every accepted `B`, off the clock)? How often does the guard alone err, and
   how often does the certificate reject a pair the guard let through?
2. How many products with `M` does an outer iteration need, against `n` for
   the two-sided pair, and how large is the subspace?
3. Do the outer iterations and the checks of `sdp_v1` (convergence,
   Proposition 5.2, Section 3) stay as they were?
4. What does the projection cost in wall-clock next to the eigendecomposition
   of exact proximal ADMM (`projection_seconds`, `certificate_seconds`)?

| Task | Study | Instance | Methods |
| --- | --- | --- | --- |
| 0 | `planted_n40` | planted, `n = 40`, `m = 80`, rank 4 | exact, two-sided, one-sided, certified |
| 1 | `planted_n80` | planted, `n = 80`, `m = 160`, rank 8 | same |
| 2 | `planted_n120` | planted, `n = 120`, `m = 240`, rank 12 | same |
| 3 | `planted_n80_dual8` | as task 1 with `rank S* = 8` | same |
| 4 | `maxcut_n40` | max-cut, `n = 40`, `c = 0.01` | same |
| 5 | `maxcut_n80` | max-cut, `n = 80`, `c = 0.01` | same |
| 6 | `planted_n80_kappa1` | as task 1 with guard factor 1 | exact, one-sided, certified |
| 7 | `maxcut_n80_kappa1` | as task 5 with guard factor 1 | exact, one-sided, certified |
| 8 | `planted_n200` | planted, `n = 200`, `m = 300`, rank 10, `c = 0.3` | exact, one-sided, certified |
| 9 | `maxcut_n200` | max-cut, `n = 200`, `c = 0.003` | exact, one-sided, certified |
| 10 | `maxcut_n400` | max-cut, `n = 400`, `c = 0.003` | exact, one-sided, certified |
| 11 | `maxcut_n800` | max-cut, `n = 800`, `c = 0.003` | exact, one-sided, certified |

Tasks 0 to 5 repeat instances of `sdp_v1` (seeds 0, 1, 2; `sigma = 0.1, 0.5,
0.9`; `delta = 1e-6`) with 4 and 8 guard vectors, and the two-sided pair with
block sizes 4 and 8 on the same host for comparison. Task 3 has a solution of
high rank, where the positive side is the wrong one to keep explicit. Tasks 6
and 7 set the guard's safety factor to 1 instead of 4, the plain residual
bound, to show what the factor and the certificate are for. Tasks 8 to 11 are
larger (`sigma = 0.5, 0.9`; tasks 10 and 11 with 4 guard vectors; task 11 with
seed 0 and `sigma = 0.5` only), without the two-sided pair, whose cost per
outer iteration is `n` products and a full Rayleigh-Ritz step. The penalties
of the larger studies come from local scans of exact proximal ADMM (`c = 0.3`
against 1 for `planted_n200`: 370 against 1,236 outer iterations; `c = 0.003`
against 0.01 for max-cut at `n = 400`: 2,092 against 5,446). The max-cut
studies at `n <= 80` also solve each instance with CVXPY (Clarabel).

Within a study every method runs in one task on one node, one after the
other, so the wall-clock comparison of a study is on one host; the tasks
avoid the two nodes on which `studies_v3` is timing its runs.

```bash
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync up --config slurm/sdp_v2.env --dry-run
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py submit slurm/sdp_sanity.slurm \
  --config slurm/sdp_v2.env --cpus 1 --mem 6G --time 05:00:00 --array 0-11%4 \
  --job-name rnla-sdp_v2 --sbatch-arg=--exclude=della-h12n14,della-h12n15 \
  --export STUDIES=sdp_v2/studies.txt --export RESULTS=results/sdp_v2
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync down --config slurm/sdp_v2_results.env
uv run python -m sdp.summarize --results-dir results/sdp_v2
uv run python -m plots.sdp --results-dir results/sdp_v2
```

Launch record: `LAUNCH_STATUS.md`.
