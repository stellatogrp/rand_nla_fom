# studies_v3: Kaczmarz with block sizes around sqrt(n)

`studies_v1` ran randomized and cyclic block Kaczmarz at side 40 with blocks
of 32 and 128 rows and the original 300 s limit, and every inexact run
stopped at the limit. The user asked to retry them with block sizes around
`sqrt(n)`: here `n = 40^3 = 64,000` (the rows of the ADMM subproblem
`(Q + cI) x = r`), `sqrt(n) = 253`, and the block sizes are the divisors of
`n` nearest to 1/4, 1/2, 1, 2, and 4 times it: 64, 125, 250, 500, and 1000
rows (1000 to 64 blocks per pass). Both variants (`block_kaczmarz`, a block
drawn at random each step, and `cyclic_kaczmarz`, the blocks in turn) run at
every size: 10 studies, 3 instances each, in the original study format.

Choices, with the user's answers of 2026-11-11:

- Per inexact run a 3,600 s wall-clock stop (user's choice) instead of 300 s;
  the near-exact baseline keeps `studies_v1`'s 600 s cap
  (`--near-exact-time-limit`): it needs hours at any block size.
- `--max-total-inner 10000000` instead of `10^6`. The cap counts block steps,
  so `10^6` would stop the smallest blocks (about 1,000 passes at 64 rows)
  before the time limit, and the block sizes would not end under the same rule.
- Strides are the same fractions of one pass as in `studies_v1` (0.01, 0.05,
  0.1, 0.5, 1), the sigma sweep at 0.1 pass; block Kaczmarz adds `sigma*` and
  the `s*` of Lemma 4.2 and Remark 4.4 (592 and 833 steps at 250 rows).
- Up to 8 Kaczmarz tasks at once (user-approved for this campaign; the
  standing cap in `../AGENTS.md` is one). The calibrate tasks, which time one
  block step of each study (`c_in`), run one at a time on `della-h12n14`, so
  that the per-step costs of the block sizes are measured alike; the runs use
  both Platinum nodes (`--exclude=della-h17n4`), and every timing records its
  host.

A local measurement on one side-40 subproblem (`(Q + cI) x = r` from zero,
2026-11-11, a laptop) found the residual factor per pass over the rows nearly
independent of the block size (randomized 0.966, cyclic 0.931, for 64 to 1000
rows): the engine's blocks are drawn from a random row permutation, and rows
of the 7-point stencil rarely overlap. The time per pass is smallest near
`sqrt(n)` (0.069 s at 250 rows, 0.085-0.086 s at 125 and 500, 0.12 s at 64,
0.15 s at 1000, 0.20 s at 32), so the block size should mainly change the cost
per pass, not the number of passes.

```bash
uv run python -m studies.campaign prepare studies_v3/campaign.json  # done; --force before a launch only
uv run python -m studies.campaign commands studies_v3
python3 ~/.claude/skills/slurm/scripts/slurm_agent.py sync down --config slurm/studies_v3_results.env
uv run python -m studies.campaign status studies_v3
```

Launch record: `LAUNCH_STATUS.md`.
