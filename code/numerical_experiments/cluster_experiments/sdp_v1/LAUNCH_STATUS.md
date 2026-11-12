# sdp_v1 launch status

## 2026-11-11: submitted

Requested by the user: implement the SDP version of Algorithm 3 and run small
synthetic experiments on Della to sanity check it (`README.md` has the
design). The user chose a pilot of task 0 followed by the remaining tasks.

Before submission: Ruff and all 425 tests passed locally (16 of them in
`tests/test_sdp.py`); the eight studies ran locally to completion;
`sync up --config slurm/sdp_v1.env --dry-run` listed only the new files
(`sdp/`, `sdp_v1/`, `slurm/sdp_sanity.slurm`, `slurm/sdp_v1*.env`) and the
upload transferred those 12 files. `studies_v3` was running (8 tasks on
`della-h12n14` and `della-h12n15`); these jobs exclude both nodes so that its
timings are not disturbed, and they carry no `icelake` constraint because
their wall-clock times are not the object.

| Job | What | Array | Memory, time | Placement |
| --- | --- | --- | --- | --- |
| 17358366 | pilot: `planted_n40` | `0` | 4 GB, 0:30 | `--exclude=della-h12n14,della-h12n15` |
| 17358427 | the other seven studies | `1-7%2` | 4 GB, 1:00 | `--exclude=della-h12n14,della-h12n15` |

The pilot completed in 37 s on `della-r3c4n14` (exit 0, 156 MB reported, empty
stderr, all 42 runs converged), so the remaining tasks were submitted
unchanged with a one-hour limit.

## 2026-11-11: complete

All eight tasks completed with exit 0 and empty stderr, on `della-r3c4n14`
and `della-r3c4n9` (Xeon Gold 6242) and `della-i13n25` (Xeon Platinum
8480CL). Wall-clock per task: 0:37 (pilot), 6:13, 43:28 (`planted_n120`),
3:28, 4:22, 1:23, 0:22, 1:33. Reported memory was at most 754 MB except for
`planted_n120` (3.1 GB of the 4 GB requested, the Clarabel reference solve at
`n = 120`; a larger instance needs more memory or no `--reference`). The
sources on Della are the local ones (`source_sha256` in every
`settings.json`). Results: `results/sdp_v1/` (`sync down --config
slurm/sdp_v1_results.env`), table `summary.csv`, figures
`<study>/sanity.{png,pdf}`.

**Checks (`sdp.summarize`): all four pass on all 336 runs** (8 studies, 3
instances, 14 methods).

- Convergence: every run stopped by `Q <= delta = 1e-6`. The doubly inexact
  method takes between 2.6% fewer and 6.1% more outer iterations than exact
  proximal ADMM on the same instance.
- Graph inclusions at the returned point: smallest eigenvalue of `Z` at least
  `-1.0e-14`, largest eigenvalue of `B` at most `1.4e-14`,
  `||Z B|| <= 6.0e-15 ||Z|| ||B||`.
- Proposition 5.2: the coupling residual is at most `0.95 delta` and the
  stationarity residual at most 0.55 of its bound
  `c delta (||E|| + sqrt(rho))`.
- Section 3: the one-step decrease against the planted solution and the
  update bound `(1 - sigma) P_k <= Q_k <= (1 + sigma) P_k` hold at every
  accepted update with no violation.
- Against CVXPY (Clarabel, `optimal` on every instance): the objective at the
  returned `Z` differs by at most `1.4e-7` on the planted studies and
  `1.2e-5` on max-cut; the duality gap of the returned pair is at most
  `9.0e-6` in absolute value. In `planted_n80_dual8` (no strict
  complementarity) the method converges to another optimal solution, of rank
  72, with the same optimal value.

**Inner work.**

- Conjugate gradients: on the planted studies the relative test needs 8 to
  13 steps per outer iteration, and the near-exact baseline 3.4 to 6 times as
  many in total. On max-cut the system has two distinct eigenvalues and
  one or two steps solve it.
- Projection: the relative test leaves the Krylov subspace almost no room
  below the whole space. In all 216 runs that combine the Lanczos projection
  with a relative test (every `sigma`, block size, and stride), the median
  over the outer iterations of `rank_needed`, the lower bound on the dimension
  of any subspace whose Rayleigh-Ritz pair can pass, is `n`, the median
  dimension used is `n`, and the method spends between 0.98 `n` and 1.02 `n`
  products with `M` per outer iteration. The bound first reaches `n` after 6
  to 21 outer iterations (102 in one max-cut run at `sigma = 0.9`). From then
  on it is `n` at every stored iteration of the planted studies; on max-cut
  it is `n` at 87% (`n = 40`) and 78% (`n = 80`) of the stored iterations and
  between `n - 1` and `n - 12` at the others, mostly `n - 1`. The reason is in
  `../sdp/README.md`: the residual of the projection is at least the norm of
  the eigenvalues of `M = 2 X - w_Z` left out of the subspace, the test
  compares it with `sigma P_k -> 0`, and `M` has no small eigenvalues to
  leave out (`M* = X* - S*/c` with `rank X* + rank S* = n` under strict
  complementarity). The number of positive Ritz values settles at the rank of
  the solution (4, 8, and 12 on the strictly complementary planted studies),
  and all the other Ritz pairs are negative ones that only serve the
  subgradient `B`.

Iteration counts differ by a few steps from a run of the same studies on a
laptop (another BLAS), for the runs whose tests are close to their thresholds.

## 2026-11-12: the `sdp` package changed after this campaign

The one-sided projection, its guard, and the Cholesky certificate were added
for `sdp_v2`. This campaign ran the package as archived byte for byte in
`../releases/sdp_r1` (source hash
`28af313fa57510c32db68391d6f477c4d1c6ab6d07e4f8ded74c44654dd6b695`, the
`source_sha256` of every `settings.json` here). Its methods are unchanged in
the working package and produce the same iterates (`tests/test_sdp.py`); the
tables of `sdp_v2` have additional columns, and `sdp.summarize` and
`plots.sdp` still read these.
