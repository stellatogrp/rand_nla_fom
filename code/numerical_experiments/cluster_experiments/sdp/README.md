# Algorithm 3 for semidefinite programs

`sdp/` implements the SDP application of `tex/General_inexactness.tex`
(the subsection after Algorithm 3, doubly inexact proximal ADMM) and the small
synthetic experiments that sanity check it. It is self-contained: it imports
no sealed release and changes nothing else in this project.

The problem is

```text
minimize <C, X>   subject to   E(X) = b,   X positive semidefinite,
```

split as `f(X) = <C, X>` and `g(y, Z)` the indicator of `y = b` and `Z`
positive semidefinite, with the coupling `(E(X), X) = (y, Z)`, penalty `c > 0`,
and proximal parameter `rho > 0`.

## The two inexact subproblems

**The `f`-subproblem** is the positive definite linear system

```text
(E*E + (1 + rho) I) X = E*(w_y) + w_Z + rho t - C / c,
```

solved approximately by conjugate gradients, warm-started at the previous `X`.
`f` is linear, so its subgradient certificate is `a = C` at every `X`: the pair
`(X, C)` is on the graph exactly, and `r_f` is the residual of that system.

**The `g`-subproblem** projects `(2 E(X) - w_y, 2 X - w_Z)`. The `y` part is
exact (`y = b`, with the subgradient `b_y = c (2 E(X) - w_y - b)`). The `Z`
part, the projection of `M = 2 X - w_Z` onto the cone, is approximate: block
Lanczos builds a Krylov subspace of `M` with basis `Q`, and the Ritz pairs
`(theta_i, v_i)` of `Q'MQ` give

```text
Z = sum over theta_i > 0 of theta_i v_i v_i'      (the approximate projection)
B = sum over theta_i < 0 of theta_i v_i v_i'      (b_Z = c B, the subgradient)
```

`Z` is positive semidefinite, `B` is negative semidefinite, and `Z B = 0`
because Ritz vectors are orthonormal, so `c B` is in the normal cone at `Z`
exactly, for every subspace. The inexactness is the residual of the Moreau
decomposition, `r_g = Z + B - M = QQ'MQQ' - M`.

With these, the quantities of Algorithm 3 are (`sdp/admm.py`, checked against
the paper's definitions in `tests/test_sdp.py`)

```text
Q^2 = ||E(X) - b||^2 + ||X - Z||^2 + ||q||^2 / rho,   q = r_f + r_g - rho (X - t),
E^2 = ||r_g||^2 + ||r_f + r_g||^2 / rho,
P^2 = ||E(X) - b||^2 + ||w_Z + B - X||^2 + rho ||X - t||^2.
```

A check stops if `Q <= delta` and accepts if `E <= sigma P`; otherwise it runs
conjugate gradients until `||r_f|| / sqrt(rho) <= f_share sigma P` and then
extends the Krylov subspace, `stride` steps of either between checks. When `M`
changes (new `X` or new outer state) the subspace is rebuilt from the retained
Ritz vectors, the most positive and most negative ones (a thick restart).

## What the subspace has to hold

`||r_g||_F` is at least the Frobenius norm of the eigenvalues of `M` that the
subspace leaves out, and the test needs `||r_g|| <= E <= sigma P_k` with
`P_k -> 0`. At a solution `M* = X* - S*/c` for the dual slack `S*`, and under
strict complementarity `rank X* + rank S* = n`, so `M*` has no zero eigenvalue
to leave out. The experiments record this as `rank_needed`, the smallest
subspace dimension for which any Rayleigh-Ritz pair can pass the test at the
accepted iterate (a lower bound computed from the eigenvalues of `M`, off the
clock), next to the dimension actually used.

## The one-sided projection

`sdp_v1` confirmed the lower bound: with the pair above, the subspace is the
whole space at almost every outer iteration (`../sdp_v1/LAUNCH_STATUS.md`).
`projection="one_sided"` keeps only the positive Ritz pairs
`(Theta_+, V_+)` explicit and takes the rest of `M` as the normal vector:

```text
Z = V_+ Theta_+ V_+'        B = (I - P_+) M (I - P_+),   P_+ = V_+ V_+'
```

`Z B = 0` holds exactly again, and the residual is
`Z + B - M = -(V_+ R_+' + R_+ V_+')` with the Ritz residuals
`R_+ = M V_+ - V_+ Theta_+`, of norm `sqrt(2) ||R_+||`: only the positive Ritz
pairs have to converge, about `rank X*` of them. What is no longer automatic
is `B <= 0`. It holds exactly when no positive direction of `M` is left
outside `V_+`, and a Krylov subspace cannot prove that. The safeguard has two
layers (`sdp/projection.py`):

- **The guard** (no extra product with `M`). The non-positive Ritz pairs of
  `M` are Ritz pairs of the deflated matrix `B`. The largest, `theta_0` with
  residual `rho_0`, estimates the top of its spectrum as
  `theta_0 + kappa rho_0` (`kappa = 4`, `--guard-factor`). A pair is accepted
  only if that estimate is not positive; otherwise the subspace grows,
  whatever the relative error test says. The restart keeps the positive Ritz
  vectors and `block_size` guard vectors, so an eigenvalue rising through zero
  is followed until it joins `V_+`, and the first decomposition of a run is
  exact, so that there is something to follow. This is an estimate, not a
  proof.
- **The certificate** (`cholesky=True`, the `certified_` methods). `B <= 0` on
  the complement of `V_+` if and only if `-B + alpha P_+` is positive
  definite, which one Cholesky factorization decides at `n^3 / 3` operations.
  A pair is accepted only after it passes; a pair that fails is replaced by
  one from a larger subspace. With the certificate every accepted pair is on
  the graph, as Algorithm 3 requires, up to rounding.

The off-the-clock diagnostics compute the largest eigenvalue of every
accepted `B` (`violation` in `histories.csv`, `violations` in `runs.csv`), so
the experiments report how often the guard alone would have been wrong. The
positive side is the explicit one: when the solution has high rank (for
example `rank S* << n`), the subspace is large again, and the mirrored
construction (explicit negative pairs, `Z` as the rest) would be the one to
use; it is not implemented.

This implementation forms `B` and the test quantities as dense matrices, so
every check costs a few `n^2` operations beyond the products with `M`.

## Running

```bash
uv run python -m sdp.run --allow-local --family planted --n 40 --m 80 --rank 4 \
  --seeds 0 1 2 --reference --results-dir results/sdp_local --name planted_n40
uv run python -m sdp.run --allow-local --family maxcut --n 40 --c 0.01 \
  --reference --results-dir results/sdp_local --name maxcut_n40
uv run python -m sdp.summarize --results-dir results/sdp_local
uv run python -m plots.sdp --results-dir results/sdp_local
```

`--help` lists the flags. `planted` draws a primal-dual solution first
(`--rank`, `--dual-rank`; the default dual rank `n - rank` is strict
complementarity) and builds the data around it, so the optimal value is known
without a solver; `maxcut` is the max-cut relaxation of a random graph.
`--reference` also solves each instance with CVXPY (Clarabel). Each study runs
exact proximal ADMM, conjugate gradients alone, block Lanczos alone, the
doubly inexact method with the two-sided pair and with the one-sided pair
(guard alone, and guard with certificate) for every `--sigmas` and
`--block-sizes`, and a near-exact baseline (`sdp/run.py` has the table;
`--groups` selects the methods).

Outputs, in `<results-dir>/<name>/`: `settings.json` (settings and a hash of
this package's sources), `runs.csv` (one row per run: inner work, the
certificate of the returned point, the theory checks), and `histories.csv`
(one row per outer iteration). `sdp.summarize` writes
`<results-dir>/summary.csv` and reports five checks per run: convergence, the
graph inclusions at the returned point, the bounds of Proposition 5.2, the
one-step decrease and update bound of Section 3 against the planted solution,
and, for the one-sided projection, that no accepted `B` has a positive
eigenvalue. `plots.sdp` draws the figures of each study.

On Della a campaign is a list of studies, one per line (`sdp_v1/studies.txt`,
`sdp_v2/studies.txt`), run as a Slurm array by `slurm/sdp_sanity.slurm`; see
`sdp_v1/README.md` and `sdp_v2/README.md`. `sdp_v1` ran the package before the
one-sided projection was added; that version is archived byte for byte in
`releases/sdp_r1`, and its methods are unchanged here
(`tests/test_sdp.py`).
