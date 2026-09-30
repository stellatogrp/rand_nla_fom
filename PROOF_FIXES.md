# Proof checks and revisions for Pranav

Reviewed on September 30, 2026, against commit `f03c9d6`.
Sources: [main draft](tex/Draft.tex),
[numerical section](tex/numerical_experiments_draft.tex), and
[implementation](code/src/drs.py).
The LaTeX labels below identify passages even if theorem numbering changes.

Pranav, please address the inner-work arguments first. The admissible-pair
monotonicity argument, one-step decrease, and residual summability appear sound.
The main gaps concern the probabilistic statements, the warm start actually
used by the Schur solvers, and the scope of the complexity claims.

This is a checklist of what to check or change. Suggested repair routes below
still need to be written out and checked in the paper.

## 1. Condition the randomized analysis on the outer history

**Priority: correctness.** Locations: `thm:stopping-time`,
`eq:inner-contraction`, and `thm:total-inner-work`.

- [ ] Introduce a filtration containing all randomness before inner solve
  $k$. The current outer iterate, warm start, $\mathcal E_{k,0}$, and
  $\bar R_k$ are measurable with respect to this history.
- [ ] State the inner contraction conditionally on that history. Allow a
  prefactor, for example

  $$
  \mathbb E[\|\varepsilon^{(k)}_{(l)}\|_M^2\mid\mathcal F_k]
  \le C_{\rm in}a^l\mathcal E_{k,0}^2,
  \qquad C_{\rm in}\ge1,\quad 0<a<1.
  $$

  Specify whether these constants are uniform over outer iterations and
  histories. Verify this for each solver to which the theorem is applied.
- [ ] Apply conditional Markov inequality and the tail-sum formula to obtain
  bounds on $\Pr(N_k>l\mid\mathcal F_k)$ and
  $\mathbb E[N_k\mid\mathcal F_k]$. Include $C_{\rm in}$ in $\beta_k$.
  Define the underlying inner sequence beyond its first accepted iterate so
  that the tail argument is unambiguous.
- [ ] Take expectations of the random terms on the right-hand side of the
  total-work bound. As written, an unconditional expected work is bounded by
  an expression involving the random $\beta_0$ and
  $\log(\bar R_0/\bar R_{n-1})$. The pathwise telescoping calculation can be
  retained, followed by the tower property.
- [ ] State the positivity and integrability conditions needed for the bound
  to be finite. Handle exact termination separately rather than taking the
  logarithm of zero. Do not condition on the entire future outer trajectory;
  that trajectory depends on the inner random draws being analyzed.

## 2. Qualify the uniform per-iteration work claim

**Priority: unsupported central claim.** Locations: abstract, contribution 3,
`rem:per-iteration`, and conclusion.

- [ ] Make the assumption
  $\sup_k\bar R_{k-1}/\bar R_k\le\Delta<\infty$ visible wherever the
  iteration-independent expected-work bound is claimed. The remark assumes
  it, but the abstract, contribution list, and conclusion omit it.
- [ ] Either prove a suitable ratio bound or state the result conditionally.
  An upper linear-convergence bound does not supply a lower bound on the next
  residual. Likewise, the remaining logarithmic progress term in the
  total-work theorem is not automatically $O(n)$.
- [ ] Consider adding a corollary for the stricter regime
  $0<\theta<1-\sigma$. The existing Lipschitz and accepted-step estimates give

  $$
  \bar R_k\ge
  \left(1-\frac{\theta}{1-\sigma}\right)\bar R_{k-1}.
  $$

  This supplies a ratio bound in that regime. It does not cover the
  $\theta=1$ setting used in many experiments.
- [ ] If the intended result is total work to a prescribed accuracy, formulate
  that stopping time explicitly. An upper bound on the terminal residual
  does not upper-bound its negative logarithm. Check the final inner solve
  separately when passing from a fixed horizon to a tolerance-based horizon.

Exact-arithmetic dimension bounds for CG and unrestarted GMRES are a separate
way to bound their iteration counts. They do not establish the claimed
spectrum-based bound for a general geometrically convergent inner solver.

## 3. Match the warm-start lemma to the Schur implementation

**Priority: theory/implementation mismatch.** Location:
`lem:warm-start-drift`; compare the `assemble` and `assess_multiplier`
functions in [drs.py](code/src/drs.py).

The lemma retains the entire previous pair $(\xi,\nu)$. That gives the stated
new residual $\varepsilon-\theta s$. The Schur implementation instead retains
$\nu$ and reconstructs $\xi=x_{\rm new}-\gamma_xA^\top\nu$.

- [ ] State separate warm-start bounds for the coupled and Schur solvers.
  For a Schur iterate whose first residual block is zero, check the identity

  $$
  \varepsilon_{\lambda,\rm new}
  =\varepsilon_{\lambda,\rm old}
  -\theta(s_\lambda+\gamma_\lambda A s_x).
  $$

- [ ] Derive and propagate the resulting constant through $\beta_k$, $\rho$,
  and the total-work bound. A bound to verify is

  $$
  \mathcal E_{k,0}\le
  \left(\sigma+\theta\sqrt{1+\gamma_x\gamma_\lambda\|A\|_2^2}\right)
  \|s^{(k)}\|_M.
  $$

**Concrete check:** take $A=[10]$, $b=0$, $f(x)=x^2/2$,
$\gamma_x=\gamma_\lambda=\theta=1$, and $\sigma=0.2$.
At $z=(10,1)$ the exact inner pair is $(0,1)$ and $s=(-5,0)$.
The next outer point is $(5,1)$. Reusing the multiplier and reconstructing
$\xi$ gives $(-5,1)$, whose residual is $(0,50)$.
Its norm exceeds the original claimed bound $(\sigma+\theta)\|s\|=6$.
This does not invalidate the lemma for the full-pair warm start it states.

## 4. Justify the contraction assumption for each inner solver

**Priority: correctness.** Locations: the paragraph after
`thm:stopping-time`, `rem:per-iteration`, and
[the inner solvers](code/src/_linear_solvers.py).

- [ ] **CG:** remove the blanket assertion that the displayed residual bound
  holds with prefactor one. Euclidean residuals need not decrease at every
  CG iteration. Translate the standard energy-norm estimate into the norm
  used by the acceptance test and retain the resulting prefactor.
  One available estimate for SPD $S$, with $\kappa_2(S)>1$, is

  $$
  \|r_l\|_2^2\le
  4\kappa_2(S)
  \left(\frac{\sqrt{\kappa_2(S)}-1}
  {\sqrt{\kappa_2(S)}+1}\right)^{2l}\|r_0\|_2^2.
  $$

  Check its conversion to the full $M$-residual and handle the one-step
  case $\kappa_2(S)=1$ separately.

  A counterexample to prefactor-one contraction is
  $S=\operatorname{diag}(2,101)$ and $r_0=(1,0.1)$.
  The first CG residual is $(99/301,-990/301)$, whose norm is larger than
  $\|r_0\|_2$. This matrix has the required form $S=I+AA^\top$ with
  $A=\operatorname{diag}(1,10)$.
- [ ] **Randomized Kaczmarz:** distinguish error contraction from residual
  contraction. The conversion in the draft introduces
  $C_{\rm in}=\kappa_2(S)^2$. Carry this through every affected inner-solve
  bound. In a total-work sum, its logarithmic contribution generally appears
  for each solve, not just once. Check the sampling assumptions against the
  [Strohmer--Vershynin result](https://arxiv.org/abs/math/0702226).
- [ ] **Block Kaczmarz:** the implementation draws a partition once and then
  cycles through its blocks. A theorem for independent random block sampling
  cannot be applied without further justification. Prove a bound for the
  implemented cyclic method, possibly per sweep, and translate sweeps into
  the counted block projections.
- [ ] **GMRES:** establish a residual estimate in the $M$ norm for the actual
  unscaled, unrestarted implementation. Singular values alone do not give a
  general nonsymmetric GMRES rate. The special structure of this $H$ may
  supply a bound, but the derivation and norm conversions need to be shown.

## 5. Fix the finite-termination and zero-residual cases

**Priority: correctness.** Locations: the paragraph after `alg:main` and the
opening of the inner-work subsection.

- [ ] Restrict the geometric-contraction stopping argument to $\sigma>0$ and
  $\bar R_k>0$. Convergence of the inner residual to zero alone does not imply
  finite termination when $\sigma=0$.
- [ ] Remove the assertion that a warm start automatically passes the test
  when the outer point is a fixed point. If $s^\star(z)=0$, the monotonicity
  lemma gives $\|s\|_M\le\|\varepsilon\|_M$. Acceptance with $\sigma<1$
  then requires $s=\varepsilon=0$. An arbitrary warm start need not satisfy
  that condition.
- [ ] Specify the initial pair $(\xi^{(0)},\nu^{(0)})$ and the convention
  for exact termination. Distinguish an ideal exact solve from an iterative
  solve with a positive relative tolerance. A finite inner-iteration cap is
  an implementation safeguard, not a consequence of the expected-work bound.

## 6. Correct the comparison of coupled and Schur conditioning

**Priority: incorrect statement.** Location: `rem:M-scaled-singular-values`.

- [ ] Keep the displayed singular-value formula, but replace the assertion
  that the scaled coupled matrix has the same conditioning as the Schur
  complement. Account for the extra unit singular values and eigenvalues.
- [ ] Check the square nonsingular case: with
  $\widetilde H=M^{1/2}HM^{-1/2}$,
  $\kappa_2(\widetilde H)=\sqrt{\kappa_2(S)}$, not $\kappa_2(S)$.
- [ ] Check the rectangular case separately. For $A=[1\;0]$ and both step
  sizes equal to one, $\kappa_2(\widetilde H)=\sqrt2$ while
  $S=[2]$ has condition number one.

## 7. Complete the outer convergence statements

**Priority: proof completeness and notation.** Locations:
`thm:relative-error-criterion`, `thm:linear-convergence`, and the following
rank-deficient remark.

- [ ] Write out the short finite-dimensional Fejer argument currently
  omitted: boundedness, continuity of $s^\star$, a fixed-point cluster point,
  and convergence of the whole sequence. Explain how the reported primal
  output $v^{(k)}$ and multiplier $\nu^{(k)}$ recover a KKT point.
- [ ] Relate the controlled step norm to a computable KKT certificate at
  $(v,\nu)$. The identities to use are

  $$
  u+A^\top\nu=-\gamma_x^{-1}s_x,\qquad
  Av-b=\gamma_\lambda^{-1}s_\lambda+A s_x,
  \qquad u\in\partial f(v).
  $$

  Keep the best-iterate squared-residual rate distinct from a last-iterate
  rate or an objective-gap rate.
- [ ] Define $q=\hat z-z=s+\varepsilon$ before it is used in the linear-rate
  proof. Make the multiplier split explicit using $(A^\top)^\dagger$ rather
  than leaving the existence and bounds on $w_1,w_2$ implicit.
- [ ] Give the projection argument for the rank-deficient extension.
  Identify the multiplier nullspace, the projection onto
  $\operatorname{range}(A)$, and how the one-step estimate yields contraction
  of distance to the fixed-point set. Replacing the smallest singular value
  by the smallest positive one is not a complete proof by itself.
- [ ] State precisely which experimental formulations meet smoothness and
  strong convexity. The current linear-rate theorem does not directly cover
  LPs, the nonsmooth LASSO formulation, or inequality QPs written using an
  indicator function for nonnegative slacks.

## 8. State what the theory covers in the implementation

**Priority: scope of the claims.** Locations: adaptive-parameter discussion,
the acceptance tests in [drs.py](code/src/drs.py), and the conclusion.

- [ ] For varying $\theta_k,\sigma_k$, extend the fixed-parameter proof or
  qualify the claim that the theorem applies. Check a uniform positive
  decrease margin such as
  $\inf_k\theta_k(2-\theta_k-2\sigma_k)>0$ and the uniform bounds needed
  for the inner-work estimates. Pointwise admissibility alone is insufficient
  for the stated uniform constants.
- [ ] Separate the exact relative-error test from the code's machine-precision
  floor. The code accepts a residual below the maximum of the relative
  threshold and an absolute floor. Explain the numerical convention; do not
  call the $\sigma=0$ experiment mathematically exact.
- [ ] Distinguish inner iteration counts from total computational work.
  The acceptance test evaluates the proximal map repeatedly inside a solve.
  Include that cost, matrix operations, and solver setup when making a total
  cost claim. One block projection, one row projection, and one Krylov step
  have different costs.
- [ ] Describe the observed insensitivity of outer iteration counts to
  inexactness as an empirical finding. The current decrease and linear-rate
  constants depend on $\sigma$ and do not prove that insensitivity.

Suggested order: repair items 1--5, correct the spectral statement, complete
the outer-proof details, and then revise the abstract and experimental
interpretation to match the results actually established.
