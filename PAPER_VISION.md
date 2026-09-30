# Paper vision: total work for warm-started inexact splitting

For Pranav and Liza. September 30, 2026.

This develops the direction from our discussion:

> A theory of total computational work for relative-error splitting methods
> with warm-started deterministic or randomized inner solvers.

The immediate corrections to the current draft are in
[PROOF_FIXES.md](PROOF_FIXES.md). This document describes the larger research
direction. **The theorem statements and complexity forms below are targets,
not results already established by the draft.**

## 1. The central idea

Treat successive inexact subproblems as a tracking problem. An accepted
approximate solution carries information into the next solve. The analysis
should quantify how much additional work is needed as the outer iterate moves,
and how that work accumulates until a computable solution certificate reaches
the requested accuracy.

The paper should give readers a reusable way to answer three questions:

1. How accurately should this subproblem be solved before taking another outer
   step?
2. How much work will all the subproblems require together?
3. How should the splitting parameters, inner solver, and preconditioner be
   chosen when the objective is time to a certified solution?

The conceptual contribution would be an explicit connection between relative
accuracy, reuse of previous solutions, and total cost. The most compelling
result would show when increasingly accurate outer solutions can be obtained
with controlled amortized inner effort, while identifying the assumptions
under which that conclusion fails.

A possible working title is **The Cost of Inexact Splitting**. A more
descriptive option is **Total Work Bounds for Warm-Started Inexact Splitting**.
Choose the title after the central theorem is clear.

## 2. The literature sets a demanding standard

Counting inner work, using warm starts, and balancing inner accuracy against
outer progress all have substantial precedents. Our contribution needs a
specific theorem that improves understanding in the relative-error splitting
setting.

| Reference | Established connection | What our paper should add or compare |
| --- | --- | --- |
| [Eckstein and Yao, relative-error DR and ADMM](https://doi.org/10.1007/s10107-017-1160-5) | Relative-error inexact splitting and practical subproblem stopping rules. | Treat this as algorithmic groundwork. Identify the additional assumptions and conclusions of our work analysis. |
| [Alves, Lorenz, and Naldi, Remark 2.8](https://arxiv.org/html/2407.05893v2) | A degenerate preconditioned HPE framework includes several splitting methods. The remark explicitly raises the difficulty of bounding combined inner and outer iterations. | Develop work bounds for a clearly specified subclass with verifiable inner-solver assumptions. |
| [Alves and Geremia, Theorems 4.3 and 4.4](https://arxiv.org/pdf/1711.11551) | Their four-operator construction bounds outer iterations, inner iterations, and their product to reach prescribed tolerances. | Compare acceptance rules, warm starts, counted operations, and dependence on final accuracy. A generic claim that inner/outer complexity has never been studied would be false. |
| [Machart, Anthoine, and Baldassarre](https://arxiv.org/abs/1210.5034) | They optimize a model of finite-accuracy computational cost for inexact proximal-gradient methods. | Explain what a computable relative certificate and a trajectory-dependent warm-start analysis contribute beyond prescribed inner budgets. |
| [Lin, Mairal, and Harchaoui, Catalyst](https://jmlr.org/papers/v18/17-748.html) | Warm starts, stopping criteria, and global complexity are central to their acceleration framework. | Compare the actual assumptions and work bounds. Warm-started inner/outer analysis alone is not a novelty claim. |
| [Zhao, Frangella, and Udell, NysADMM](https://proceedings.mlr.press/v162/zhao22a.html), and [Rathore et al., rlaopt](https://arxiv.org/abs/2609.08136) | Randomized preconditioning already accelerates inexact ADMM, with theory and implementations. | Give a principled account of setup cost, reuse, relative stopping, and total work. Include relevant methods as baselines on shared problem classes. |
| [Giselsson and Boyd, metric selection](https://stanford.edu/~boyd/papers/metric_select_DR_ADMM.html) | Splitting parameters and metrics can be selected using outer convergence bounds. | Study how the cost of implicit steps changes the parameter choice. |

The proposed distinction is the combination of **computable relative tests,
explicit warm-start drift, conditional randomized analysis, and amortized work
to a solution certificate across multiple splitting methods**. We need a
theorem-by-theorem comparison before claiming this combination is new. The
sources above establish relevant precedents; they do not establish novelty
of the proposed result.

## 3. Generalize to the framework behind Remark 2.8

The recommended scope is a work analysis for a fixed-metric preconditioned
hybrid proximal extragradient (HPE) framework, with explicit assumptions on
the inner solver. The current linearly constrained problem becomes the first
detailed application.

Use the reduced formulation when the preconditioner is semidefinite. For a
factorization $M=CC^*$, work with the reduced variable $w=C^*u$ in a space
with an actual norm. Establish how a reduced residual certifies the original
problem. This avoids assuming that a seminorm controls components in its
kernel. The relevant starting point is Algorithm 1 and Theorem 2.7 of
[Alves, Lorenz, and Naldi](https://arxiv.org/html/2407.05893v2).

Make the framework modular, with four ingredients:

| Ingredient | Required verification |
| --- | --- |
| Outer geometry | An accepted pair gives a quantified decrease or another suitable outer complexity estimate. |
| Computable certificate | The tested residuals certify the original inclusion or KKT system without knowing an exact resolvent or optimum. |
| Inner convergence | A deterministic bound or a conditional expected bound controls the error relevant to the test. |
| Warm-start drift and cost | The actual retained solver state has a controlled error at the next subproblem, with its initialization and update costs counted. |

Each application should supply these four ingredients in a short verification
table. Keep any assumptions needed for existence, recovery of the original
variables, or finite inner termination explicit.

The current draft and the HPE paper use different relative-error tests.
Translate the tests and parameter ranges carefully. A crude norm comparison
may unnecessarily shrink the admissible range, so retaining each method's
native decrease estimate may be preferable.

Begin with a fixed metric and one inexact component. General maximal
monotonicity does not imply an efficient inner algorithm or a geometric
residual bound. The framework should make that extra computational structure
visible. A claim covering arbitrary operator splitting would be too broad
until its hypotheses can be checked for each method.

## 4. Make finite-accuracy total work the main theorem target

Let $\mathcal R(\widehat u)$ be a computable solution certificate. Define
$K_\epsilon$ as the number of inner problems attempted before returning
$\mathcal R(\widehat u)\le\epsilon$, including the terminal inner problem.
The certificate could be a KKT residual or a monotone-inclusion residual.
Specify whether any accompanying bound concerns a last iterate, a best
iterate, or an ergodic output.

Count the work of the complete algorithm:

$$
W_\epsilon=W_{\rm setup}
+\sum_{k=0}^{K_\epsilon-1}
\left(W_{{\rm outer},k}+W_{{\rm inner},k}+W_{{\rm test},k}\right).
$$

The test cost matters. In the present implementation, testing an inner
candidate can require another proximal evaluation. A claim about total
computational work should include that operation and preconditioner setup.

An ambitious target, under a uniform geometric inner bound and an appropriate
drift estimate, is a statement with the schematic form

$$
\mathbb E\!\left[\sum_{k=0}^{K_\epsilon-1}N_k\right]
\le A_0+A_1\mathbb E[K_\epsilon]
+A_2\log\!\left(1+\frac{R_{\rm init}}{\epsilon}\right).
$$

Here $R_{\rm init}$ is a deterministic upper bound on a precisely defined
initial residual scale.
The constants must expose dependence on conditioning, the relative tolerance,
the outer parameters, and the drift bound. For this target, they should not
hide further dependence on final accuracy or the iteration count. Convert the
iteration bound into work using the appropriate operation costs.

The key question is whether warm starts allow the accuracy cost to be
amortized over the trajectory. A bound obtained by multiplying every outer
iteration by the worst inner cost is a useful baseline for comparison.
If an additive logarithmic dependence is unattainable in the intended
generality, characterize the additional trajectory or regularity term that
is necessary and exhibit a case where it matters.

### A promising way to handle the final inner solve

Investigate a two-outcome inner stopping rule:

1. Accept the candidate for an outer update when its relative-error test
   passes.
2. Return a solution immediately when the candidate itself has a verified
   certificate $\mathcal R\le\epsilon$.

The second outcome can make finite-accuracy termination meaningful even at
an outer fixed point, where a purely relative test may demand an exact inner
solve. It must use an actual solution certificate. Replacing the relative
threshold with an arbitrary absolute error floor does not by itself provide
that certificate.

A potential proof route is a residual scale clipped at the requested
accuracy, with the necessary norm-conversion constants. Try to telescope its
logarithmic changes across outer steps. This is a proposed argument to
investigate, including the terminal solve and the random stopping time.

## 5. A sequence of theorem targets

| Target | Desired result | Main issue to resolve |
| --- | --- | --- |
| T1. Conditional stopping cost | A bound for any inner solver satisfying $\mathbb E[E_{k,l}^2\mid\mathcal F_k]\le C a^l E_{k,0}^2$, with the deterministic case included. | Random outer states, contraction prefactors, and the correct certificate norm. |
| T2. Drift under the implemented warm start | Relate the next initial error to the previous accepted error and the movement of the subproblem. | Retaining a complete graph pair, a multiplier, or a Krylov state can give different bounds. |
| T3. Amortized work to accuracy | Combine T1 and T2 with the outer geometry to bound $W_\epsilon$. | Residual drops, the last inner solve, and dependence between random inner and outer paths. |
| T4. Useful specialization under an error bound | Sharpen $K_\epsilon$ and hence total work under an explicitly stated local or global error-bound condition. | Identify which nonsmooth applications satisfy the condition and include the cost of reaching any local regime. |
| T5. Limits of the guarantee | Give examples showing why specified drift, solver, or certificate assumptions are needed. | Distinguish failure of a bound from failure of the algorithm. |

T1--T3 are the core. T4 would broaden the interpretation beyond globally
smooth, strongly convex objectives. A local metric-subregularity or
polyhedral error-bound result is a candidate, not an automatic consequence
of the existing proof.

For the general monotone case, keep the expected rate appropriate to the
chosen residual and averaging convention. Do not promise a last-iterate
linear rate without extra structure.

## 6. Use two applications that exercise different assumptions

### Application A: the current RNLA linear solve

Retain linearly constrained convex optimization as the detailed spectral
example. Supply the correct warm-start estimate for each implementation and
instantiate the inner bound for CG and randomized Kaczmarz. Include GMRES or
cyclic block Kaczmarz once their actual residual estimates are established.

Add one randomized preconditioner, if it yields a useful comparison. Analyze
and measure the cost of building it once and reusing it over the outer run.
Separate randomness in a one-time sketch from randomness in inner
iterations. If a guarantee holds on a good-sketch event, state its probability
and the behavior or fallback outside that event before claiming an
unconditional expected-work bound.

The RNLA contribution should explain when a more expensive setup pays for
itself through cheaper subsequent solves. Comparisons with NysADMM and
rlaopt should use compatible formulations and matched solution certificates.

### Application B: a nonlinear proximal solve inside primal-dual splitting

Use an inexact Chambolle-Pock/PDHG formulation of

$$
\min_x g(x)+h(Kx),
$$

with an easy dual proximal step and a nonlinear primal proximal subproblem.
A candidate is a smooth convex logistic loss $g$ and a regularizer $h(Kx)$
whose dual proximal map is cheap. The inner objective then has the form

$$
g(x)+\frac{1}{2\tau}\|x-r_k\|^2.
$$

For a smooth convex $g$ with a known Lipschitz gradient, the quadratic term
provides a strongly convex inner problem. Investigate gradient descent and
randomized coordinate descent as two inner solvers. Translate their usual
error or objective estimates into the residual used by the outer test, and
derive the drift caused by changes in $r_k$.

This example should demonstrate transfer beyond linear algebra. Include a
competitive method that handles the smooth loss explicitly, such as a
suitable forward primal-dual method. Establish where the implicit approach
earns its extra cost.

Davis-Yin is a natural third application if it follows with a short, useful
verification. A list of equivalent formulations would add less than two
fully worked applications with different inner computations.

## 7. Turn the analysis into a design decision

The paper will be more useful if the work bound changes a parameter choice.
Choose one of the following directions for the main paper.

**Joint choice of outer parameters and inner accuracy.** Develop a model for
$W_\epsilon$ as a function of the splitting step sizes, relaxation, and
relative tolerance. Compare its recommended settings with settings chosen
from the outer rate alone. Start with a problem class where the constants can
be estimated and the prediction can be tested without fitting every instance.

**Preconditioner setup versus reuse.** Compare the one-time setup cost with
the savings predicted over the remaining solves. A model involving sketch
rank, effective conditioning, and target accuracy could yield a practical
selection rule. Include unsuccessful regimes where setup costs dominate.

**Frequency of certificate checks.** When checks are expensive, study testing
every few inner steps or according to a growing schedule. Count both extra
solver work and saved checks. Prove a cost guarantee for the selected policy
before describing it as principled.

A fully adaptive policy can be a later extension. If included, it needs
uniform convergence margins and a work analysis that includes adaptation
cost. A policy tuned using an unknown optimum or an exact resolvent would
not meet the intended practical goal.

## 8. Experiments that could establish the main claim

Organize the numerical section around hypotheses. Reuse the current scripts
and results format, adding only experiments needed to test these questions.

| Question | Experiment | Evidence to report |
| --- | --- | --- |
| Does reuse explain the low inner cost? | Warm versus cold starts with the same solver, parameters, and stopping test. | Work to matched certificates, initialization errors, and inner counts along the trajectory. |
| Does the theory predict accuracy dependence? | Vary the requested final accuracy across several orders of magnitude. | Total operator evaluations and time versus accuracy, including the final solve. |
| Are residual drops the missing difficulty? | Controlled spectra and examples with abrupt residual changes. | Consecutive residual ratios, work spikes, and agreement with the proposed amortized bound. |
| Does the framework transfer? | Linear RNLA and nonlinear proximal examples using the same theoretical ingredients. | Separate checks of drift, inner contraction, and outer certificates. |
| Does optimizing work alter the chosen method? | Compare outer-rate tuning with the proposed work-based choice. | A reproducible case where the preferred parameters differ and the total-cost prediction is informative. |
| When does randomness help? | Deterministic, randomized, and preconditioned solvers at matched accuracy. | Setup-inclusive time, primitive operation counts, distributions across seeds, and unsuccessful regimes. |
| How much does certification cost? | Every-step testing versus a justified batched policy. | Time spent in tests, extra inner work, and total time. |

Include high-accuracy solves, reasonable fixed inner budgets, and a justified
decreasing-tolerance schedule as baselines. Give each a comparable tuning
budget. Where an existing relative-error method has the same outer updates,
attribute the new benefit to the work analysis or new policy being tested.

The main plots should show solution certificate versus cumulative work and
wall-clock time. Count matrix-vector products, row/block projections, gradient
evaluations, and proximal evaluations separately before assigning any common
cost weights. Report preprocessing and memory costs. Match hardware and
threading, disclose caps and failures, and include variability across seeds.

Exact residuals or exact subproblem solutions can be used on small diagnostic
instances to inspect the theory. They should not enter the deployed stopping
rule, and their diagnostic cost should be kept out of algorithm timings.

## 9. Keep the first paper focused

The essential package is:

- A corrected analysis of the current algorithm.
- A reusable work theorem in a fixed-metric HPE or comparable splitting
  setting, with verifiable inner-solver and drift assumptions.
- A finite-accuracy guarantee using computable certificates, including the
  terminal inner solve.
- Deterministic and randomized instantiations, with at least one nonlinear
  proximal application.
- One concrete design consequence and experiments testing its mechanism.

High-probability work bounds, changing metrics, multiple inexact components,
sublinear inner solvers, acceleration, and adaptive sketch refresh are
valuable extensions. Add them only when they sharpen the central message.
For example, a changing metric introduces new subproblem drift and setup
cost; it should enter the analysis as such.

Tightness would strengthen the paper. Start with examples showing necessary
dependence on conditioning and drift. An oracle-complexity lower bound is a
larger project requiring a precisely defined oracle and cost model. Do not
use “optimal” for a bound merely because it agrees with observed runs.

## 10. A concrete order of work

1. **Make the current result reliable.** Complete the proof checklist and
   align every theorem with the algorithm actually implemented.
2. **Write the abstract assumptions and candidate theorem.** State the work
   unit, filtration, warm-start rule, certificate, and finite-accuracy stopping
   convention on one page. Test each assumption on the current example.
3. **Resolve the finite-accuracy amortization question.** Try the combined
   relative-acceptance/solution-certificate rule. Search for small
   counterexamples as well as a proof. Record any additional hypothesis the
   argument actually needs.
4. **Verify the nonlinear application.** This is the main test of whether the
   framework has broader content. If its assumptions cannot be verified,
   narrow the claim rather than leaving the application formal.
5. **Choose one design consequence.** Use either parameter selection,
   preconditioner reuse, or certificate scheduling to turn the theorem into
   an actionable algorithmic recommendation.
6. **Run the hypothesis-driven experiments and rewrite the narrative.** Lead
   with the work-to-certificate result. Use outer convergence and spectral
   facts to support that result.

The standard for the finished paper should be concrete: a reader can take a
splitting method with an expensive implicit step, check a short set of
assumptions, obtain a total-work guarantee, and make a better implementation
decision. Achieving that would make the contribution useful well beyond the
particular linear system in the current draft.
