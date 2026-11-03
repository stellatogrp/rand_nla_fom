# Corrected v7 campaign

The first priority is the corrected QP tuning grid (2,079 configurations) and
Kaczmarz screen (174 configurations). Both are upgraded from the original
QP tuning manifest using the frozen v7 planner. Unsupported Nyström model
prescriptions are removed, and Kaczmarz checks use block-scaled strides.
All three initialization modes remain represented. Main QP budgets remain
1,800 solve seconds, one million inner work units, 50,000 outer iterations,
300 seconds and 100,000 work units per outer solve. Every worker also has a
finite setup allowance, memory cap, and Slurm hard limit.

The QP lane uses one core, 16 GB Slurm memory / 12 GB worker address-space cap,
and eight serial configurations per array task, at concurrency eight. Validate
and inspect the one-million-variable rank-128 Nyström case before releasing the
full array. The screen has its own concurrency-one lane, 120-second solve
budgets, and a largest-block representative pilot. CPU timings are pinned to
one advertised CPU family (`icelake`). Main cases load and calibrate separately.

Independent real-data pilots follow on two native-labeled matrices: tall
realsim (72,309 by 50,000) and wide smallNORB (24,300 by 30,000). Each compares
LASSO with Nyström-PCG and PDHG, and logistic regression with GD and PDHG, at
two penalty/step scales. These are exploratory pilots, not parameter-selection
results. Solve budgets are 30 minutes, setup allowances 15 minutes, and every
worker has its own loading/setup/calibration. The main tolerance stays 1e-6.
Convergence is not assumed in advance. If the pilots remain censored, inspect
residual histories before expanding to full regression sweeps.

`source_seal.json` covers launch code, fixed manifests, the plan, and the main
Slurm runner. `snapshots/` preserves prelaunch states; the first preflight
snapshot differs only in line wrapping corrected before submission. Neither
snapshot nor the preserved preflight seal is overwritten. Once jobs are queued,
create a new campaign directory for code or input changes. Do not rerun prepare
with changed inputs under this name. The numerical release remains unchanged.

Follow-up planning refuses missing runs, unresolved failures, shared-loading
results, or converged rows without clean timing. Explicit resource failures
remain valid censored outcomes. Commands after their prerequisite stages finish:

```bash
uv run python -m campaign_v7.advance qp-steps \
  --manifest campaign_v7/manifests/qp_tune.csv \
  --results-dir results/qp_tune_v7_20261031 \
  --output results/campaign_v7_plans/qp_steps.csv
uv run python -m campaign_v7.advance kaczmarz-expand \
  --manifest campaign_v7/manifests/kaczmarz_screen.csv \
  --results-dir results/kaczmarz_screen_v7_20261031 \
  --output results/campaign_v7_plans/kaczmarz_expanded.csv
uv run python -m campaign_v7.advance regression-readiness \
  --manifest campaign_v7/manifests/regression_pilots.csv \
  --results-dir results/regression_pilots_v7_20261031 \
  --output results/campaign_v7_plans/regression_readiness.csv
# After the generated steps manifest has actually run to completion:
uv run python -m campaign_v7.advance qp-evaluate \
  --manifest results/campaign_v7_plans/qp_steps.csv \
  --results-dir results/qp_steps_v7_20261031 \
  --output results/campaign_v7_plans/qp_evaluate.csv
```

The steps grid includes the QP reference ratio sqrt(kappa)=100, GD's minimax
inner step, solver-specific supported model arms, and native baseline step
sweeps. Evaluation uses seeds 3, 4, 5, disjoint from selection seeds 0, 1, 2.
Follow-up manifests have separate byte/source seals and can be run by the same
campaign runner. Inspect generated task counts and resource measurements before
submitting each next array through the slurm skill. A dependent planning job
prepares the manifest; it does not launch an unvalidated array automatically.
