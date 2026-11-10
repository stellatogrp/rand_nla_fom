# Main numerical experiment figures

`fresh_v8/` holds the QP figures generated from fresh, instrumented evaluations:

1. `01_qp_outer_convergence`: original-problem residual versus outer solves.
2. `02_qp_inner_convergence`: residual versus cumulative inner updates.
3. `03_qp_wall_clock`: independent worker time, including all setup and traces.
4. `04_qp_theoretical_bounds`: measured inner graph residual versus verified
   deterministic GD/CG contraction bounds.

Each figure is saved as PNG and PDF. Supporting CSVs preserve run identities,
parameters, seed-level timing components, input hashes, and bound comparisons.
The theory figure retains any numerical exceedances in its CSV. It makes no
claim about unsupported solver bounds or an unverified outer KKT rate.

Fresh runs hold warm-selected parameters fixed across warm, fixed, and Gaussian
initialization. They use the original 1e-6 certificate, one core, independent
setup, and finite stopping budgets. Plotting is a separate CPU job after all
manifest tasks finish. It refuses missing results or truncated traces.

Reproduce from the standalone project directory:

```bash
uv run python -m plots.main_summary \
  --results-dir results/legacy/main_figures_v8_20261103 \
  --manifest figure_campaign/inputs/evaluate.csv \
  --output main_figures/fresh_v8
```

See `figure_campaign/README.md` for scheduling and provenance. The original QP
results remain preserved under `results/legacy/qp_evaluate_v7_20261031/`.

`collect.py` can wait for the dependent plotting job and download both the
scientific CSVs and figure outputs using only the Slurm skill helper. It polls
at the helper's one-minute interval, has a finite waiting budget, and stops
immediately on any helper error without connection retries. Progress is saved
in `collection_status.csv` and `collection.log`.

## Sigma and stride sweeps

`sweeps_v7/` contains separate sigma and stride figures for setup-inclusive
wall time, cumulative inner updates, and outer solves. Each setting uses
three seeds: points show arithmetic means without error bars.
Panels separate problem sizes and warm/fixed/Gaussian initialization. The sigma
sweep fixes s=1; the stride sweep fixes sigma=0.1; both fix L/c=10.
GD/CG model-prescribed settings are marked at their actual resolved values.

Reproduce with `uv run python -m plots.sweep_effects --results-dir
results/legacy/qp_tune_v7_20261031`. All six figures have
PNG/PDF versions and supporting run, summary, and source-hash CSVs.

`sweeps_v7/warm_start_sweeps` combines only the warm-started wall-time panels,
with sigma on the top row and stride on the bottom row. It reports arithmetic
means without error bars and has PNG/PDF versions. Reproduce with
`uv run python -m plots.warm_sweeps --results-dir results/legacy/qp_tune_v7_20261031`.

`fresh_v8/warm_only/` contains warm-start-only versions of all four fresh QP
figures: outer convergence, inner convergence, wall time, and GD/CG theoretical
bounds. They use arithmetic means without error bars or shaded ranges. Native
baselines remain as references. Reproduce with
`uv run python -m plots.warm_evaluation --results-dir
results/legacy/main_figures_v8_20261103`.

The results these figures read were moved to `results/legacy/` on 2026-11-06;
the three plotting scripts above are sealed with their original default paths,
so pass `--results-dir` as shown.
