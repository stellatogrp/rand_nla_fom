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
  --results-dir results/main_figures_v8_20261103 \
  --manifest figure_campaign/inputs/evaluate.csv \
  --output main_figures/fresh_v8
```

See `figure_campaign/README.md` for scheduling and provenance. The original QP
results remain preserved under `results/qp_evaluate_v7_20261031/`.

`collect.py` can wait for the dependent plotting job and download both the
scientific CSVs and figure outputs using only the Slurm skill helper. It polls
at the helper's one-minute interval, has a finite waiting budget, and stops
immediately on any helper error without connection retries. Progress is saved
in `collection_status.csv` and `collection.log`.
