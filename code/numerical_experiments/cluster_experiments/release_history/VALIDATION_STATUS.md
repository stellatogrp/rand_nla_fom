# Completed release validation

All 12 accounting_v7 cluster pilots completed and passed their accounting
audits. Jobs: 16520266_3 (targeted cap regression) and
16520278_0-2,4-11 (remaining cases, serialized).

The v6 suite had 11 passing audits and one failed Kaczmarz cap audit.
The failed v6 result is preserved; v7 uses integer update limits.
The v7 capped Kaczmarz solve took exactly 64,000 row updates. Calibration
performed another eight row updates, separately recorded and charged.

The data for these validation cases are seeded synthetic problems, not
performance evidence for LIBSVM/OpenML/UCI or random-feature workloads.

Evidence:

- `results/validation_v7_20261031/accounting_audit.csv`
- `results/validation_v7_20261031/publication_tables/`
- `results/validation_v7_20261031/figures/summary.csv`
- Original per-run `result.csv` and per-task audits in the same directory.

The full local test suite passed 316 tests. Ruff and formatting checks passed.
Existing production sweeps were not modified or restarted.
