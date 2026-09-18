# rand_nla_fom

This paper studies the integration of tools from randomized numerical linear algebra (RNLA, also called RandNLA). Specifically it studies an algorithm derived from Douglas-Rachford splitting which uses an inexact linear solve to solve the problem min_{x: Ax=b} f(x). 

## Layout
- `code/`: Python package and experiment scripts (run with `uv run`)
- `tex/`: paper and notes (paper-writing skill conventions, load `definitions.tex`)
- `results/`: generated CSVs and figures (gitignored, never edit by hand)

## Rules
- No notebooks. Scripts under `code/experiments/` with argparse flags.
- Save results as CSV; plotting lives in `code/plots/` and reads the CSVs.
- Run `uv run ruff check . && uv run ruff format .` and `uv run pytest` before committing.
- Do not fabricate numbers, citations, or proof steps. Mark anything unverified as TODO.