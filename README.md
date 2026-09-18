# rand_nla_fom

Integration of randomized numerical linear algebra into a Douglas--Rachford
splitting method: the algorithm solves

```text
minimize f(x)  subject to  A x = b
```

with the block-diagonal metric `M = diag(I/gamma_x, I/gamma_lambda)`, and
solves the linear system in its inner step *inexactly*, stopping at the
relaxed relative-error test `||epsilon||_M <= sigma ||s||_M`.

## Layout

```text
code/      Python package, experiment scripts, and plotting
tex/       Paper and notes
results/   Generated CSVs and figures (git-ignored, never edited by hand)
```

`results/` is empty until a study is run; both `code/experiments` and
`code/plots` create the directories they need.

## Getting started

```bash
cd code
uv sync                                        # install the package and dev tools
uv run python -m experiments.run_experiments standard --plot
```

See [`code/README.md`](code/README.md) for the method summary, the study
catalogue, and the CSV contract between experiments and plots.

## Checks

```bash
cd code
uv run ruff check . && uv run ruff format .
uv run pytest
```
