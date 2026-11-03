# Numerical experiment conventions

Use US English in code comments, docstrings, CLI help, documentation, plot text,
and new identifiers. Preserve external APIs, URLs, dataset identifiers, proper
names, exact quotations, and published titles.

This directory contains experiments for the randomized numerical linear algebra
and inexact splitting paper. Keep experiments as scripts with CLI flags, never
notebooks. Save results as CSV, keep plotting separate, and never edit generated
results by hand. Do not fabricate numbers, citations, or proof steps; mark
unverified claims as TODO. Cluster work follows `cluster_experiments/AGENTS.md`
and the slurm skill. Run the relevant Ruff and pytest checks before committing.
