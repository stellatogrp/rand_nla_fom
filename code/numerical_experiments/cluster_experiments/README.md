# Large-scale Della experiments

This is an isolated uv project inside `numerical_experiments`. Run all commands
below **from this folder**. It does not modify the original small dense studies.
The remote project root is `/scratch/gpfs/BSTELLATO/pranav/rand_nla_fom` and
contains the contents of this folder. The connection alias is `della-gpu`, but
these jobs run on CPU compute nodes, using the discovered `bstellato` account.

## Original-format studies

[`studies/`](studies/README.md) runs these problems (the equality QP and the
LASSO by ADMM, ridge logistic regression by the proximal point method) and all
of their inner solvers in the format of the original studies of
`../README.md`: the same configurations per instance (the classical and
near-exact baselines, the sigma sweep with `sigma*`, the stride sweep with
`s*`), the original stopping rule `||z_k|| <= 1e-9` and hard stops, and the
original `parameters.csv`, `results.csv`, `histories.csv`, `curves.csv`, and
`summary.csv` with the same columns and theorem checks. `plots.studies` draws
the original figures from them. On Della a campaign (`studies.campaign`,
`slurm/studies_campaign.slurm`) runs every configuration of every study as its
own array task, in separate lanes for the QP, Kaczmarz, and regression studies,
so a stalled configuration delays no other result; `studies_v1/` is the first
campaign, and `studies_v2/` reruns its QP and regression studies with a
1,200 s instead of a 300 s limit per inexact run. It imports the sealed
`releases/regression_v9` engine and changes nothing below.

## Results layout

The results of the earlier pipeline (`experiments.run`, one directory per task
configuration) were moved on 2026-11-06 from `results/<experiment>/` to
`results/legacy/<experiment>/`, so that `results/` holds only the
original-format studies (for example `results/studies_v1/`). Commands and sync
configurations that read those results point to `results/legacy/`; the launch
records (`*/LAUNCH_STATUS.md`, `release_history/`) and the sealed sources keep
the paths they were written with. Pass `--results-dir
results/legacy/<experiment>` to a sealed tool whose default names an old path.
Della keeps the same layout, so that `sync down` never copies the old results
back to their old place; the human moved them there on 2026-11-06, from a
terminal, with:

```bash
ssh della-gpu 'cd /scratch/gpfs/BSTELLATO/pranav/rand_nla_fom/results && mkdir -p legacy && for d in */; do case "$d" in legacy/|studies_*/) ;; *) mv -n "$d" legacy/ ;; esac; done && ls'
```

## Choosing matrix sources

For the five NysADMM dataset families, downloads, native labels, random Fourier
features, streamed dense storage, and source-code provenance, see
[NysADMM datasets](datasets/NYSADMM.md). Raw downloads and small pilots are now
available for cod-rna, rcv1, realsim, smallNORB, and CIFAR-10.

New manifests default to **offline random sparse designs**, so neither a
SuiteSparse download nor a conversion job is needed. Existing manifests retain
their original data source. The QP generator and all solver/sweep interfaces are
unchanged.

```bash
# Offline defaults: seeded Gaussian and random-sign sparse matrices.
uv run python -m experiments.manifest pilot --output manifests/random_pilot.csv \
  --seeds 0
uv run python -m experiments.manifest tune --output manifests/random_tune.csv

# Choose sources explicitly, including files from any alternative collection.
uv run python -m experiments.manifest tune --dataset-spec datasets/my_sources.json \
  --data-dir data --output manifests/alternative_tune.csv

# Explicitly restore the original SuiteSparse selection when available.
uv run python -m experiments.manifest tune --dataset-preset suitesparse \
  --output manifests/suitesparse_tune.csv
```

Copy `datasets/local.example.json` and change its paths, or mix sources in a
single JSON object:

```json
{
  "collection_design": {
    "kind": "local",
    "path": "alternative/design.mtx",
    "source": "Collection name, matrix identifier, and original URL"
  },
  "another_design": {
    "kind": "local",
    "path": "alternative/design.mat",
    "key": "Problem.A",
    "source": "Collection name and identifier"
  },
  "generated_design": {
    "kind": "random_sparse",
    "rows": 100000,
    "cols": 50000,
    "entries_per_row": 20,
    "distribution": "normal",
    "seed": 314159
  }
}
```

Supported local inputs: SciPy sparse `.npz`, Matrix Market `.mtx`/`.mtx.gz`,
MATLAB `.mat` (default key `A`, dotted struct keys such as `Problem.A`), and
two-dimensional real `.npy`. MATLAB v7.3/HDF5 needs conversion beforehand.
Prefer sparse formats for large inputs; reading a dense file does not make its
storage requirements sparse. Paths are relative to `--data-dir`, or absolute.
For Della, use relative paths and put matching files under remote `data/`.
Code sync excludes `data/`; transfer external files separately before running.

The default random preset is `datasets/random.json`: two 100,000 × 50,000
matrices, Gaussian and random sign, with 20 column draws per row. The default
pilot scales each to 6,000 × 2,000; smoke scales to 48 × 24. Explicit custom
specifications always retain their requested dimensions, even in smoke/pilot;
use a small specification for local smoke tests. Every design runs in native
and transposed orientations, providing matched tall/wide cases. Values are
scaled by `1/sqrt(entries_per_row)` before the common column normalization.
Generation uses O(rows × entries_per_row) storage, never a dense rows × cols
array. Sampling is with replacement; duplicates are summed, so actual nnz can
be smaller. These random designs have no claimed prescribed condition number;
the equality QP still has its exact `10^4` guarantee.

The matrix seed is independent of each response/experiment seed: all solvers
and response replicates see the same design. Change the source seed to add a
new matrix instance. Source specs are embedded in task CSVs, so later runs need
no JSON catalog file. Local inputs are pinned by SHA256 during manifest creation
and checked on loading; changing input bytes creates new task IDs when manifests
are regenerated and makes old tasks fail explicitly. Provenance records source
settings, checksums, dimensions, nnz, normalization, and response seed. Tuning
groups and plot groups distinguish different specs sharing the same dataset name.
Stepsize sweeps inherit the source specs from their tuning winners.

No loader downloads files or silently substitutes random data. To support another
format, add a loader in `experiments/matrices.py` and register it with
`register_source(name, loader)`. Loaders receive `(spec, data_dir)` and return
`(matrix, metadata)`; shared validation canonicalizes CSR and rejects complex,
empty, zero, or nonfinite designs. This requires no changes to problem or solver
code. Streamed dense NPY inputs use a bounded row reader instead of CSR
conversion. Responses default to synthetic for existing sources. NysADMM prepared
sources explicitly use pinned native labels and record their class mapping.

Optional preflight for large sources runs in a compute allocation:
`slurm/data.slurm` with `--export DATASET_SPEC=datasets/my_sources.json` validates
sources and writes `results/matrix_validation.csv`. For a **small workstation**
fixture, use `uv run python -m experiments.matrices --dataset-spec <file>
--allow-local`. Random designs are generated independently per task and their
generation cost is included in instance setup time.

## Problems and comparisons

| Family | Instances | Main method | Comparisons |
| --- | --- | --- | --- |
| Equality QP | 40³ = 64,000; 64³ = 262,144; 100³ = 1,000,000 variables; n/4 equalities | Inexact ADMM, matching the graph test in `../admm.py` | GD, accelerated GD, CG, GMRES(50), Nyström-PCG; randomized and cyclic block Kaczmarz; PDHG, OSQP, indirect SCS |
| LASSO | Configurable random/local/SuiteSparse designs in both orientations | Inexact ADMM on a QP reformulation | GD, CG, Nyström-PCG; PDHG, OSQP, indirect SCS |
| Ridge logistic | Same designs, seeded Bernoulli labels | Inexact proximal point, vanilla GD inner steps | Reference PDHG and indirect SCS with exponential cones; OSQP recorded as not applicable |

The synthetic Hessian is
`Q = I/kappa + (1-1/kappa) L_3D/12`, where `L_3D` is the periodic
three-dimensional grid Laplacian and `kappa = 10^4`. Even grid sides give
**exact** eigenvalue endpoints `1/kappa` and `1`, certified by the constant
and checkerboard modes. Randomly selected coordinate equalities interact
through the coupled objective; their projection needs no factorization.
A seeded primal-dual KKT point determines `q` and `b`, providing a reference
solution without solving a large reference problem. This structured generator
differs from the old dense random eigenbasis generator.

There is no claim that `n > 50,000` alone precludes every sparse factorization.
The 3D graph produces substantial fill-in, particularly at one million variables.
A dense float64 Hessian alone would use approximately 32.8 GB, 550 GB, and 8 TB
at these sizes, before factors and workspace. Our paths store sparse matrices,
vectors, low-rank sketches, and bounded Krylov/block workspaces. OSQP is a real
direct-factorization baseline, run under the same memory/time ceilings; a
factorization that fits and wins must be reported. Resource failures are explicit
results, not evidence of a general mathematical impossibility. No interior-point
reference is required for the large instances.

Optional SuiteSparse catalog (shape/nnz validated after conversion):

| Matrix | Native shape | Nonzeros | Use |
| --- | --- | --- | --- |
| [JGD_GL7d/GL7d16](https://sparse.tamu.edu/JGD_GL7d/GL7d16) | 955,128 × 460,261 | 14,488,881 | Large native tall and transposed wide |
| [LPnetlib/lp_osa_30](https://sparse.tamu.edu/LPnetlib/lp_osa_30) | 4,350 × 104,374 | 604,488 | Large native wide and transposed tall |
| [JGD_GL7d/GL7d12](https://sparse.tamu.edu/JGD_GL7d/GL7d12) | 8,899 × 1,019 | 37,519 | Pilot only |
| [LPnetlib/lp_ken_18](https://sparse.tamu.edu/LPnetlib/lp_ken_18) | 105,127 × 154,699 | 358,171 | Optional cached dataset |

Every design is column-normalized after choosing its orientation; zero columns
remain zero. Original benchmark objectives, LP bounds, and responses are not used.
LASSO responses are seeded sparse signals plus Gaussian noise. Logistic labels
are seeded Bernoulli draws from scaled linear scores; these are **synthetic
responses on real sparse designs**, not a claim of native classification labels.
The logistic objective is mean log loss plus a ridge term, ensuring a finite
minimizer even on separable data.

LASSO is explicitly the QP

```
min_{u,v >= 0} 0.5 ||A(u-v)-b||² + lambda 1ᵀ(u+v),   x = u-v.
```

`lambda = 0.01 ||Aᵀb||_infinity`. ADMM uses Hessian-vector products with
`[A,-A]`, never its Gram matrix, and the nonnegative orthant projection.
OSQP/SCS use the equivalent residual lift `r=A(u-v)-b`, with diagonal
quadratic term `0.5 ||r||²`. The split QP Hessian is singular, so it does not
have a finite SPD condition number; its proximal subproblem is SPD. The finite
`kappa >= 10^4` guarantee concerns the synthetic equality QP, as in the old study.

## Sweeps and the meaning of optimal

`experiments/manifest.py` generates immutable task configurations as CSV rows;
each configuration has a stable SHA256-derived run ID. Standard tuning uses
seeds 0, 1, 2, matched across solvers and parameter choices.

1. At `ratio=10`, sweep `sigma={0.01,0.05,0.2,0.5,0.9,1}` with stride 1.
2. Sweep stride `s={1,2,5,10,20}` at `sigma=0.1`. Here `stride` means the
   paper's `s`, called `b` in the old scripts. Sigma 1 is a bounded diagnostic
   outside the theorem's strict range. A near-exact inner reference is included.
3. Add the paper's cost-model `sigma*` and calibrated integer stride recommendation
   for each problem, penalty, GD/CG/Nyström-PCG/block-Kaczmarz configuration where
   verified contraction constants are available. Calibration and its own solver
   setup are charged to tuning time. The integer search covers 1 through 1000,
   matching the existing study; it is a recommendation within that search range.
4. Kaczmarz block sizes are `{1,8,32,128,512}`; Nyström ranks are `{32,128}`.
   GD/AGD act on the inner optimization objective. CG/GMRES/PCG and Kaczmarz solve
   the same optimality system `(Q+cI)x=cw-q`. Each Kaczmarz update is a true block
   row projection using a Cholesky factor of `H_block H_blockᵀ`; an LRU cache
   retains at most eight blocks. Later block factorizations count as solve work.
5. After tuning finishes, generate `steps` from converged tuning results.
   For each problem/solver/block-size/rank group, select the configuration with
   the smallest geometric-mean **setup-inclusive** runtime, requiring all observed
   seeds to converge. A solver group with no eligible configuration stops
   selection: inspect censored runs or increase budgets rather than calling one
   optimal. OSQP/logistic not-applicable records are excluded from this selection.
6. Sweep `ratio={0.1,1,10,100,1000}` with `c=L/ratio` for ADMM and
   `gamma=ratio/L` for proximal point. Add the QP reflected-prox contraction
   recommendation `c=sqrt(mu L)` explicitly. For GD, test 0.5 and 1 times
   `2/(L+mu+2c)`, including the minimax constant inner step. Recommendations are
   recomputed at every new penalty. Baseline PDHG balance, OSQP rho, and SCS scale
   are also swept over `{0.01,0.1,1,10,100}`.

`L` is exact for the synthetic QP. For sparse real designs, a certified
`||A||_1 ||A||_infinity` bound is used, not an optimistic power-iteration estimate;
streamed dense designs use the certified squared Frobenius-norm bound.
thus optimal inner steps are optimal for these certified spectral bounds.
The model uses Euclidean gradient contraction for GD, conservative CG residual
bounds (with preconditioner condition bounds for Nyström), and an expected
uniform-partition-block Kaczmarz bound. Restarted GMRES, accelerated GD, and
cyclic Kaczmarz have no claimed model-optimal values; they use measured grids.
Nyström preconditioning follows
[Frangella, Tropp, Udell](https://arxiv.org/abs/2110.02820).

There is no known universally optimal outer gamma or runtime-optimal ADMM c for
all these problems and inner solvers. Model recommendations and empirical grid
winners are distinguished. Refine around a measured winner with `--ratios`, and
use `--seeds 3 4 5` for a separate evaluation set after selecting parameters.
Do not describe a finite-grid winner as a globally optimal parameter.
`experiments/theory.py` and `plots/style.py` are standalone snapshots of the
existing theory and plotting modules, preserving their formulas and encodings.

## Inner warm-starting ablation

Every new `smoke`, `pilot`, and `tune` manifest includes three inner starts
for each QP/LASSO/logistic method, block size, rank, sigma, and stride:

- `warm`: use the previously accepted inner solution; the first start is zero.
- `fixed`: reset to the same constant vector at **every** outer iteration,
  zero by default (`--fixed-initial-value` sets its entries).
- `gaussian`: draw a fresh vector at **every** outer iteration from
  `N(0, gaussian_scale²/n I)`, where n is the inner system dimension. Default
  scale 1 gives expected squared norm 1; the LASSO split uses its full 2d space.

Outer primal/dual/proximal centers still start at zero in every arm. No inner
point is projected or clipped before the inner solve. Krylov recurrences and
AGD momentum restart for each subproblem, while each arm retains its one-time
Nyström setup or bounded Kaczmarz cache. The Gaussian RNG uses
`SeedSequence([initialization_seed, outer_iteration])`, with default seed
`experiment_seed + 200003`. It is independent of matrix, response, sketch, and
block-sampling RNGs, and reproduces the same draws after checkpoint resume.
`--initialization-seed` and `--gaussian-scale` can override the defaults.

```bash
# All three modes are the default; choose a subset explicitly if needed.
uv run python -m experiments.manifest pilot --output manifests/init_pilot.csv \
  --seeds 0 --inner-initializations warm fixed gaussian
uv run python -m experiments.manifest tune --output manifests/warm_tune.csv \
  --inner-initializations warm
```

Tripling the inner runs increases the grid size; external PDHG/OSQP/SCS references
remain single runs because this harness does not expose their internal starting
points. Array sharding and the cap of four remain in effect. Existing manifests
without an initialization field preserve the original warm-start behavior.
`steps` retains the initialization mode of each selected configuration, with
selection performed separately for each arm. Standard sweep plots label the mode.

For the clean paired ablation, finish a warm-start sweep first, then generate:

```bash
uv run python -m experiments.manifest initialization \
  --from-results results/warm_tune_v1 --from-manifest manifests/warm_tune.csv \
  --output manifests/initialization.csv --seeds 0 1 2
```

This chooses the best converged warm configuration per problem/inner solver/
block size/rank, resolves model-recommended sigma/stride to their **measured
numeric values**, and uses those exact values and the same ratio/GD factor in
all three arms. The paired phase can also use a completed `steps` manifest and
its results. Every task records the source reference run ID; sources, response
seeds, stopping rules, budgets, and setup policy match across arms. Choose new
response seeds for held-out comparisons after tuning when desired.

The warm-start cost-model optimum is not a theorem for fixed or Gaussian
restarts: its warm-start residual bound no longer applies. On ordinary sweeps,
their inherited recommendations are labeled `warm_start_reference_model`, not
claimed optimal. In the dedicated ablation the parameters are already numeric
and held fixed. Gaussian generation/reset work is timed inside the solve;
`initialization_seconds` is a reported component, not an extra charge added to
`total_seconds`. Results record the mode, seed, scale, fixed value, and each
history row records the initial point norm and initial graph residual.

Run `plots.report` on the ablation results as usual. Besides mode-labeled figures,
it writes `initialization_comparison.csv` with per-instance total time, solve
time, inner work, and check-count ratios relative to warm starts. Ratios greater
than one favor warm starts. Only pairs with matching **resolved** parameters and
both arms converged receive ratios; failures and missing references remain
explicit. Aggregate runtime plots likewise keep all three modes separate.

## Local validation

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format .
uv run pytest
uv run python -m experiments.manifest smoke --output manifests/smoke.csv --seeds 0 \
  --time-limit 5 --max-inner 20000 --max-outer 2000
uv run python -m experiments.run --manifest manifests/smoke.csv --task-id 0 \
  --results-dir results/smoke
uv run python -m plots.report --results-dir results/smoke --output results/smoke_figures
```

The runner refuses non-smoke workloads outside a Slurm allocation. Numerical
tests check spectral endpoints and planted KKT conditions, all seven inner
solvers against the same exact small system, LASSO lifting equivalence, work
limits, and agreement with independent SCS and reference PDHG solutions.

## Optional SuiteSparse preparation

The public SuiteSparse endpoint redirected to an HTTP file store that returned
403 during preparation on 2026-10-30. Full benchmark downloads are therefore
**not staged or claimed verified**. The HTTPS file-store endpoint also failed.
The downloader fails visibly and never replaces real data with synthetic data.
Retry from the login node, or place the original SuiteSparse `.mat` files in
`data/` from an existing collection cache, then run `download` to register their
SHA256 checksums without downloading again. Compute conversion validates exact
shapes, nnz, finiteness, and source checksum, and saves CSR NPZ plus CSV provenance.
TODO: verify all three live dataset downloads/conversions before SuiteSparse runs.
This does not block the default offline random experiments or alternative local
files loaded directly through a dataset specification.

In your own terminal, on the login node (downloads/setup only):

```bash
ssh della-gpu
cd /scratch/gpfs/BSTELLATO/pranav/rand_nla_fom
bash slurm/env_setup.sh
.venv/bin/python -m experiments.data download --datasets GL7d12 GL7d16 lp_osa_30
```

The skill requires human-run bootstrap; it has been completed and `doctor`
confirmed the remote venv. `data/` is excluded from code syncing so a laptop
cannot overwrite the remote collection cache. Conversion belongs in
`slurm/data.slurm`, never on the login node. There are no downloads, package
installation, or internet calls in compute jobs.

## Cluster preparation and submission

The additional pipeline and accounting improvements are isolated in
[`releases/accounting_v3`](releases/accounting_v3/README.md). That release removes
the all-datasets barrier, checkpoints feature generation, supports explicitly
labeled shared-data pilots, and keeps main runs independent with measured setup
and retry costs. Existing root QP workers keep their numerical implementation.

[`releases/accounting_v4`](releases/accounting_v4/README.md) supersedes v3 for
new runs. In v3, `--force` and retries add earlier attempts to `total_seconds`,
and one hard kill makes it NaN permanently; v4 ranks on the clean final attempt.
v4 also counts stopping-check gradients, budgets Kaczmarz by work, gives OSQP
its full solve budget (OSQP 1.1.3 counts setup against `time_limit`), adds a
tolerance ladder for OSQP/SCS, and evaluates winners on held-out seeds.
`plots.corrections` applies the timing and work-count corrections to existing
results without rerunning anything.

All helper commands below run locally, from this folder:

```bash
SKILL=/Users/pranav/.agents/skills/slurm
python3 "$SKILL/scripts/slurm_agent.py" doctor
python3 "$SKILL/scripts/slurm_agent.py" discover
uv run python -m experiments.manifest pilot --output manifests/pilot.csv --seeds 0 \
  --time-limit 600
uv run python -m experiments.manifest tune --output manifests/tune.csv
python3 "$SKILL/scripts/slurm_agent.py" sync up
```

`submit` first uses `sbatch --test-only`; it does not submit without `--yes`.
The random defaults require no data conversion. For SuiteSparse only, validate
`submit slurm/data.slurm --cpus 1 --mem 8G --time 00:30:00`, repeat with `--yes`
after approval, and wait for successful conversion. Validate and submit **one** pilot:

```bash
python3 "$SKILL/scripts/slurm_agent.py" recommend --cpus 1 --mem 16G --time 00:30:00
python3 "$SKILL/scripts/slurm_agent.py" submit slurm/job_array.slurm --array 0 \
  --cpus 1 --mem 16G --time 00:30:00 \
  --export MANIFEST=manifests/pilot.csv --export EXPERIMENT=pilot_v1
# Repeat with --yes only after approval, then:
python3 "$SKILL/scripts/slurm_agent.py" watch JOBID --max-minutes 10
python3 "$SKILL/scripts/slurm_agent.py" stats JOBID_0
```

Pilot representative QP, real-data, Nyström, Kaczmarz, and direct-factorization
tasks individually. `--families` and `--solvers` generate focused manifests;
for example `pilot --families qp --solvers osqp --seeds 0 ...` isolates the
factorization pilot. Choose memory and time from completed pilots; 16 GB Slurm /
12 GB worker are provisional ceilings, not measured recommendations.
If reducing Slurm memory, also set `--export WORKER_MEMORY_GB=<smaller limit>`.
Time requests must cover `time_limit + 600 s` setup allowance plus shutdown.

The full tuning manifest is split into parts of at most 500 rows. Check the
live MaxArraySize, and use `--shard-size` if smaller. Submit one shard at a time
with at most four active tasks; avoid submitting many shards simultaneously
(their concurrency caps add). Never submit the full printed grid spec when
it exceeds the discovered array limit. Example after pilots:

```bash
python3 "$SKILL/scripts/slurm_agent.py" submit slurm/job_array.slurm \
  --array 0-499%4 --cpus 1 --mem 16G --time 01:00:00 \
  --export MANIFEST=manifests/tune.part000.csv --export EXPERIMENT=tune_v1
```

After every tuning shard completes, sync results and generate the next phase:

```bash
python3 "$SKILL/scripts/slurm_agent.py" sync down
uv run python -m experiments.manifest steps --from-results results/tune_v1 \
  --from-manifest manifests/tune.csv \
  --output manifests/steps.csv
python3 "$SKILL/scripts/slurm_agent.py" sync up
# Pilot a steps configuration, then submit its shards with a cap of four.
uv run python -m plots.report --results-dir results/tune_v1 \
  --output results/tune_v1/figures
```

Retain manifests unchanged for the run; regenerate into a new filename when
changing budgets or source. `--force` explicitly reruns a task from scratch.
The skill marker `slurm/.project-id` identifies this project; never override a
different remote marker. No jobs are automatically launched by preparation.

### Batched production arrays

`slurm/batch_array.slurm` runs disjoint slices of a frozen manifest serially.
The original 5,697-task mixed QP grid is archived. The user authorized separating
Kaczmarz screening and raising the fast-lane concurrency to eight. Generate the
new versioned plan with:

```bash
uv run python -m experiments.plan \
  --from-manifest manifests/qp_full_tune_20261030.csv \
  --output-dir manifests/speed_v2
```

This emits 2,187 non-Kaczmarz tasks with all three initialization modes and
three seeds, plus 30 screening tasks: randomized/cyclic Kaczmarz, all five block
sizes, all three initialization modes, side 40, seed 0, sigma 0.2, stride 1,
and a 120 s solve/inner budget with 100,000 total updates. Screen with one
concurrent task. After screening finishes, `--expand-from RESULTS` emits full
grids only for block/method pairs with at least one converged screening arm.
It requires every screening result, including censored runs. Do not claim that
a failure on this screen proves failure for other penalties or strides.

For the fast lane, `BATCH_SIZE=8` and array `0-273%8` cover all 2,187 tasks.
Every configuration keeps its own supervised worker, time/memory limits,
checkpoint, and result CSV. A failed configuration does not suppress the rest
of its batch; the batch exits unsuccessfully if any worker fails. Resubmitting
the same batch skips existing terminal results and resumes interrupted tasks.
The 5.5-hour fast-lane job limit covers eight worst-case worker deadlines of
1,800 + 600 s, 10 s shutdown each, and launch overhead. It grants no configuration
extra solver time. Caps are eight fast QP tasks, one Kaczmarz screen task,
two preparation/regression tasks, and one profiling task. Dependencies keep
profiling jobs sequential. Inspect full-data pilots before regression arrays.
Stepsize phases depend on complete tuning results. The staged warm-only tuning
proposal was not selected: all three initialization modes remain in the grid.

### Versioned execution optimizations and CPU profiling

New manifests include `implementation=speed_v2` and `blas_threads`. Keep their
results separate from historical timings; the old CSVs remain unchanged. Each
run constructs its inner solver once, calibrates it, restores its RNG/cache to
the post-setup state, and reuses it for optimization. Setup time is charged
once in `solver_setup_seconds`; `tuning_seconds` contains calibration without
that setup. No setup is shared across independent runs. Nyström Hessian
products are batched in at most 32 columns. Kaczmarz applies the same block
projection on the block's sparse coordinate support. Metrics reuse supplied
gradients and cached LASSO target scales; the checking stride and common
terminal tolerance are unchanged.

Dense regression supports `--dense-storage resident_dense` in the manifest CLI.
The source remains the same checksummed float32 NPY; workers load bounded row
blocks into a read-only float64 array. This avoids repeated reads and implicit
full-matrix precision conversions. Transpose/normalized views share that array.
`--max-resident-gb` caps the float64 payload before allocation (default 32 GiB);
Slurm and worker limits must additionally cover all solver workspaces. The
largest current payload is about 29 GB. A provisional resident pilot requests
64 GiB Slurm / 56 GiB worker; measure it before a production array. Cold loading
and conversion remain in `generation_seconds` and `total_seconds`. The streamed
path remains available and is the reference for storage comparisons.

`slurm/performance.slurm` runs supervised profiles with independent finite wall
and memory limits. Run one core first and inspect its stats, then repeat with
2/4/8 allocated cores. The script sets `EXPERIMENT_THREADS` from the allocation;
the worker enforces it with `threadpoolctl`. Ordinary production jobs default
to one thread. The profile records repeated solve or gradient timings, cold
loading time, CPU seconds/efficiency, CPU model, BLAS libraries, source hashes,
and peak RSS. Compare streamed and resident full-data profiles as well as QP
Nyström solves. Gradient-only results do not establish end-to-end solver wins.

```bash
uv run python -m plots.scaling --results-dir results/legacy/scaling_speed_v2 \
  --output results/legacy/scaling_speed_v2/summary.csv
```

Recommendations require matched CPU models, at least two successful repeats,
a one-core reference, at least 1.2x speedup, and at least 90% CPU efficiency.
The report does not automatically allocate extra cores. Set both the hashed
`blas_threads` configuration and matching Slurm CPU/`EXPERIMENT_THREADS` values
when promoting a measured profile. Sparse iterations and feature preparation
remain serial unless a representative profile justifies otherwise.

## Accounting, stopping, and outputs

Every optimization task requires finite positive budgets. Defaults are 1,800 s
of solve time, 1,000,000 total inner updates, and 50,000 outer iterations.
Each individual inner solve also stops after 100,000 updates or 300 s (including
its initialization), with statuses `inner_iteration_limit` and
`inner_time_limit`. Override these with `max_inner_per_outer` and
`inner_time_limit` in the task configuration. Apply the same budgets across
compared methods and initialization arms; small-block Kaczmarz receives no
extra time. Resolved budgets are recorded in the result CSV. These are censored,
nonconverged runs, not successful fast solves.

The parent independently terminates the worker after `time_limit + setup_budget`
(600 s setup allowance by default), then forcibly kills it after a 10 s grace
period if necessary and records `worker_timeout`. This covers loading, tuning,
setup, native solver calls, and metrics even when they cannot check a deadline.
Slurm's finite wall-time limit is the final hard stop for optimization and
feature-preparation jobs. A submission must leave room for the parent deadline
and shutdown. **Running jobs do not keep the code they started with**: batch
arrays start a new interpreter for every task, and pending jobs source
`slurm/common.sh` and import their release when they start. Never sync edits to
a source tree that queued or running jobs use; add a new release directory and
confirm with `sync up --dry-run` that only new paths would transfer.

Each array task owns `results/<experiment>/<run_id>/result.csv`, `history.csv`,
and a periodic atomic O(n) checkpoint. Results record requested and resolved
parameters, setup, tuning/calibration, solve and total time, peak process RSS,
versions, source hashes, data provenance, Slurm IDs, and explicit termination
status. `total_seconds` includes instance loading/construction, tuning, inner
setup, metrics, and iteration work. Nyström sketches are built once per run,
not at each outer iteration. Independent cold runs do not amortize setup across
the parameter sweep. Calibration uses its own setup, which is charged too.
The main comparison is setup-inclusive wall time. Inner iterations across GD,
Krylov methods, and block projections are not equivalent units; report row
updates and block factorizations alongside their respective work counts.

All methods start from zero and report the same original-problem normalized
stationarity and feasibility residuals. LASSO uses its soft-threshold gradient
mapping, logistic uses its gradient norm, and QP uses its projected-gradient
mapping and equality residual. A backend's native "solved" flag is insufficient:
the common tolerance is checked independently. Native tolerances default to
one tenth of the common target to reduce differences in stopping conventions;
the actual native tolerance is recorded. OSQP/SCS may still show
`tolerance_not_met` at their native tolerances; tighten their requested tolerance
before claiming accuracy-equivalent wins. The PDHG baseline is our reference
Chambolle-Pock/Condat implementation, not an external PDLP package. OSQP uses
its default direct backend; SCS explicitly requests its CPU indirect backend,
with compatibility for both current and older SCS settings APIs.

The relative inner test and terminal graph test follow the old studies; the
common residual additionally certifies reported convergence. Budget exhaustion,
timeout, error, interruption and not-applicable statuses remain visible.
Only converged runs contribute runtime summaries; final residual charts include
unfinished runs and labels show converged/total counts. Checkpoints are saved
at outer boundaries at most once per minute. Interrupted tasks resume from the
last valid checkpoint (up to a minute of recent work may repeat); external
baselines restart from zero. Restart overhead is not a clean timing experiment:
rerun interrupted tasks with `--force` for publication timing comparisons.

Plots read CSVs only and write PNG/PDF and aggregate CSVs. They reuse the
Okabe-Ito palette, distinct markers, vermillion recommendations, and black
baselines, paginated to avoid exceeding the six validated sweep colors.
Figures contain measured results only; smoke figures are not benchmark results.

Cluster conduct follows [Della](https://researchcomputing.princeton.edu/systems/della),
[login-node rules](https://researchcomputing.princeton.edu/get-started/dos-and-donts),
[resource courtesies](https://researchcomputing.princeton.edu/get-started/mistakes-avoid),
and [storage guidance](https://researchcomputing.princeton.edu/support/knowledge-base/data-storage):
one core and no GPU for serial CPU code, no computation on login nodes, no
explicit QoS, capped arrays after pilots, polling no faster than 60 seconds,
active output on scratch, and completed-results backups off scratch.
