# NysADMM dataset preparation and source-code status

All commands run from `cluster_experiments`. These are dataset-inspired
experiments with our existing QP/LASSO/ridge-logistic solvers, inexactness and
stride/stepsize sweeps, and warm/fixed/Gaussian ablations. They are not a claim
to reproduce the original NysADMM timings or its L1-logistic algorithm.

## Verified data sources

The five requested datasets were successfully downloaded from the official
[LIBSVM collection](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
on 2026-10-30. Downloads are sequential, cached, checksummed, and have no
automatic retries. No SuiteSparse service is involved.

| Dataset | Input rows | Input features | Full random features from NysADMM Table 2 | Source files |
| --- | ---: | ---: | ---: | --- |
| cod-rna | 59,535 | 8 | 60,000 | `binary/cod-rna` |
| realsim | 72,309 | 20,958 | 50,000 | `binary/real-sim.bz2` |
| rcv1 | 20,242 | 47,236 | 30,000 | `binary/rcv1_train.binary.bz2` |
| smallNORB | 24,300 | 2,048 | 30,000 | `multiclass/smallNORB-32x32.mat` |
| CIFAR-10 | 60,000 | 3,072 | 60,000 | `multiclass/cifar10.mat` and `cifar10.t.mat` |

The output feature counts come from
[NysADMM, Section 5, Table 2](https://arxiv.org/pdf/2202.11599).
The smallNORB source is LIBSVM's downsampled stereo-image variant. The paper
does not establish that it used this same raw image representation. CIFAR-10
combines its 50,000 training and 10,000 test images to match the paper's 60,000
optimization rows. **Do not treat those combined data as a held-out prediction
evaluation.** The input dimension in the paper is 3,073; this source has 3,072
pixel features, and we do not invent an undocumented intercept/preprocessing step.

LIBSVM's raw files, source URLs, exact SHA256 checksums, and byte counts are in
`data/nysadmm/raw/` and adjacent `.source.csv` files. Image MAT headers were
inspected: the design is `Z` and labels are `y`. Binary text files are loaded
with scikit-learn's SVMlight reader. All raw dimensions and finite values are
checked. We use LIBSVM's hosted versions here; no OpenML/UCI API query is needed
for these five sources. Other local UCI/OpenML exports still work through the
general matrix adapters.

## Feature map and labels

The implemented [Rahimi-Recht random Fourier map](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
approximates `k(x,y)=exp(-gamma ||x-y||²)`:

```
z_j(x) = sqrt(2/D) cos(w_j.T x + b_j)
w_j ~ N(0, 2 gamma I),  b_j ~ Uniform(0, 2 pi)
```

Defaults are row L2 normalization of raw inputs, gamma 1, feature seed 0, and
float32 output. These are explicit experimental choices, **not verified original
NysADMM hyperparameters**. Set `--preprocessing row_l2|none`, `--gamma`, `--seed`,
and `--dtype float32|float64` to change them. Each feature has its own RNG keyed
by `(feature_seed, feature_index)`, so changing computation tile sizes preserves
the map up to floating-point rounding. No post-map column normalization is done
by default; that would change the kernel approximation.

Native labels are preserved in a separate NPY file. LASSO treats them as numeric
responses and defaults to lambda 1, matching the paper's stated LASSO setting.
Our logistic model remains the mean log loss plus ridge regularization, with
vanilla GD as its inner solver. Class 1 versus all other classes is the default
binary mapping (`--positive-class` changes it), including for CIFAR-10 and
smallNORB. The original paper's multiclass-to-binary mapping is unverified.
The mapping, source hashes, subset selection, feature parameters, precision,
preparation time, and label hashes are recorded in the generated source spec.
Response seeds repeat solver randomness on the fixed native data; they do not
generate new labels. Native labels cannot be attached to transposed designs.
The full prescribed designs already include tall (realsim) and wide (smallNORB,
rcv1, cod-rna) cases without transposition.

## Small local pilots

```bash
uv sync --frozen
uv run python -m experiments.benchmark_data download \
  --datasets cod-rna rcv1 realsim smallnorb cifar10
uv run python -m experiments.benchmark_data prepare \
  --datasets cod-rna rcv1 realsim smallnorb cifar10 \
  --max-rows 128 --features 64 --allow-local \
  --output manifests/nysadmm_all_pilot_sources.json
uv run python -m experiments.manifest smoke \
  --dataset-spec manifests/nysadmm_all_pilot_sources.json \
  --families lasso logistic --seeds 0 --solvers cg gd scs pdhg osqp \
  --inner-initializations warm fixed gaussian \
  --time-limit 10 --max-inner 100000 --tolerance 0.0001 \
  --output manifests/nysadmm_all_smoke.csv
uv run python -m experiments.run --manifest manifests/nysadmm_all_smoke.csv \
  --task-id 0 --results-dir results/nysadmm_smoke
```

Subsets include at least one example of each class, with remaining rows sampled
uniformly without replacement using a recorded seed. Small pilots are an
ingestion/numerical check, not the paper-sized experiments. Explicit local
preparation is limited to at most 512 rows and 512 features; production feature
preparation requires a Slurm allocation.

## Full feature preparation on Della

First run the environment update from your terminal after code sync:

```bash
ssh della-gpu 'cd /scratch/gpfs/BSTELLATO/pranav/rand_nla_fom && bash slurm/env_setup.sh'
```

New preparation needs the locked scikit-learn dependency. Raw files must be
staged before compute jobs; no job downloads or installs packages. Downloading
missing files on the login node is allowed; feature generation is computation
and must use `slurm/features.slurm`. Validate a small representative preparation
job through the slurm skill first, then obtain approval before actual submission:

```bash
SKILL=/Users/pranav/.agents/skills/slurm
python3 "$SKILL/scripts/slurm_agent.py" submit slurm/features.slurm \
  --cpus 1 --mem 4G --time 00:10:00 \
  --export DATASET=cod-rna --export FEATURE_COUNT=64 --export MAX_ROWS=128 \
  --export OUTPUT_SPEC=manifests/della_cod_rna_pilot_sources.json
```

After its stats, size full preparation appropriately. Omit `MAX_ROWS` and use
the full counts in the table, for example `FEATURE_COUNT=60000` for cod-rna.
Run one dataset per preparation job. `RBF_GAMMA`, `FEATURE_SEED`, `POSITIVE_CLASS`,
`INPUT_SCALING`, and `FEATURE_DTYPE` configure preparation explicitly. Build
experiment manifests on the cluster using the generated source spec and the
ordinary `tune` phase. The preparation job writes JSON specs to `manifests/`;
those specs are inputs, not result measurements.

Dense maps are written by column tiles into a Fortran-order intermediate,
then converted once into the C-order NPY consumed by solvers. This avoids
repeatedly flushing scattered pages across the full output. Temporary storage
can reach approximately twice the final payload. Float32 payload sizes
are approximately 14.4 GB (CIFAR-10), 2.916 GB (smallNORB), 14.462 GB (realsim),
2.429 GB (rcv1), and 14.288 GB (cod-rna), excluding headers. Float64 doubles
these sizes. Check scratch quota and copy completed artifacts off scratch.
Feature preparation maps its output file and flushes each feature tile. Solver
workers can read bounded row blocks without mapping the full file or explicitly
load a float64 resident design (`--dense-storage resident_dense` in the manifest
CLI). Resident payload limits and allocation overhead are documented in the
README. Loading/conversion is charged to each independent run. Benchmark both
storage modes before extrapolating throughput.

The streamed path supports GD, CG, GMRES/PCG where exposed, and reference PDHG.
External OSQP/SCS interfaces require explicit matrices. Conversion is guarded by
`max_explicit_matrix_gb` (default 2 GB, conservative workspace estimate); larger
requests report `representation_memory_limit` instead of densifying silently or
claiming a solver failure. Smaller pilots exercise those external baselines.
Dataset preparation is shared across configurations: `total_seconds` includes
data loading/validation and solver setup/iteration work; `data_preprocessing_seconds`
records the one-time feature preparation, and `end_to_end_seconds` additionally
charges it in full. Report both when comparing cold starts.

## Original source search

On 2026-10-30 GitHub's public API returned 404 for
`https://github.com/udellgroup/NysADMM`. Public repository searches including
forks found [GurevichPE/nysadmm](https://github.com/GurevichPE/nysadmm) and its
fork, but their README identifies a separate NLA project with different
contributors. They are not authenticated mirrors of the authors' experiments.
Searching public READMEs for the original repository URL returned no matches.
Archive lookups were not accessible in this environment, so this is not proof
that no archived copy exists.

Related publicly hosted implementations remain available:

- [GeNIADMM.jl](https://github.com/tjdiamandis/GeNIADMM.jl), the authors' related
  GeNI-ADMM experiment code, with Nyström ADMM comparisons.
- [udellgroup/rlaopt](https://github.com/udellgroup/rlaopt), the group's newer
  optimization implementation, not the 2022 experimental source snapshot.

TODO: authenticate any recovered original snapshot before adopting its kernel
bandwidths, label mapping, input scaling, or random seeds as reproduction settings.
