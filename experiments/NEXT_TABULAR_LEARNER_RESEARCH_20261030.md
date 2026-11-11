# Research: one fixed ModernNCA learner

Prepared 30 October 2026 UTC. Research only: no actual feature/target arrays,
model fitting, possession scoring, weights, package installation, runtime
change, protocol freeze, candidate build or promotion. Author text/source was
retrieved into memory; the only new repository artifact is this note.

## Decision

**Go for a separately reviewed, invented-data ModernNCA feasibility gate;
no-go for native fitting until it passes.** This is a plausible local learner,
not an established improvement. Keep the ongoing R61 encoder continuation and
TabPFN MPS work independent. Do not launch another model while they own the
machine's resource budget.

Investigate only ModernNCA with the exact official default configuration below.
Its learned metric and label-weighted retrieval are meaningfully different
from the failed fixed CatBoost, ExtraTrees, ownership stack, probability pools,
ordinary small MLP and pretrained TabICL interfaces. No ModernNCA trial appears
in the inspected research reports/history. Its temporal-shift weakness is a
material reason to keep this a bounded candidate, rather than predict a gain.

Selected8 remains .741176/.685714/.670270. The comparable buffered development
record is R37's **.7542857142857143**, which a new candidate must strictly exceed
while satisfying all existing guards. A higher frame accuracy, a tie with R37,
or beating only the rejected R54 TabICL result would not meet that objective.
[Current checkpoint](CURRENT_RESEARCH_CHECKPOINT.md),
[R37 evidence](PLAYER_REPRESENTATION_REPORT.md),
[failed learner/fusion history](WORK_HISTORY.md).

## Published evidence and its scope

Read the final ICLR 2025 paper, especially sections 3–6 and appendices A.1–A.4,
B.2, B.3 and B.5. ModernNCA learns a nonlinear embedding and predicts by a soft
weighted average of training labels. It samples neighbors during training and
uses all training neighbors at inference. Its final Euclidean distance is
unsquared. The 300-dataset study uses random 64/16/20 splits, 100 tuning trials
and 15 final seeds; it reports strong classification ranks against boosted
trees. These are accuracy/RMSE results, not interval spellF1 or default-only
sports results. Most timing experiments use a Tesla V100; table 9's 87.5-second
training and 5.36-GB mean memory are not local CPU guarantees. Appendix B.5
explicitly identifies temporal distribution shift as a limitation because the
neighbor search does not model implicit temporal relationships. The reported
high-dimensional limitation concerns d much larger than N.
[Final conference paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/ecd294194d5155432aab533f7ce0f6fe-Paper-Conference.pdf).

My task-specific inference: an ownership-dependent metric could join useful
local configurations that axis-aligned tree leaves fragment, and a locally
fitted representation does not inherit a tabular foundation model's synthetic
prior. Conversely, long correlated possessions can make nearby training labels
overconfident and leave held-out temporal blocks without representative
neighbors. Thousands of frames do not provide thousands of independent
possessions. Neither mechanism predicts which outcome wins here. Do not add
time/frame identifiers, change the purged folds, or use random validation to
make the learner look stronger.

## Exact author source and defaults

The paper names LAMDA-Tabular/TALENT as its implementation. The old
qile2000/LAMDA-TALENT URL redirects there. Resolved `main` once using the
first-party GitHub API to commit
`39b54024902955e51063a34bce17d514cd68ff42`; links below bind that revision.
This is the current author implementation, not a claim that this commit is the
original conference release.

| Setting | Official value |
|---|---|
| Numeric embedding | PLR-lite; 77 frequencies, frequency scale .04431360576139521, embedding width 34 |
| Metric dimension | 128 |
| Post-encoder blocks | **0** |
| Declared block width/dropout | 512/.1; inactive with zero blocks |
| Distance temperature | 1 |
| Training neighbor sampling fraction | .5 |
| Optimizer | AdamW, learning rate .01, weight decay .0002 |
| General training defaults | batch 1024; maximum 200 epochs; standard scaling; float64 |
| Author checkpoint selection | best validation result; stop after more than 20 consecutive non-improving epochs |

These are explicit repository defaults, not the paper's per-dataset tuned
configurations. Retaining zero blocks is intentional: PLR supplies nonlinearity;
silently inserting the paper's illustrative MLP blocks would change the chosen
recipe. [Model defaults](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/TALENT/configs/default/modernNCA.json),
[general defaults](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/TALENT/configs/deep_configs.json),
[precision argument](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/utils.py),
[optimizer/training/checkpoint implementation](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/methods/modernNCA.py).

The training wrapper removes every current mini-batch index from the external
candidate pool. The model then uniformly samples that remaining pool, appends
the mini-batch, masks its diagonal, computes `cdist(p=2)`, softmaxes negative
distances, and aggregates one-hot labels. Classification returns
`log(probability + 1e-7)` for NLL. Preserve all of these details, especially
external-pool exclusion before sampling; copying only the model and supplying
all training rows would introduce duplicate self-neighbors. Inference supplies
only the outer training candidates and no query labels.
[Author model](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/models/modernNCA.py),
[author batching/candidate construction](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/methods/modernNCA.py).

## Proposed fixed integration, not a frozen plan

Use the previously defined R54 conditional-live interface so that this proposal
changes the learner rather than also inventing a new decoder or target. This
retains the already tested factorization; its earlier CatBoost/TabICL failures
remain negative evidence. Use the immutable original 102 plus R37 64 features,
not R61's new encoders selected after its outcomes.
[R54 interface](TABICL_NATIVE_CONTEXT_PLAN.json),
[factorization rationale/limits](FACTORIZED_LATENT_RESEARCH_20261027.md).

1. Keep all 13 existing outer purged temporal folds and all 7,500 graded query
   frames. Retain the 250-frame exclusion and complete 79-frame latent
   footprint. The learner's retrieval pool must be a subset of that fold's
   permitted training rows; it must never include outer query rows, labels,
   held-out neighbors or full-match statistics.
2. Reuse R54's deterministic 5-Hz training observation schedule, own-training
   H/A eligibility and paired team augmentation, with its 3,000 augmented-row
   bound. This avoids a new context-stride/cap search. Benchmark 2,500 and
   6,000 invented **augmented rows** as lower/upper engineering envelopes;
   they are not a demand to inflate or truncate actual folds. The task's
   2,500–6,000 original frame counts must not silently be interpreted as the
   same number after doubling. Save original and augmented counts separately.
3. Use train-only mean/empty-zero imputation and standard scaling, with
   constant-column scale one; keep all 166 columns in order. Mean imputation
   follows the author default, while empty-zero and strict schema retention
   are explicit project choices. Fit preprocessing on the paired training
   pool only; freeze it for every query. Assert paired-column swap semantics.
   The author preprocessing fits numerical statistics to training data and
   supports all-empty zero replacement.
   [Official preprocessing](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/lib/data.py).
4. Use the default model/optimizer/batch/float64 settings above, but train
   exactly **200 complete epochs and use the final checkpoint**. Disabling
   early stopping/best-validation selection is an explicit project deviation,
   chosen prospectively to avoid exposing any outer labels to checkpoint
   selection or introducing another nested temporal tuner. It can overfit;
   do not relabel it as the paper's tuned protocol. No learning-rate schedule,
   class weights, cosine-distance arm, temperature fitting, numeric embedding
   variants, feature selection, shorter best-looking training run or ensemble.
5. Convert the two log outputs to normalized probabilities with softmax
   (accounting for the author's epsilon). Use H=0/A=1, predict both original
   and swapped queries, exchange H/A columns back, and average. Use fixed
   512-row query calls with the existing last-row padding convention; all
   queries remain eligible regardless of their unknown true class. Preserve
   original slot0 D mass via `qH=(1-d0)*h`, `qA=(1-d0)*(1-h)`, `qD=d0`.
   Unsupported latent rows use the full original vector.
6. Replace only slot0, leave the other selected probability inputs and complete
   decoder exact, and apply the existing final `live_overlay`. Raw D mass alone
   does not guarantee decoded D identity; the overlay must preserve every
   selected8 D/X decision. Keep selected8 and R37 as named historical
   comparators. This is one new candidate, with no post-score fallback arm.

## Local runtime, dependencies, license and feasibility

No pretrained checkpoint or authenticated service is required. The metric
model and PLR helper use Torch, NumPy and standard-library code; they do not
need FAISS or an approximate-neighbor index. PLR-lite applies per-feature
sine/cosine features, a shared linear map and ReLU before flattening.
[Exact helper/import closure](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/lib/tabr/utils.py).

The full TALENT package is broader: its current declared minimum Torch is 2.8,
and includes Optuna, category_encoders, pandas/scikit-learn and other methods'
dependencies. Do not install it into an existing frozen environment or claim
that SportsTorch2.5 satisfies that declaration.
[Pinned package requirements](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/pyproject.toml).

Read-only distribution metadata inspection found: research .venv has no Torch
or TALENT; SportsTorch has Torch2.5.0/NumPy2.1.2; both the original and separate
loader-repair TabPFN dependency targets advertise Torch2.15.0/NumPy2.1.2.
Neither advertises TALENT/category_encoders/Optuna. These are metadata facts,
not a new import or compatibility test. A later minimal standalone adapter
could retain the exact model and helper in a new directory, load only that
source closure, and reuse the already isolated **repaired Torch2.15 CPU**
runtime read-only. Keep preprocessing/controller work in the research runtime
with explicit array handoff; do not mix their site-packages. This source-only
adapter still needs review, an import smoke test and the invented gate; it is
not a validated installed TALENT package. No new loader repair is proposed.

Both TALENT and the upstream TabR helper carry MIT-style notices; preserve
both copyright/permission notices when retaining substantial source. No
model-weight license or terms-acceptance action is involved.
[TALENT notice](https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/39b54024902955e51063a34bce17d514cd68ff42/LICENSE),
[upstream TabR notice](https://raw.githubusercontent.com/yandex-research/tabular-dl-tabr/main/LICENSE).

The following are arithmetic estimates, not measurements. With 166 columns,
PLR expands each row to 166×154 sine/cosine values before reducing to
166×34=5,644 values and then a 128-dimensional metric. The default model has
740,612 trainable scalars, about 5.65MiB at float64. A 6,000-row expanded
sine/cosine tensor alone is about **1.14GiB**, before temporary tensors,
backpropagation, optimizer or framework overhead. A 512×6,000 distance matrix
is only about23.4MiB. Therefore the small final metric does not establish a
6GiB peak. Re-encoding candidates on every training step and query call can
dominate CPU cost; 200 epochs may exceed existing fit limits even at the
3,000-row native bound. Do not assume GPU paper timings apply.

Keep CPU, one Torch/BLAS thread, one worker, fixed random streams and
deterministic-algorithm enforcement. Author code uses random permutations for
both mini-batches and neighbor subsets. Seed all involved generators; preserve
the exact epoch/batch schedule and candidate row order. PyTorch explicitly
does not guarantee reproducibility across releases/platforms, so a seed alone
is insufficient evidence.
[Official reproducibility guidance](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.0/docs/source/notes/randomness.rst).

## Admission conditions for a later experiment

A later separately frozen invented-data gate should run the complete final-200
epoch recipe at both stated shapes, including train-only preprocessing,
paired augmentation, partial last training batch and fixed query calls.
Require finite losses/probabilities, explicit H/A order, normalization, swap
equivariance, correct self exclusion, first-query repeat, frozen model state,
saved-checkpoint reload and an independent fresh-process **refit** reproducing
weights and canonical probabilities byte-for-byte. Query-order/shape probes
are diagnostic and may not choose a preferred schedule after seeing results.
Record actual supervised peak RSS, full fit time and all-query time. Require
the existing 6GiB/300-second operation bounds unless a different prospective
bound is explicitly reviewed before execution; do not waive a failure.

Only after that gate can an authorized native plan bind exact source/runtime,
input/cache/checkpoint hashes, context indices and all selected artifacts.
Freeze the strict historical-record test plus the unchanged selected8 primary,
mean-fold, macroF1, short-IoU, matched/spurious/missed, D-recall and falseD
guards. A passing seed42 screen alone permits seeds43/44; all nine paired
checks, exact independent replay, public/full parity, two identical
199,319-row CSV builds, validator and independent review remain necessary
before promotion. Failed runs and old substantial-gain criteria remain intact.
This note authorizes none of those actions.

There are only 17 adaptively reused public possessions. No source, repeated
seed, synthetic feasibility pass or higher development score establishes
private/full-match improvement or an optimum. No private/manual/sealed data,
external messages, commit, upload, submission or publishing is part of this work.

## Source identity evidence

Hashes below were computed from small source bodies in memory, not from an
installed package or downloaded model. They identify this research snapshot.

| Path at pinned TALENT commit | Bytes | SHA256 |
|---|---:|---|
| `TALENT/model/models/modernNCA.py` |4470|`02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f`|
| `TALENT/model/methods/modernNCA.py` |15409|`608c85e94714718fd5690a8096890b6232f2da94d0c94d5489734278df112d3e`|
| `TALENT/model/lib/tabr/utils.py` |12259|`46f26474f559cdda4eda92c5d28d504905c3180ad0ead1bc3f296af72c74230b`|
| `TALENT/configs/default/modernNCA.json` |585|`ea4dc58f92ea9c5d0be7e1c35b38d28d247b6d3c589da10e16c2dcd951a74700`|
| `TALENT/configs/deep_configs.json` |529|`fcaafb28503aa72d8170c87851c582704acfac1fbbeed9da8b06e53949d9aad6`|
| `LICENSE` |1101|`93cf90600c386cb5719779090fbc37080d5833eeec8d1f66cc370ba48af2d422`|
| `pyproject.toml` |705|`5a8b2fcf178bb30670a162fab2fd95fc77d40a1c329c5371df1f4a0a232bb544`|
