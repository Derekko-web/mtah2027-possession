# Research: temporal block exclusion inside the NCA training objective

Prepared 1 November 2026 UTC. Primary-source research and source inspection
only. No scientific imports, native features/targets/arrays, model calls,
scores, packages, runtime changes, protocol freeze or old source edits. This
new note is the sole repository artifact produced by this research.

## Decision and scope

**Conditional GO for one separately reviewed invented-data feasibility gate;
NO-GO for native fitting from this note.** The hypothesis is sufficiently
different to investigate once: replace exact-row leave-one-out likelihood with
fixed **250-native-frame block-excluded soft-neighbor likelihood during
training**. Keep R68 closed. This would be a new project objective using the
ModernNCA architecture, not a repeat of the closed author-default experiment
and not an author-validated temporal ModernNCA method. Root retains the decision
whether to prepare it after the current GP work; no concurrent model work is
authorized here.

The reason for this narrow GO is structural: the existing purged outer query
cannot retrieve a training center within 250 frames, whereas a training anchor
currently can retrieve many temporally adjacent own-training examples. A
metadata-only mask can make that particular training retrieval constraint
closer to the fixed outer query constraint without adding a feature, label,
validation statistic, tuner or decoder. This is an inference from the objective
and project split, **not evidence that temporal memorization caused R68's
failure or that masking will improve possession quality**.

The old result remains negative evidence: R68 completed all 13 fixed final-200
fits but scored .708571/.491429/.616216 and lost short-possession matches.
R73 SVM also failed despite improved frame macroF1. No targeted repair for its
lost GT12 or any particular scored interval is proposed. The current stronger
buffered benchmark is strictly above **136/175 = .7771428571428571**, even
though the complete buffered R65 TabPFN diagnostic is itself ineligible. All
selected8 guards, all nine seed42/43/44 protocol pairs, fresh 39-refit replay,
full/public parity, two identical full CSV builds and independent review remain
necessary before promotion. These project facts come from the retained
[current checkpoint](CURRENT_RESEARCH_CHECKPOINT_20261101.md),
[R68 report](MODERNNCA_NATIVE_REPORT.md),
[R73 report](KERNEL_SVM_NATIVE_REPORT.md) and
[R65 buffered diagnostic](TABPFN_NATIVE_BUFFERED_GUARD_DIAGNOSTIC.md).

## What the primary sources support

Goldberger et al.'s original NCA optimizes a stochastic leave-one-out training
classification surrogate: the anchor excludes itself, soft weights select
other labels, and their same-class mass gives its correctness probability.
Section 2 also presents the log-probability alternative and explains that
embedding scale changes effective neighbor count. This supplies the objective
connection, not a temporal-block theorem. Read Section 2, equations 1–7,
pp. 1–3. [Original NCA, NeurIPS 2004](https://papers.nips.cc/paper/2566-neighbourhood-components-analysis.pdf).

ModernNCA uses log soft-neighbor likelihood, unsquared Euclidean distance and
random neighborhood sampling, restoring all training neighbors at inference.
Its sampling study concerns uniform subsets; its results do not establish
temporal exclusion. Appendix B.5 explicitly acknowledges distribution-shift
limitations involving implicit temporal relationships. It does not propose or
validate this mask. Read Sections 3.2–4.2 and 6.2, Appendix B.2 (sampling/loss)
and B.5. [Final ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/ecd294194d5155432aab533f7ce0f6fe-Paper-Conference.pdf).

Roberts et al. show why dependent nearby holdouts can produce optimistic error
estimates and discuss buffered leave-one-out/block validation. They also warn
that blocking can remove predictor conditions, impose extrapolation and
overestimate interpolation errors. Their recommendation depends on the
prediction objective. This supports keeping the existing purged evaluation and
taking support loss seriously; it does not prove a training-mask benefit.
Read pp. 914–917, Box 1 pp. 918–919 and guidance/limitations pp. 924–926.
[Author-hosted Ecography 2017 paper](https://www.biom.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf/at_download/file).

Bergmeir, Hyndman and Koo give a narrower counterexample to blanket claims that
all temporal neighbors must be removed: ordinary CV can work for stationary,
purely autoregressive prediction with uncorrelated errors. They identify
inefficient use of data as a limitation of h-block exclusion. These conditions
have not been established for this ownership classifier. Read abstract and
Sections 1–3, pp. 1–5. [Author-hosted paper](https://robjhyndman.com/papers/cv-wp.pdf).

TISEAN's primary author documentation excludes delay vectors containing an
unknown forecast target and warns that the remaining support may be
insufficient. Its local prediction implementation checks remaining neighbor
support. This is adjacent evidence for explicit exclusions and support checks,
not supervised metric-training evidence. Read the complete
[causality-window documentation](https://www.pks.mpg.de/tisean/TISEAN_2.1/docs/docs_c/causality.html),
[local prediction description](https://www.pks.mpg.de/tisean/TISEAN_2.1/docs/chaospaper/node18.html)
and `zeroth.c` neighbor-search/exclusion branch, lines 239–247.
[Primary code](https://www.pks.mpg.de/tisean/TISEAN_2.1/source_c/zeroth.c).

No primary source inspected establishes that a 250-frame exclusion is an
optimal dependence horizon, that it prevents all same-possession shortcuts, or
that it improves ModernNCA on sports intervals. The value 250 is defensible
here only as the already fixed project outer-purge constraint. It must not
become a grid of radii, a possession-length rule or an outcome-chosen value.

## Exact author behavior and the proposed single deviation

The pinned author method removes **all current batch row indices** from the
external candidate pool before invoking the model. The model uniformly samples
`floor(.5 * external_rows)`, embeds anchors and sampled candidates, then
**prepends** the current mini-batch to the sampled candidates. It masks only the
corresponding self diagonal before softmax and label aggregation; NLL receives
`log(probability + 1e-7)`. Neither author function has frame metadata or a
temporal exclusion. Read complete `ModernNCA.forward`, author model lines
65–116, and `train_epoch`, method lines 203–256.
[Pinned model](https://github.com/LAMDA-Tabular/TALENT/blob/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/models/modernNCA.py),
[pinned training method](https://github.com/LAMDA-Tabular/TALENT/blob/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/methods/modernNCA.py).

The existing project's outer purge is already explicit in
`modernnca_native.context_frames`: it checks every own-training center is more
than 250 frames from outer validation, selects complete +/-79-frame
own-training footprints **before** the `%5 == 0` H/A context join, and caps
paired rows at 3000. It does **not** block neighbors between training anchors.
`context_record` orders all original rows followed by their swapped copies,
with complemented H/A labels. Read these source functions and the retained
[R68 plan](MODERNNCA_NATIVE_PLAN.json); no native payload was opened.

The following is the complete mathematical recommendation for a later method,
not an implementation plan or permission to run it:

1. Let `U` contain the exact old paired own-training rows, and let `f[r]` be
   the native integer center of row `r`. Original row `r` and its swapped mate
   have **the same center and frame group**, with complemented labels. Do not
   use augmented row number, 5-Hz position or possession identity as time.
2. Preserve the author shuffled mini-batch `B`, external pool `E = U minus B`,
   and uniform sample `S` of exactly `floor(.5 * len(E))` row indices. Preserve
   its RNG draws, row order and candidate order `C = concatenate(B, S)`.
   Sample rows, not paired groups; do not introduce pair-balanced sampling.
3. For each anchor `i` in `B`, allow candidate `j` in `C` exactly when
   `abs(f[i] - f[j]) > 250`. Equality at 250 is excluded. Apply this **row by
   row** to both the prepended batch and external sampled portion, covering
   both original and swapped views. Existing exact-self exclusion remains
   logically true, now subsumed by the block mask. Do not remove from `E` the
   union of every batch member's temporal window: that would be a different,
   potentially empty, candidate pool.
4. With `z = phi(x)` and unchanged `T = 1`, compute

   `p_i(c) = sum_j a_ij exp(-||z_i-z_j||_2/T) 1[y_j=c] / sum_j a_ij exp(-||z_i-z_j||_2/T)`,

   where `a_ij = 1[abs(f[i]-f[j]) > 250]`. The batch objective is the unchanged
   mean `-log(p_i(y_i) + 1e-7)`. Implementationally, a later adapter could set
   excluded distances to positive infinity before the existing softmax and
   one-hot aggregation. Mask the **denominator and numerator together**;
   removing only positive labels, or zeroing weights without renormalization,
   would not implement this objective.
5. Fail the entire fixed method if any realized anchor has zero eligible
   candidates. There is no defined softmax probability over an empty set.
   No resampling, shorter gap, skipped anchor/batch, copied self-neighbor,
   fallback model or altered batch size may repair it. Also fail a prospective
   support gate if an anchor has no eligible candidate with its own class:
   the epsilon otherwise creates a constant loss with no useful gradient for
   that anchor. This support decision uses only own-training labels and is
   fixed before execution. Do not guarantee support by label-aware sampling.
   Missing opposite-class support can yield a trivial probability one; retain
   the author calculation and record its count without changing the recipe.
6. Frame centers are immutable **training-mask metadata only**. They must
   follow exactly the same `B/E/S` permutations as features and targets, stay
   outside the 166 feature matrix, and never enter imputation, scaling, PLR,
   distances, the decoder or an inference input. No query labels, query
   validation features or full-match statistics may determine the mask or
   candidate selection. The outer split is the unchanged prior boundary.
7. At outer inference, use all permitted own-training paired candidates with
   the original model inference rule and no query labels. Those centers are
   already more than 250 frames from every outer query. The training mask does
   not authorize changing inference context, masking decoded spells or using
   only past candidates. Later full-match behavior still requires its own
   authorized parity and construction process.

Everything else stays fixed: original 102 plus original R37 64 columns;
paired-train-only orbit-tied mean/empty-zero imputation and standard scaling;
PLR-lite 77 frequencies, scale .04431360576139521, width 34; 128-dimensional
metric; zero post-encoder blocks; AdamW .01/.0002; batch 1024; float64 CPU2;
200 complete epochs and final checkpoint; fixed canonical 512 queries;
original/swapped query probability average; slot0 D/live-mass factorization;
unsupported-latent fallback; all other selected inputs, decoder and final live
overlay. These are preserved project choices, including its earlier
final-checkpoint deviation, not a new claim of following the paper's tuned
validation protocol. [Author defaults](https://github.com/LAMDA-Tabular/TALENT/blob/39b54024902955e51063a34bce17d514cd68ff42/TALENT/configs/default/modernNCA.json),
[general defaults](https://github.com/LAMDA-Tabular/TALENT/blob/39b54024902955e51063a34bce17d514cd68ff42/TALENT/configs/deep_configs.json),
[prior integration rationale](NEXT_TABULAR_LEARNER_RESEARCH_20261030.md).

## Why the hypothesis might fail

These are task-specific deductions, not published performance claims:

- Adjacent windows may already be hard or uninformative in the fitted metric.
  R68's loss/distance concentration has not been diagnosed here. Its lower
  interval score alone does not identify overfitting, and a mask cannot fix
  missing ownership information in the fixed 166 columns.
- The mask removes **both classes**, including the same-frame swapped mate
  with the opposite H/A label and useful nearby boundary negatives. It is not
  necessarily a uniformly harder task. Any improvement could come from
  changing confusability or density rather than removing an easy positive.
- With only 17 released possessions, removing nearby positive support can
  force incompatible configurations together, underfit short states or
  extrapolate. A long possession can still provide same-label examples more
  than 250 frames away, so this is not leave-one-possession-out training.
- Full paired context is globally balanced, but row-level sampling and
  row-specific exclusion need not retain both labels for every anchor. Pairing
  gives both labels for each retained frame **in the full pool**, not a
  guarantee about the realized sampled pool. Support must be checked where
  the loss is actually evaluated.
- Soft-neighbor probabilities depend on candidate density. Removing one
  anchor's local group changes its normalizer and effective class support;
  all-neighbor inference retains a different density. Matching the minimum
  temporal distance does not make the training and outer-query distributions
  equal, and does not calibrate probability sharpness.
- A gap does not regularize embedding scale. PLR's trainable frequencies,
  shared linear map and final encoder can still learn a sharp, brittle metric;
  zero post-encoder blocks provide no output normalization. The original NCA
  scale observation makes this a concrete concern. Do not add normalization,
  temperature fitting, weight-decay search or alternate PLR settings as a
  rescue. Read the pinned helper's `PeriodicEmbeddings` and `PLREmbeddings`,
  lines 121–193. [Author helper](https://github.com/LAMDA-Tabular/TALENT/blob/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/lib/tabr/utils.py).
- This proposal follows observed R68 failure and uses adaptively reused public
  development data. A fixed single deviation limits additional flexibility;
  it does not undo research-selection bias. Even a future pass would establish
  a development result under the existing guards, not private accuracy or an
  optimum. No epoch, radius, context, view, seed or decoder may be selected
  from its native outcomes.

## Theoretical feasibility and later evidence boundary

There are no new parameters or dependencies. Sampling and embedding work stay
the same; an explicit mask adds `O(batch_rows * candidate_rows)` comparisons
and storage. At the existing 3000-row bound, a full 1024-row batch has
`1024 + floor((3000-1024)/2) = 2012` candidates. Its boolean mask is about
1.97 MiB; one same-shaped int64 temporary is about 15.72 MiB. Actual temporary
allocation and gradients remain implementation dependent. Since `%5 == 0`
permits at most 101 distinct centers within inclusive +/-250, at most 202
paired rows can be temporally excluded per anchor in the full context; this
does not establish support for small contexts or arbitrary realized samples.
These are arithmetic bounds, not native-data inspection or runtime measures.

The mask does not avoid PLR encoding or the dense distance calculation, and
old R68 technical passes cannot be inherited by changed training code. Any
later admission must actually establish correct candidate/frame alignment,
same-frame mate exclusion, distance-250 exclusion and distance-251 inclusion,
both candidate portions, full/partial batches, finite supported likelihood,
zero-support closure, unchanged inference, swap behavior, reload, immutable
state and exact fresh-process refit. Invented-data tests must exercise both
well-supported temporal groups and explicit impossible support; they must not
choose a native-friendly timeline after looking at labels or scores.

Retain one serial worker on the M1 Pro 16 GB machine, CPU2, <=6 GiB,
fit <=1800 s, queries <=300 s and worker <=2400 s. Later supervision must apply
the current dual calendar/monotonic clock requirements and preserve every
failed run; this note makes no timing guarantee. GP owns any live model budget.
No gate, model source, plan, native run or promotion was created here.

## Source identities and inspection record

TALENT source is pinned to
`39b54024902955e51063a34bce17d514cd68ff42`, the same author revision retained by
the original research. Remote source/PDF bodies were read and hashed in memory
with the project-private head Python in `-B -I -S` standard-library mode.
Blocked-validation papers and TISEAN were independently read by a delegated
primary-research agent; the main researcher also read the linked papers'
relevant text and complete causality/local-prediction documentation. No source
body was installed or copied into a model directory.

| Source | Bytes | SHA256 |
|---|---:|---|
| ModernNCA final ICLR PDF | 890821 | `f54a558cd730500d908c7163d0a7812aaf607bd418c8a4288552bb413ced18ee` |
| Original NCA NeurIPS PDF | 267467 | `13831bbd3e691251dadba09b8da4d5ae456d2b03c2af6422e944b4d5a2fa974d` |
| TALENT `model/models/modernNCA.py` | 4470 | `02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f` |
| TALENT `model/methods/modernNCA.py` | 15409 | `608c85e94714718fd5690a8096890b6232f2da94d0c94d5489734278df112d3e` |
| TALENT `model/lib/tabr/utils.py` | 12259 | `46f26474f559cdda4eda92c5d28d504905c3180ad0ead1bc3f296af72c74230b` |
| TALENT `configs/default/modernNCA.json` | 585 | `ea4dc58f92ea9c5d0be7e1c35b38d28d247b6d3c589da10e16c2dcd951a74700` |
| TALENT `configs/deep_configs.json` | 529 | `fcaafb28503aa72d8170c87851c582704acfac1fbbeed9da8b06e53949d9aad6` |
| Roberts author PDF | 811805 | `77f836a01a52fbca9ab5a6b8b4292121d8d896b04eb66fbf351358fec208c1f4` |
| Bergmeir/Hyndman/Koo author PDF | 237603 | `fc083acb269cb22e7a533572b46b64f5b88831ead50f3232b20c4d5016960dd7` |
| TISEAN causality documentation | 1288 | `60a5a87439b244bee9e0ef44e83b2de2e55bae2c360c8d2edf04f9337a10fa7f` |
| TISEAN `zeroth.c` | 9085 | `eb566da2374895b4e1319b7b319f2d5cfd4bc4298bca0eff3f17496d84e8c883` |

The useful outcome is a bounded, falsifiable new training-objective hypothesis
with explicit limits. There is no claimed score improvement or authorization
to reopen R68.
