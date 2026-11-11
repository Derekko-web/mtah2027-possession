# Research: one purged-calibration RBF ownership learner

Prepared 31 October 2026 UTC. Research only. No scientific package was imported,
native array or label row read, model fitted, probability generated, score run,
runtime altered, plan frozen, checkpoint acquired or selected artifact changed.
The existing isolated head interpreter was used with `-I -S` for standard-library
HTTP/text/hash reads only. This note is the only new repository artifact.

## Decision and distinct hypothesis

**One fixed RBF C-SVC with training-only, purged temporal sigmoid calibration is
a plausible next bounded learner.** First require a complete invented-data
resource/reproduction gate in the existing isolated head runtime. Do not run it
concurrently with the active TabPFN/ModernNCA work. This recommendation establishes
neither a possession improvement nor priority over an already running screen.

This is a margin-regularized nonlinear ownership classifier on the immutable
102 original plus 64 R37 latent features. It changes the objective and hypothesis
class; it does not add another probability pool, decoder, observation source or
pretrained checkpoint. ModernNCA learns a distance representation and averages
neighbor labels; this method instead fits a soft-margin separating function in
a fixed RBF reproducing-kernel space. Its signed support-vector expansion is
not a normalized weighted average of neighboring labels.

No SVC/kernel ownership trial appears in the inspected history or research
sources. This is a statement about the inspected repository, not a claim of
scientific novelty. A new ordinary MLP, another tree learner, TabM-style ensemble,
boundary surrogate or generic temporal smoothing would have a weaker novelty
case here. R30 already trained a five-layer dilated temporal convolutional
network for D/live; calling the next compact temporal network an unexplored
architecture would omit relevant negative evidence.
[History](WORK_HISTORY.md), [R30](TEMPORAL_DEAD_REPORT.md),
[earlier MLP](NEURAL_CONTROL_PLAN.json),
[ModernNCA research](NEXT_TABULAR_LEARNER_RESEARCH_20261030.md).

The inputs already contain ordered time taps, contact history and the bounded
temporal player representation. This is an ownership learner over those existing
temporal summaries, **not a newly trained joint sequence model**. No claim about
learning arbitrary duration or event order follows. Extending the raw temporal
window would change the hypothesis and footprint and is not part of this recipe.

## Primary evidence and its limits

The LIBSVM authors' implementation paper, section 2.1, gives the C-SVC objective:

```
minimize  0.5 * ||w||^2 + C * sum_i xi_i
subject to y_i * (w^T phi(x_i) + b) >= 1 - xi_i, xi_i >= 0.
```

Its dual uses a positive-semidefinite kernel and bounded coefficients; prediction
uses a signed support-vector expansion. Sections 4–5 discuss decomposition,
shrinking and caching. These provide the relevant objective and implementation,
not a theorem that dependent sports frames generalize like independent examples.
[Author paper](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf).

The authors' practical guide recommends scaling and an RBF kernel, but normally
also selects C and gamma by cross-validation. It explicitly does not guarantee
the highest accuracy. Here **C=1 and gamma='scale' are fixed software defaults**, not
the guide's tuned protocol or an author-validated setting for possession. Skipping
a parameter search keeps this a single prospectively defined hypothesis.
[Author guide, sections 1.2–3](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf).

Lin, Lin and Weng's primary note defines the sigmoid probability approximation,
smoothed binary targets and a convex two-parameter likelihood objective. It
also explains numerical problems in the original pseudocode. This supports using
the maintained implementation instead of inventing a margin temperature.
[Primary note, equations 1–2 and sections 2–3](https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf).

Niculescu-Mizil and Caruana, ICML 2005, section 2.1, explain why calibrating on
the same fitted examples biases the mapping, and use separate calibration data.
Their experiments compare probability quality on eight binary datasets, with
random samples and many model settings. Their finding that sigmoid calibration
can help SVM probabilities is not evidence of calibrated ownership probabilities
from 17 dependent possessions, nor of higher interval spellF1.
[Author paper](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf).

Task-specific inference: a smooth margin surface may use the joint latent and
physical geometry differently from axis-aligned splits and neural retrieval.
It could also blur a short possession, treat correlated frames as repeated
evidence, and fail under temporal distribution shift. R37's latent representation
helps aggregate scores while retaining the same three short matches and adding a
spurious buffered possession; that observation motivates a different learner but
does not predict its gain.
[R37](PLAYER_REPRESENTATION_REPORT.md).

## Exact source, default settings and probability semantics

Resolve scikit-learn 1.9.1 to author commit
`866c0f51e7560ef0303cbcc5f159df5382ea9e3f`. The existing isolated head runtime's
`_classes.py`, `_base.py` and `calibration.py` match that commit byte-for-byte
by standard-library read/hash inspection. Distribution metadata reports
scikit-learn 1.9.1; this research did not import it or establish binary health.

Use these explicit SVC settings:

| Setting | Fixed value |
|---|---|
| Kernel / C / gamma | `rbf` / `1.0` / `scale` |
| Shrinking / solver tolerance | `True` / `1e-3` |
| Kernel cache | `200` MB |
| Iterations | `-1`; external 300-second operation limit still applies |
| Class weights | `None`; paired H/A augmentation balances classes exactly |
| Internal probability fit | disabled |
| Decision function / tie behavior | `ovr` / `break_ties=False`; binary interface |
| Random state | outer seed, initially 42 |
| Numeric arrays | C-contiguous float64 |

These are the pertinent author defaults, made explicit. Degree/coef0 are
irrelevant to RBF and remain their defaults. The source computes
`gamma = 1/(n_features * X.var())`, with gamma 1 when variance is zero; after
standardization, constant columns mean the result need not equal exactly 1/166.
Record the resolved gamma for every inner and final fit rather than substituting
a guessed value.
[Pinned SVC](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/svm/_classes.py),
[pinned fit/gamma implementation](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/svm/_base.py).

**Do not use `SVC(probability=True)`.** Its internal five-fold calibration is
not the required temporal/group-aware split and could separate a row from its
team-swapped duplicate. The 1.9 API deprecates that probability parameter and
recommends `CalibratedClassifierCV(..., ensemble=False)` instead. Leave SVC's
1.9.1 probability argument at its `"deprecated"` default, which the pinned
implementation resolves to effective `False`; assert and record that effective
value. An explicit `False` is mathematically equivalent but emits the known
deprecation warning in this release.
[Official SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

Wrap the complete preprocessing/SVC pipeline in
`CalibratedClassifierCV(method='sigmoid', cv=explicit_inner_splits,
ensemble=False, n_jobs=1)`. This builds out-of-fold margins for calibration,
then trains one final base estimator on all permitted training rows. Prediction
uses **one classifier and one calibrator**, not an average of inner models.
Explicit `ensemble=False` matters: the current default `auto` would otherwise
produce an ensemble. The API accepts explicit train/test index iterables.
[Official calibration API](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).

The pinned sigmoid implementation uses smoothed targets, stable binomial loss
and L-BFGS-B with `gtol=1e-6`, `ftol=64*float64_epsilon`; no held-out checkpoint
selection occurs. Preserve it unchanged. Its fitted two-parameter map is not
a post-score temperature adjustment. Purging removes nearby label leakage; it
does **not** prove independent calibration samples or actual probability
calibration under temporal shift.
[Pinned calibration implementation](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/calibration.py).

## One fixed integration recipe, not a frozen native plan

1. Keep all 13 existing outer folds, all 7,500 graded frames per protocol,
   250-frame exclusions, immutable 166-column order and original R37 encoder.
   Reuse R54's deterministic 5-Hz eligible H/A training schedule and paired team
   augmentation, with at most 3,000 augmented rows. Require the complete actual
   ±79-frame encoder footprint to lie inside that fold's permitted training
   frame set **before** joining training H/A eligibility, as in the corrected
   current native-context interface. Do not use R61's adapted
   encoders, query labels, human labels, private/sealed material or frame/clock
   identifiers. The latent footprint remains ±79 frames; the original features
   retain their separately established longer physical-history footprint within
   the 250-frame protocol protection. Do not describe ±79 as the total reach of
   all 166 features. No extra context is added.
2. Construct calibration folds **before augmentation**. Sort the permitted,
   scheduled original H/A centers by original frame, and split that ordered
   index list into three contiguous groups with `array_split` semantics.
   Each group is one inner validation set. For that group's training set, use
   only outer-permitted centers strictly outside the inclusive interval
   `[minimum_inner_frame-250, maximum_inner_frame+250]`. This deliberately also
   excludes any holes inside the group's bounding interval. Mirror every
   training/validation index to its augmented counterpart, so an original row
   and its swapped copy always share the same role. The three inner validation
   sets must partition every augmented outer-training row exactly once. These
   folds are a fixed project choice, not an author SVM default.
3. If any group is empty or any purged inner training set is empty, stop that
   candidate as technically unsupported; do not fall back to random CV, shrink
   the purge or choose a different grouping after seeing outcomes. Paired
   augmentation guarantees both binary classes in each nonempty inner set.
   Save frame/pair identities and exact exclusions for every inner/final fit.
4. First replace every nonfinite feature value with NaN through a fixed
   elementwise transform, with no learned statistic or column removal. Put this
   transform, a 166-column `SimpleImputer(strategy='median', add_indicator=False,
   keep_empty_features=True)`, `StandardScaler()` and the SVC **inside** the
   calibrated estimator's pipeline. Each inner clone fits its own imputer and
   scaler using its inner training rows only; the final clone fits them using
   only the complete outer-training pool. All-empty numeric columns become
   zero, and constant-column scales remain one. Do not preprocess once on the
   entire outer pool before inner calibration: that would expose held-out
   observations to calibration-model preprocessing. Preserve all columns;
   no missing-indicator expansion, PCA, feature selection or scaling sweep.
5. Use equal row weights, no class weights and no resampling beyond the existing
   paired schedule. The paired classes are exactly balanced at every level.
   Read probabilities by the explicit class identity H=0/A=1; never infer class
   order from a column position without an assertion. Obtain conditional H
   probabilities for original and team-swapped queries, swap columns back and
   average. This enforces the existing symmetry without an additional model.
6. Predict in the established fixed 512-row calls, with its fixed last-row
   padding rule, including all query rows regardless of their unknown targets.
   Compose with the original slot0 dead mass:
   `qH=(1-d0)*h`, `qA=(1-d0)*(1-h)`, `qD=d0`. Unsupported latent rows retain
   the complete original slot0 vector. Replace only slot0 in complete selected8
   inference and use the existing final live overlay to preserve every D/X
   decision. Keeping raw D mass alone does not ensure decoded D identity.
7. Keep the remaining five probability slots in the six-slot selected inference
   interface, smoothing, decoding, rescue and all
   thresholds exact. There is one 166-feature candidate, no 102-feature arm,
   linear-kernel arm, C/gamma grid, isotonic arm, confidence blend, best inner
   split, duration model or post-score repair.

The outer contexts and labels never include outer queries. The inner calibrator
sees only margins predicted without training on those inner labels and only
their own outer-training labels as calibration targets. It never sees outer
query labels, scores or selected predictions as targets. The existing latent
features can use target-blind observations near queries within their declared
footprint; that is already part of the offline input protocol, not access to
query labels.

The three inner models plus one final model require four SVC fits per outer
fold, or 52 fits for seed42. Inner models are discarded after margin creation;
save their provenance/margins for audit. A later final full-match model must use
the same three-group calibration construction on the permitted full public
training pool, not substitute random five-fold calibration.

## Runtime, licenses and admission conditions

There is no new checkpoint, network inference or package requirement. The
existing isolated head runtime already contains the scikit-learn, NumPy, SciPy
and joblib dependency family used by this recipe. Import/compiled compatibility
still needs a later gate; the old shared executables' drift must not be ignored.
Pin the existing private interpreter, complete package/source/binary hashes and
BLAS settings before any scientific import. Do not install or modify packages.

The official docs warn that SVC fitting scales at least quadratically in sample
count and recommend C-ordered float64 inputs. No source supplies an M1 Pro runtime
guarantee. A 3,000×3,000 float64 kernel matrix is approximately 68.7 MiB, and
3,000×166 features occupy 3.8 MiB; these are arithmetic estimates, not measured
RSS. The 200-MB cache, native workspace, Python and calibration clones must still
be measured. Keep one worker, one BLAS thread and `n_jobs=1`; run the three inner
fits sequentially. SVC performs kernel prediction per query and has no reason to
retain a 7,500×3,000 full query matrix in the chosen 512-row API calls.
[Official SVM guide](https://scikit-learn.org/stable/modules/svm.html).

Scikit-learn is BSD-3-Clause. Its retained LIBSVM source separately carries the
Chang/Lin copyright and three redistribution conditions. Preserve both notices
with any retained source or distributed artifact; no model-weight agreement,
authentication or terms-acceptance flow is needed.
[Pinned scikit-learn license](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/COPYING),
[pinned LIBSVM source/notice](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/svm/src/libsvm/svm.cpp).

A later invented-data gate should run the **complete** imputation/scaling,
three-purged-fold calibration, final fit and fixed query schedule at 600 and
3,000 augmented training rows, 166 features and 7,500 queries. These are 300 and
1,500 original centers spaced contiguously at 5 Hz before pairing, so the
250-frame inner purges are substantive and leave valid training sets. A
64-augmented-row contiguous context could empty an inner fit and is not the
moderate benchmark. Include paired labels, nonfinite/constant/all-empty columns
and uneven-group pure fixtures.
Also exercise empty/purge-exhausted groups as intentional rejection tests.
Check that modifying an inner validation row cannot change that inner model's
preprocessing/fit, that mirrored pairs never cross roles, and that the required
250-frame purges and class identities hold. Verify the effective internal
probability flag is false and exactly one calibrated final estimator remains.

Require finite normalized outputs, team symmetry, unchanged fitted state after
prediction, repeated first queries, query-order/batch/padding diagnostics,
checkpoint reload and a completely independent fresh-process **refit** with
byte-identical canonical probabilities and learned state. Do not require pickle
container bytes to prove equality if timestamps or paths differ; compare the
semantic learned arrays and exact probability bytes. Record all inner/final
support indices, dual coefficients, intercepts, preprocessing statistics,
resolved gammas, calibrator coefficients and fit status. Any nonzero SVC
`fit_status_`, convergence failure, resource overrun or inconsistent output
stops the candidate. Enforce the existing ≤6-GiB peak and <300-second operation
limits for the complete four-fit operation and all-query operation; these are
requirements, not yet observed results.

Only a passing technical gate supports freezing a later native experiment.
Its buffered main spellF1 must strictly exceed the comparable R37 record
**0.7542857142857143** and pass every unchanged selected8 primary/mean-fold,
macroF1, short/matched/spurious/missed, D-recall and false-D safeguard. A seed42
pass permits the fixed seeds43/44; all nine protocol/seed pairs must pass,
followed by independent complete replay, actual public/full feature parity,
two identical 199,319-row CSV builds, format validation and independent review.
The SVC without internal probability fitting and the explicit sigmoid protocol
have no seed-dependent random fitting step here. Consequently standalone head
fits labelled seeds42/43/44 can be identical. The remaining selected probability
slots still use their matched seeds, and all nine protocol/seed pairs remain
required; repeated deterministic heads do not supply independent evidence.
Do not promote on a frame metric, a tie, one protocol or a merely better failed
learner. The 17 adaptively reused possessions remain development evidence and
cannot establish a private/full-match gain or an optimum.

## Source identity evidence

These small official files were fetched into memory only. The three Python
source hashes also match the read-only local isolated head runtime.

| Path at pinned commit | Bytes | SHA256 |
|---|---:|---|
| `sklearn/svm/_classes.py` |68141|`06b89760e22b5df441820f531a9989edfdf22bd5e405a68c5a33fb39e60644b9`|
| `sklearn/svm/_base.py` |44790|`be4d5a830d357be28e2a85d6c80c208d8aa8536858a62e17cad50abf0f7fe62f`|
| `sklearn/calibration.py` |61543|`52339970c65ed05602a8515ebbbdacb2fef1ef24fc683cbace7683fd964269e9`|
| `sklearn/svm/src/libsvm/svm.cpp` |69309|`8e9c41f2e01ac93e07ece96fe8943699cfb7f0323e071d7321fb0032a666c3d6`|
| `sklearn/svm/src/libsvm/LIBSVM_CHANGES` |769|`9f93ab1d9eb903d0aa0c5c6919fa61e7fb56180ba2461741234c40196a18c7d2`|
| `COPYING` |1532|`50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0`|

Read failures are not hidden evidence: the first attempted system Python
standard-library request stopped at the local Xcode-license guard, before its
script executed. Retrying only the standard-library source read with the
existing private interpreter's site loading disabled succeeded. No license was
accepted and no environment repair occurred. Two guessed LIBSVM license-file
URLs returned 404; the actual bundled `svm.cpp` header and project `COPYING`
were then read. The Microsoft publication URL returned 404, so the author-hosted
Lin/Lin/Weng note and Cornell primary paper supply the probability evidence;
this note does not claim that unavailable Platt chapter was read.
