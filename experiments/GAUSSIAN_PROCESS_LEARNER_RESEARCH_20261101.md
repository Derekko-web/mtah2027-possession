# Research: one fixed Laplace Gaussian-process ownership learner

Prepared 1 November 2026 UTC. Research only. The existing private Python was
used with `-B -I -S` for standard-library text, HTTP, metadata and hash reads.
No scientific library was imported, native feature/label array decoded, model
fitted, probability produced, score computed, package installed, runtime changed,
plan frozen or selected artifact changed. This note is the only new artifact
created by this research task. A separate source-only skeptic checked the recipe
and implementation hazards without running a model or writing an artifact.

## Decision

**GO for considering one separately admitted invented-data feasibility gate;
NO-GO for native processing on the strength of this research.** A fixed binary
Laplace GaussianProcessClassifier is a distinct likelihood/posterior hypothesis
with existing dependencies. Its dense computational cost, silent convergence
limits and probability behavior must pass an actual technical gate first.
This note grants no resource admission, creates no execution plan and schedules
no model. Finish or close the currently admitted R73 work before any later GP
model worker; do not compete with another model worker.

No Gaussian-process ownership learner appears in the inspected
[work history](WORK_HISTORY.md), existing Markdown research reports or plan JSON
files. The only unrelated Laplace search hits were Torch distribution source
hashes. This is a bounded repository search, not scientific novelty.
The [RBF SVM research](NEXT_TEMPORAL_LEARNER_RESEARCH_20261031.md),
[R72 actual gate](KERNEL_SVM_RUNTIME_REPORT.md) and
[factorized target research](FACTORIZED_LATENT_RESEARCH_20261027.md) already cover
the RBF geometry and conditional live ownership interface. Those components
must not be relabeled as new.

The proposed difference is the Bernoulli/logistic likelihood, a GP function
prior and variance-aware approximate posterior prediction. R72 instead fits
a C-SVC soft margin and a separate sigmoid calibrator. Sharing its RBF bandwidth
makes this an objective/probability comparison, not a claim that a new kernel
was discovered. The GP still consumes the same existing temporal summaries;
it learns neither a new sequence representation nor a new temporal decoder.

## Primary sources actually read

Rasmussen and Williams, **GPML chapter 3**, sections 3.3–3.4.4, equations
3.12–3.25 and algorithms 3.1–3.2, supply the binary model and stable
implementation. With binary targets `t`, the posterior mode maximizes

`sum_i [t_i*f_i - log(1+exp(f_i))] - 0.5*f.T*K^-1*f`.

The Laplace posterior covariance is `(K^-1+W)^-1`; probability prediction
integrates the sigmoid against an approximate Gaussian latent prediction.
The numerical algorithm factors `B=I+sqrt(W)*K*sqrt(W)`, avoiding an explicit
inverse of K. The chapter warns that local Gaussian curvature can misrepresent
the posterior. Training is cubic per Newton step; predictive variance adds
quadratic work per query. These facts do not establish calibration or temporal
generalization for the challenge's dependent possessions.
[Author chapter](https://gaussianprocess.org/gpml/chapters/RW3.pdf).

Williams and Barber, **PAMI 1998**, section 3.1 and appendix A, were read in
the institution's indexed final paper and the author's compressed PostScript.
Appendix A supplies the five-error-function approximation to the sigmoid
integral that the maintained implementation uses. Their broader Bayesian
method also integrates kernel-parameter uncertainty; the fixed-kernel recipe
below does not reproduce that part of their method. Their reported benchmark
results do not imply a possession gain.
[Final paper](https://publications.aston.ac.uk/id/eprint/4491/1/IEEE_transactions_on_pattern_analysis_20%2812%29.pdf),
[author copy](https://homepages.inf.ed.ac.uk/ckiw/postscript/pami_final.ps.gz).

**GPML chapter 4**, section 4.2.1, equation 4.9, gives the squared-exponential
covariance `exp(-distance^2/(2*ell^2))` and its strong smoothness assumption.
It supplies the covariance formula, not an optimal length scale for these
features or a reason to assume smooth possession boundaries.
[Author chapter](https://gaussianprocess.org/gpml/chapters/RW4.pdf).

The **official scikit-learn guide**, classification subsection, describes
Laplace/logistic prediction and an example where larger approximate marginal
likelihood yields worse held-out log loss, including probabilities near 0.5
away from data. The guide also warns about dense rather than sparse operation
and reduced efficiency beyond a few dozen features. At 166 columns these are
direct reasons for caution, not proofs of failure. Bayesian probabilities are
not automatically calibrated, especially under a changed temporal distribution
and repeated augmented/correlated observations.
[Official guide](https://scikit-learn.org/stable/modules/gaussian_process.html).

## One prospective kernel and learner recipe

Use exactly the same permitted outer-training rows, ordered original102 plus
fixed R37 latent64 features and original/team-swapped pairing as the existing
ownership interface, with at most 3,000 augmented rows. Preserve all 166 columns.
Map nonfinite values to NaN, fit median imputation with
`keep_empty_features=True, add_indicator=False`, then fit `StandardScaler` on
that outer-training pool only. All-empty columns become zero; no new indicator,
PCA, feature selection, class weighting or separate calibration is proposed.
There are no inner calibration predictions, so there is no separate inner
preprocessing stage. Outer query observations cannot fit any statistic.

Let `Z` be this finite float64 transformed training matrix, `p=166`, and
`v=Z.var()` using the population/global variance semantics of SVC's dense
`gamma='scale'`. Set

`gamma_scale = 1/(p*v) if v != 0 else 1`

`ell = sqrt(1/(2*gamma_scale))`.

Require finite positive resolved values. Use the single covariance
`ConstantKernel(1.0, constant_value_bounds='fixed') *
RBF(ell, length_scale_bounds='fixed')`.
The pinned author [SVC source](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/svm/_base.py)
defines the variance rule; the pinned
[GP kernel source](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/gaussian_process/kernels.py)
defines the RBF denominator and fixed-bound semantics. This is a project choice
to hold the earlier SVM geometry comparable. It is not the GP software default,
a GPML-author bandwidth recommendation, a train-label optimization, or a median
heuristic. There is one recipe and no bandwidth/optimizer/kernel sweep.
The SVC/GP kernel equivalence is mathematical; their distance evaluation and
floating-point arithmetic need not produce byte-identical kernel matrices.

For centered unit-variance nonconstant columns, this is approximately
`sqrt(active_columns/2)`, or 9.1104 when all166 columns are active.
Use actual `p*v`, not the integer active-column shortcut: constant-column and
floating-point handling can make them differ. The all-zero-variance fallback
is `ell=sqrt(0.5)`, preserving the stated SVC equivalence.

The geometry concern can be seen without observing any native array. With
166 centered unit-variance coordinates, two independent draws from that
empirical distribution have mean squared distance332. The kernel at that
distance with ell1 is `exp(-166)`, about `8.1e-73`; at the proposed ell it is
`exp(-2)`. This is an illustrative calculation at the mean distance, not the
mean kernel value or a claim that real feature distances concentrate there.
Feature dependence, missingness and a few informative dimensions can still
make this shared metric poor. When off-diagonal kernels vanish, the RBF
length-scale derivative also shrinks; starting an optimizer at ell1 is not a
reliable remedy. No optimizer is used in the proposed recipe.

| Parameter | Proposed value | Relationship to software defaults |
|---|---|---|
| `kernel` | unit-amplitude fixed RBF above | explicit project scale rule |
| `optimizer` | `None` | deviation from L-BFGS-B default; no hyperparameter search |
| `n_restarts_optimizer` | `0` | default; inactive with no optimizer |
| `max_iter_predict` | `100` | default, including posterior calculation during fit |
| `warm_start` | `False` | default; each posterior calculation starts from zero |
| `copy_X_train` | `True` | default; retain independent training storage |
| `random_state` | matched seed, initially42 | explicit reproducibility choice |
| `multi_class` | `one_vs_rest` | default; inactive in the required binary case |
| `n_jobs` | `1` | explicit one-worker setting; binary path has one GP |

The [official API](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)
defines these controls. Its prose says kernel parameters are optimized by
default, but the inspected 1.9.1 `kernel=None` branch actually constructs both
default bounds as fixed. Avoid this ambiguity by specifying the complete
kernel, both fixed bounds and `optimizer=None`. There is no GPR `alpha`
parameter in this classifier; do not add jitter, WhiteKernel or a regression
surrogate if the fixed classifier fails.

This binary fixed-kernel path initializes an RNG but has no optimizer-restart
draws. Standalone heads across seeds can consequently be identical; this is
not independent statistical evidence. The other selected probability slots
still require their matched seeds and every original confirmation pair.

## Exact implementation hazards and state

The pinned [GPC source](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/gaussian_process/_gpc.py)
was read through binary fit, posterior mode, likelihood, latent prediction,
probability prediction and public dispatch. With fixed kernel, `fit` still
runs posterior mode twice: likelihood evaluation, then prediction precompute.
`warm_start=False` starts both from zero. The loop stops when successive
approximate likelihood improvement is below `1e-10`, including a decrease.
Reaching100 iterations has no cap warning or public `fit_status_`.
`f_cached` is written even without warm starts. A later
`log_marginal_likelihood(theta=...)` changes it even with a cloned kernel;
the no-argument accessor returns the saved scalar. Probability prediction
reads stored `pi_`, `W_sr_`, `L_` and training data and uses the author's fixed
error-function coefficients. Latent variance uses SciPy `solve`, not an
explicit `solve_triangular` call.

These facts impose admission checks; they are not reasons to modify the
installed algorithm. A future gate must observe both actual posterior loops
without changing their arithmetic, recording iteration count, successive
likelihoods and the actual stop condition. It must prospectively reject cap
exhaustion, nonfinite values, substantial decreases and invalid probabilities.
Exact numerical tolerances belong in the separately reviewed prospective
gate, before the first scientific call; this research selects no tolerance
after a result. A successful `fit` return or lack of warning alone cannot pass.

Keep a semantic record of the complete estimator/preprocessor state, including
training arrays and labels, imputer/scaler statistics, classes, both kernel
objects and parameters, `pi_`, `W_sr_`, `L_`, `f_cached`, likelihood and RNG state.
Prediction/reload checks must compare the complete state, not selected
attributes or pickle-container bytes. During those checks use the cached
no-argument likelihood accessor, not an extra posterior evaluation. Repeated
canonical query bytes and an independent fresh-process complete refit must
match exactly under the fixed call schedule. Query order/batch/padding checks
and team symmetry need the same prospective numerical policy as the existing
technical contracts; no post-failure clipping or changed call shape.

The pinned [author tests](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/gaussian_process/tests/test_gpc.py)
cover fixed kernels, hard/probability consistency and likelihood behavior.
Their analytic-gradient comparison is marked expected-failure against issue31366.
That does not affect the proposed no-optimizer path, and those small fixtures
provide no 3,000-row resource or exact fresh-process guarantee. They were read,
not executed.

## Resource feasibility is unresolved

At n3,000, one dense float64 square matrix is72,000,000bytes, about68.7MiB.
Several kernel/Newton/Cholesky/work arrays coexist. A 3,000×166 training matrix
is about3.8MiB; a 3,000×512 cross-kernel is about11.7MiB. These are arithmetic
storage estimates, not peak RSS measurements. There is no LIBSVM-style
200-MB cache control here, and the algorithm retains all training rows.

The two posterior calculations double the Newton work relative to assuming a
single solve. Disabling hyperparameter optimization avoids repeated likelihood
and gradient evaluations, but leaves the cubic posterior iterations. Per-query
latent variance is materially more expensive than an SVM margin. The leading
quadratic query-work scale for 7,500 original plus7,500 swapped queries is
135billion n-squared scalar operation units at n3,000, before padding,
diagnostics and checkpoint reload. This is a complexity calculation, not an
instruction count or runtime forecast. It cannot be extrapolated safely from
R72's under11-second whole-stage result.

The local SciPy1.18.1 `solve` wrapper permits automatic matrix-structure
detection when `assume_a=None`; it is incorrect to assume from the generic call
alone that it necessarily refactors L as a dense general matrix on every call.
The maintained call must remain unchanged and its actual total cost measured.
[Official SciPy solve documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html)
currently displays1.18.0; the local1.18.1 source identity is recorded below,
so the live page is not used to claim exact binary implementation parity.

A possible later gate must cover the complete fit/preprocessing operation and
the complete fixed512-row original/swapped7500-query, repeat, diagnostic and
checkpoint-reload operation, at both600 and3,000 augmented166-column contexts.
Retain paired labels, true schema swap, nonfinite/all-empty/constant features
and feature-only queries. Require both independent fresh-process rebuilds;
there are four sequential stages, without native arrays or query labels.
Do not cache query results to make prediction timing pass.

Admission requires measured peak≤6GiB, each complete operation<300seconds,
each whole worker<900seconds, one numerical thread and one model worker.
Both UTC elapsed time and the established local monotonic clock must be
enforced from outside the worker, with conservative launch-to-exit evidence.
Python's private macOS `mach_absolute_time` clock excludes host suspension;
a monotonic duration alone cannot certify UTC wall bounds.
[Existing clock audit](NATIVE_RUNTIME_CLOCK_AUDIT_20261031.md),
[R72 audited wall evidence](KERNEL_SVM_RUNTIME_REPORT.md).
Any failure closes this fixed candidate; do not shorten the context, change
precision/kernel/iteration cap, waive a clock bound or run a variant afterward.

## Boundaries of any later native decision

Passing invented data would show only execution feasibility. A later reviewed
native freeze must bind the actual immutable feature builders, complete runtime,
schema, eligible rows and context constructor before model calls. Retain only
released labels11185..18684, all13 outer folds across the5/5/3 protocols,
250-frame purges, the own-training5Hz H/A schedule and paired cap3,000.
Require the complete actual ±79-frame frozen R37 encoder footprint to lie in
that fold's permitted training frame set before applying H/A eligibility.
That ±79 describes the encoder; original physical-history features retain
their separately established footprint under the existing250-frame protection.

Every query, including unknown live status, follows the same target-blind
procedure. Read classes by explicit H0/A1 identity. Average original and
team-swapped probability vectors after reversing swapped class columns.
Compose `qH=(1-d0)*h`, `qA=(1-d0)*(1-h)`, `qD=d0`. Unsupported latent rows
retain the complete original slot0 vector. Replace only slot0; retain the
other five selected inference slots, decoder, thresholds and final D/X-preserving
live overlay. There is no clock/frame feature, query-label filtration, public
label-derived segment feature, manual/private/sealed target, new observation,
hosted inference or acquired checkpoint.

The [closed R65 diagnostic](TABPFN_NATIVE_REPORT.md) raises the current comparable
buffered main benchmark to **0.7771428571428571**, although its incomplete,
worker-aborted learner remains ineligible. A future candidate must strictly
exceed the strongest comparable observed main score, including that diagnostic,
and pass every unchanged selected8 primary/mean-fold, macroF1,
short/matched/spurious/missed, D-recall and false-D safeguard. Merely beating
R37's0.7542857142857143 is insufficient. Seed42 success would still require
all9 protocol/seed42–44 pairs, independent full replay, actual public/full
feature parity, two identical199,319-row CSV builds, validation and independent
review before promotion. The17 adaptively reused possessions remain development
evidence; neither primary sources nor a successful screen establish a private
full-match improvement or global optimum.

## Source, runtime and license identities

The official1.9.1 annotated tag object
`f0668270731746c1b4517f0e9e2cd5b8c2ba3784` resolves to author commit
`866c0f51e7560ef0303cbcc5f159df5382ea9e3f`. The tag is unsigned; do not describe
it as a verified cryptographic signature. The three GP Python files match
the existing runtime byte-for-byte. All network files below were read into
memory only; no source archive or paper was written by this task.
[Official tag reference](https://api.github.com/repos/scikit-learn/scikit-learn/git/ref/tags/1.9.1),
[tag object](https://api.github.com/repos/scikit-learn/scikit-learn/git/tags/f0668270731746c1b4517f0e9e2cd5b8c2ba3784).

| Read source | Bytes | SHA256 |
|---|---:|---|
| pinned `sklearn/gaussian_process/_gpc.py`, local exact |39387|`90628fca68b26aac2c4f57305b6302c549952d3873e6651209ffcafb2c95ff3d`|
| pinned `sklearn/gaussian_process/kernels.py`, local exact |85311|`e985326714f38ab7a6c7cbe10a9eb3409fdcec198c1708de2194fe979bf54f7b`|
| pinned `sklearn/gaussian_process/tests/test_gpc.py`, local exact |11927|`ade3641cae0eb5508cab192b77d2d0854cf1b108b0d36e0fe595610ce2d73485`|
| pinned `doc/modules/gaussian_process.rst` |25323|`29525ca7515a4846eb732da403cfc9afb10231e1e33a491e6be920ea07759057`|
| pinned upstream `COPYING` |1532|`50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0`|
| local wheel `licenses/COPYING`, includes bundled notices |16881|`fc8dc597df0e3f51da65e2a71021e7f92e0b481998ec572a53c03ac0441db040`|
| local `sklearn/svm/_base.py` |44790|`be4d5a830d357be28e2a85d6c80c208d8aa8536858a62e17cad50abf0f7fe62f`|
| local `sklearn/preprocessing/_data.py` |130712|`4e4588e51cf26364c0b28c1ee7694a9029798a6c6e34b1098bcbac1149da0325`|
| local `sklearn/impute/_base.py` |42890|`863ce9620e3a1808927af9c959899ed713dd85cb844c261f88268032d87561ca`|
| local `scipy/linalg/_basic.py` |76094|`2f1fd4a6f90744695655b2c63c5a22f4e3379944e5b7bb5a7589151860d2cd7b`|
| local `scipy/linalg/_decomp_cholesky.py` |16753|`8049dead6bf409e4e0d45cefb9c3d0e05974fc6df8c3f735d74cf97c48a83f91`|
| local `_batched_linalg.cpython-312-darwin.so`, hash only |263112|`f343ce939a78ac27ed272569e3a52756e01ccdbaaf7b118c1fd37c0850b2e0d0`|
| author GPML chapter3 PDF |815687|`e010089108ff5be515a6b713ae7a86448974ab1bd8b03f571163936f4b6f5f1e`|
| author GPML chapter4 PDF |691362|`86008036a346076ff07f8f0a3a94725ad12c572e6bf62bd16ee7f057a42d4175`|
| author Williams/Barber compressed PostScript |131548|`55c7067b05203355ee8cedf882ce97a2027b754c2e9e3d06bd74953af42291cd`|
| that PostScript after stdlib gzip decompression |591368|`5e862436b014b7005cd21891e89d33bcc972a2d24adf641239d5fed91511e5ce`|

The private executable is
`data/control_isolated_python_v1/head_runtime/bin/python`, Python3.12.14.
Under source-only `-I -S`, prefix/base_prefix resolve to the copied private
`data/control_isolated_python_v1/python` base. Its existing site-packages
symlink resolves to `.venv/lib/python3.12/site-packages`; a future gate must
bind all significant resolved files and binaries rather than trust this path.
Metadata reads identify scikit-learn1.9.1, NumPy2.5.4 and SciPy1.18.1; these
reads establish package identity claims, not compiled import health.

| Existing distribution metadata | SHA256 |
|---|---|
| `scikit_learn-1.9.1.dist-info/METADATA` |`32e503c710a61e4e04be4edb450f9f8141233ac6af3f3db59a877667538fa4ec`|
| `numpy-2.5.4.dist-info/METADATA` |`e2b909d6a37d76c2bc130f528d7fed0c517322bce16dc7846b6a6df30e200403`|
| `scipy-1.18.1.dist-info/METADATA` |`1adfbde8be136cbf348b889c51028e7e1824cdb90d29a26d8e81501948864fae`|

Scikit-learn source declares BSD-3-Clause; retain its notice and the existing
wheel's additional bundled-library notices if distributing the runtime.
The wheel COPYING is longer than upstream because it includes libomp/LLVM
notices; the differing hashes are expected and were inspected, not hidden.
Existing NumPy/SciPy and their bundled dependency notices remain applicable.
No new model-weight license, login, terms acceptance or downloaded executable
is required for this proposed existing-library learner.
[Pinned upstream license](https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/COPYING).

Access limitations are retained: an initial guessed Microsoft paper URL returned
404; direct institution-PDF retrieval through private stdlib HTTPS failed
certificate validation, and one browser opening was restricted. The search
reference subsequently exposed the institution's indexed full paper; the
author-hosted compressed PostScript fetched successfully and was inspected
with stdlib gzip. The browser cannot parse gzip itself. No TLS validation was
disabled, external converter executed or unavailable paper claimed as read.
