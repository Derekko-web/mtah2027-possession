# Gaussian-process runtime gate independent source review

**Approved for root source freeze at the exact hashes below. No unresolved blocking findings remain. This review grants no model/resource admission, native processing or performance claim.** Root must separately freeze/verify the source and record the required matching admission before any scientific worker starts.

This completes the independent source review; the earlier partial review messages were not a completed approval. The source, tests and plan were revalidated unchanged, and the pure suite and runtime/notice checks were repeated before writing this report. A separate source-only posterior-observer subreview found no blocker. The reviewer creates only this report and `data/control_gaussian_process_runtime_v1/independent_review.json`; no source, plan, old report, frozen file, package or runtime is edited.

## Reviewed identities

| Artifact | SHA256 |
|---|---|
| `experiments/gaussian_process_runtime_gate.py` | `9d14c861f27e87d721980fc5813f6d197e0a027e936416031143d2a136784f17` |
| `experiments/test_gaussian_process_source_contract.py` | `b7ce1d4ef7ab080ec40b3d6e7999bbd44929b547f6034069ac46a24e4edce10d` |
| `experiments/GAUSSIAN_PROCESS_RUNTIME_PLAN.json` | `b7f52f1cb7b5bc56da5feb4b22abdcdc2f0e942ca3e1d7428d9db8c380d6c7bf` |
| `experiments/GAUSSIAN_PROCESS_LEARNER_RESEARCH_20261101.md` | `e4b06c1c656a356aa30d409978b50850e1fbcf313b3ac9ed6a869ba8181ca224` |
| `source_metadata.json` | `1c60d95bf2d0255e45ccae24aa688347031305cb3994a43eac2b7b4c22285123` |
| `runtime_manifest.json` | `25117aa192d4625e3cdbc45ae45cab992508fb1ce4beabc1f768935b3492e839` |
| `runtime_symlinks.json` | `cd3043819dd05f0204046e5e85007b5e2f71d3939974efd7197a6b4ce337b9c1` |
| Retained `source_preflight.json` | `b442ccfafcc4a79222afa4e6af1e56791156d30849fc3b7815a98f6add8af9e7` |

Round-relative metadata paths are under `data/control_gaussian_process_runtime_v1/`.

## Author code and convergence proof

The observer is tied to scikit-learn author revision `866c0f51e7560ef0303cbcc5f159df5382ea9e3f`. The retained `_gpc.py` source matches its installed source exactly, SHA256 `90628fca68b26aac2c4f57305b6302c549952d3873e6651209ffcafb2c95ff3d`. Source-only AST inspection locates `_posterior_mode` at line443, the first iteration statement at467, the stopping condition at491 and its actual `break` at492. A non-executing compile of that complete pristine source identifies the exact nested method code object; its marshalled hash is `4aaa172842e82a9e61a588a8a3cadbeabacce91886e46117674bf2340182a6ae`.

Before a later fit, the worker requires the imported original method's compiled code bytes/hash to match that pristine code and rejects a pre-existing trace. Its temporary wrapper delegates the unchanged author method exactly once per invocation. `sys.settrace` selects only frames whose `f_code` is that original code object, checks the exact estimator and frame identities, and reads local values without assigning locals or changing estimator/Newton arithmetic. The previous trace is restored in `finally`, including on failure.

The pinned optimizer-None author flow genuinely invokes mode finding twice: once inside the fixed-kernel likelihood calculation and once for predictive temporaries. Both pass `return_temporaries=True`; warm start is false. The observer checks an actual all-zero initial `f` at the first iteration line, finite actual arrays on every recorded iteration, finite likelihood, contiguous iteration indices, explicit initial previous=`-inf`, and the actual break-line event. It requires exactly two successful calls, each with fewer than100 recorded iterations. A loop-cap exit cannot pass by masquerading as convergence.

Every later likelihood delta must agree with its consecutive observed values and may not be below `-1e-10`. The final actual delta must lie in `[-1e-10, 1e-10)`, with the observed stopping history matching the unchanged author predicate. The small declared negative interval is a fixed roundoff allowance, not an unbounded likelihood-decrease acceptance. No result-dependent tolerance, retry, extra solver call, optimizer or arithmetic replacement is introduced. Trace inspection/hash overhead belongs to the complete fit operation.

## Fixed learner, prediction and retained state

The single declared classifier is the unchanged author binary Bernoulli/logistic Laplace `GaussianProcessClassifier`: fixed unit ConstantKernel times fixed RBF, optimizerNone, zero optimizer restarts, max_iter_predict100, warm_startFalse, copy_X_trainTrue, seed42, binary one-vs-rest and n_jobs1. The length scale is fixed from actual transformed own-training population/global variance as `sqrt(166*Z.var()/2)`, with the declared zero-variance fallback `sqrt(.5)`. This matches the stated RBF geometry without a kernel/scale/optimizer grid or a separate calibrator.

Median imputation with retained empty columns and StandardScaler fit only the supplied training pool after a copying nonfinite-to-NaN transform. All166 columns remain. `FittedGP` owns only that preprocessing and delegates probabilities to the unchanged author `predict_proba`; no GP prediction formula, probability temperature, clipping or confidence blend is substituted. The invented moderate/full pools contain600/3000 balanced paired rows and7500 feature-only queries. No query labels or accuracy scores are generated.

The full semantic serializer traverses every fitted `__dict__`, exact array/scalar witnesses and callable identity, including explicit read-only `RandomState.get_state()`. It retains preprocessing, classifier/template/base estimator, both kernel objects, training inputs/labels/classes, pi/W-sqrt/L/f-cache and cached likelihood state; iteration traces also retain the actual finite W and other Newton temporaries. No fitted field exclusion or RNG reset is used. The only likelihood getter used by checks is the no-argument cached getter; checks do not request theta-dependent posterior recalculation.

Canonical prediction uses ascending512-row calls with last-real-row padding, separate original/swapped calls, swap-back averaging and finite normalized two-column checks. First original/swapped calls and team symmetry must be exact. Prefix-eight/reversed-batch values remain nonselecting diagnostics. Complete semantic state is checked before/after all queries and after local checkpoint reload; all canonical original/swapped probabilities must repeat after reload. At each size a new-process complete refit must reproduce recipe, fit/state/kernel records, both posterior traces and canonical raw/averaged probability bytes. Pickle container bytes and varying timing records are not used as semantic-equality proof.

## Dual clocks, resource admission and evidence closure

The gate has its own dual-clock helper rather than a dependency on mutable native helpers. Atomic operation markers and JSONL events retain integer UTC and monotonic nanoseconds around each complete fit and complete query/reload operation. Both elapsed values must be nonnegative and below300seconds; the complete worker must be below900seconds in both clocks. Negative elapsed intervals reject. These guards do not claim detection of every clock adjustment that leaves computed intervals positive. The external supervisor checks active deadlines and independently requires exactly two complete events, their arithmetic/summary bindings and the final atomic marker before accepting success.

One worker, one numerical thread and6-GiB RSS remain fixed. A nonzero/resource-aborted worker cannot pass a retained success summary. The controller requires all four distinct named moderate-first/rebuild and full-first/rebuild stages in order and stops at the first failure. New output directories and exclusive logs/checkpoints prevent reuse of an old output. No alternate kernel, precision, context, partial-stage rescue or resource waiver is present.

Both controller launch and worker require separate matching root admission. It must assert invented-only GP scope, R73 closed/passed, other model workers stopped and the reviewed freeze hash. Preparation, tests, source preflight, this review and source freeze do not grant that admission. Before scientific imports the worker checks the active private base/executable and effective -P/-S/-B/hashseed0 flags, uses a cleaned inherited Python environment, binds package origin/version, disables network access and rejects Torch. No native observation/label/scorer reader or package-install path is present.

I independently repeated `source_contract()` and `verify_runtime()` under the private head interpreter with `-B -I -S`. All **9,770** significant private interpreter/stdlib/package file bindings and **13** symlink records match the complete current inventory. All four pinned author files and **26** retained distribution metadata/license/notice files match; each retained notice also matches its existing installed file. The upstream BSD notice and bundled dependency notices remain preserved. The retained source-only preflight binds exactly the reviewed source/plan/research hashes and records no scientific imports, model calls or native-array reads.

## Validation and limits

All **20 pure tests independently pass**. They exercise fake author-frame identity/real-break records, exact source/AST/pristine compilation, fixed convergence failures, paired roles/variance, full RNG serialization, dual-clock operation/supervisor rejection, exact operation evidence, private-runtime/environment guards, exact four-stage identities, failure-stop and resource-abort precedence. The suite and metadata verification import no NumPy, SciPy, sklearn, Pandas, Torch or joblib. No scientific array or model is decoded, and this reviewer does not call preflight or freeze.

At completion of this review there is no GP freeze, resource admission or runtime-assessment directory. Actual compiled scientific import compatibility, numerical convergence, serialization, runtime/memory and exact fresh-refit behavior remain unmeasured until a separately admitted invented gate. The source-only preflight and this freeze approval are not an actual technical pass. No native possession improvement, full-match accuracy, full candidate or promotion is established or authorized.
