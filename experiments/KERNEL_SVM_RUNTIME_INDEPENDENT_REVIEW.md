# R72 kernel SVM gate independent source review

**Freeze approval withheld pending the runtime-launch correction below and final source/preflight re-review. No model work is admitted by this review.** R65 retains the active model resource slot. This review creates no freeze, admission, technical result, native experiment or promotion authority.

The review inspected the full gate, pure tests, plan, research note, retained author sources and licenses, private-runtime metadata and source-only preflight. A separate source-only calibration/state subreview found no blocking defect in those portions. The root review then identified the blocking launch issue below, which this review accepts. No implementation or existing artifact was changed by the reviewer; this new review is the reviewer's only repository write. Neither reviewer imported scientific packages, read native arrays or labels, installed packages, fitted a model or ran inference.

## Blocking finding: worker isolation ignores the declared hash seed

In the pre-fix draft, `supervise()` line 695 launches the worker with `-I`. Python isolated mode implies ignoring `PYTHON*` environment variables, including the declared `PYTHONHASHSEED=0`. `initialize()` checks environment strings at lines 328–330, but that does not establish the interpreter's active flags or hash-randomization state. Consequently the draft cannot truthfully claim a fixed hash seed or its full declared startup protocol. This is a source-preparation blocker; it was found before any model execution.

The required correction is a private-interpreter worker launch with `-P -S -B`, removal of inherited `PYTHON*` settings before applying the fixed worker environment, and explicit pre-scientific-import checks of the active private base prefix/resolved executable and `safe_path`, `no_site`, `dont_write_bytecode`, `ignore_environment` and `hash_randomization` flags. The intended active values are safe path/no site/no bytecode true, environment ignored false and hash randomization false, with `PYTHONHASHSEED=0`. Source-only preparation may continue to use `-I -S`. Pure acceptance/rejection fixtures should cover the actual startup guard rather than only the environment dictionary.

Root preserved the original draft/preflight/metadata in `data/control_kernel_svm_runtime_v1/before_runtime_guard/` before correcting this unfrozen source. The hashes and observations below document the initial reviewed draft; they do not approve a changed source. Final amended hashes and a matching source-only preflight must be checked before replacing this withheld decision.

## Reviewed identities of the pre-fix draft

| Artifact | SHA256 |
|---|---|
| `experiments/kernel_svm_runtime_gate.py` | `ffe289ad080f0bb961da723b194abdf91bee44c8e050618e75651208dfa76e87` |
| `experiments/test_kernel_svm_source_contract.py` | `73fdaf6d71e80dc677940bb9c1592eeb3cd8041d2a29c176b074dbb5cef0573a` |
| `experiments/KERNEL_SVM_RUNTIME_PLAN.json` | `4a559707fb759ac93be35d521989a5a4867976fa5d62f393ae9da6ea7fb931f0` |
| `experiments/NEXT_TEMPORAL_LEARNER_RESEARCH_20261031.md` | `f1408fa35d0d8594f237c92a28d13803f4f7db3001434e641369683d4e3cd512` |
| `data/control_kernel_svm_runtime_v1/source_metadata.json` | `474061797e8246c76be4f9e95fc0c9b1b32dae5b19dbf3cecb5cdd13d3b858b5` |
| `data/control_kernel_svm_runtime_v1/runtime_manifest.json` | `25117aa192d4625e3cdbc45ae45cab992508fb1ce4beabc1f768935b3492e839` |
| `data/control_kernel_svm_runtime_v1/runtime_symlinks.json` | `cd3043819dd05f0204046e5e85007b5e2f71d3939974efd7197a6b4ce337b9c1` |
| `data/control_kernel_svm_runtime_v1/source_preflight.json` | `42b60410017c1aeabfa8f2192e133c6bf3ac9398f074bea3d5d0d7eb43747b7d` |
| `experiments/modernnca_schema_contract.py` | `f59109473e50b527a8e697dd9bd5a4d0a12e804ec4e0850b33b6d5400486c0ff` |

The retained source revision is scikit-learn `866c0f51e7560ef0303cbcc5f159df5382ea9e3f`. All five retained author files match their declared lengths and hashes; the three Python sources also match the read-only installed sources. Both the scikit-learn BSD-3-Clause notice in `COPYING` and the separate Chang/Lin redistribution notice in the retained `svm.cpp` header are present. No new checkpoint, package acquisition or license acceptance is needed for this prepared method.

## Learner and author delegation

The implementation contains one ordinary author Pipeline, with no unused estimator subclass or replacement fit algorithm. It fixes RBF SVC C=1, gamma=`scale`, shrinking, tolerance, 200-MB cache and the stated remaining configuration. It leaves the pinned deprecated `probability` argument omitted and rejects any actual fitted SVC whose `_effective_probability` is not `False`, whose classes differ from `[0,1]`, or whose fit status is nonzero. Learned support vectors, coefficients, intercepts, preprocessing statistics and resolved gamma are checked and recorded.

The nonfinite-to-NaN copy, median imputer with retained empty columns, and StandardScaler are inside the Pipeline passed to `CalibratedClassifierCV(method='sigmoid', ensemble=False, n_jobs=1)`. They are therefore cloned and fitted separately on each actual inner training subset. No preprocessing fit occurs on a full pool before inner validation.

The pinned author `calibration.py` lines 457–502 call `cross_val_predict` for the explicit folds, then fit one final full-context estimator and fit the sigmoid on the resulting OOF margins. The temporary Pipeline observers delegate their original methods exactly once, preserve `decision_function` naming with `functools.wraps`, verify the ordered raw X/y fingerprints of all three inner fits and the final fit, and reject unexpected or repeated held-out margin calls. No recorder path or audit attribute is added to estimator state. The code requires exactly four actual fits, all three genuine held-out margin calls, one calibrated final classifier, one sigmoid calibrator and H0/A1 class order.

## Splits, state and reproduction

The pure split function partitions the original rows into three chronological contiguous chunks with array-split sizes. It excludes every training center within 250 actual frames of any held-out center before assigning both mirrored mates the same role. All augmented validation indices appear exactly once; low-count or purge-exhausted contexts reject. These are purged blocked folds, as declared, with training allowed on both sides of a held-out chunk. They are not an expanding forward-only scheme.

The invented moderate/full contexts have 300/1,500 original centers spaced five frames apart, then append mirrored rows and complemented labels for 600/3,000 total training rows. Each shape has 166 columns and 7,500 feature-only queries. Nonfinite values, paired all-empty columns and paired constant columns are deliberately included. No query labels or accuracy scores are created. Generated raw arrays are marked read-only and their complete byte fingerprints are rechecked.

The recorder saves all three actual inner pipelines and the final pipeline, all actual OOF margins, the reconstructed full OOF margin vector, explicit learned preprocessing/SVC records, sigmoid state and the complete calibrated model state. `semantic_state` traverses every nested estimator `__dict__`, with exact array bytes and scalar representations; callable identities are paired with immutable runtime/source hashes. Prediction checks compare complete fitted state before/after, then reload the calibrated model and all four actual fitted pipelines by semantic digest. A fresh process at each size regenerates the data and requires exact recipe, fold, inner/final learned-state, OOF-margin and original/swapped/averaged canonical-probability bytes. Pickle container bytes are not used as equality evidence.

The query schedule uses ascending canonical 512-row calls, pads the last batch with its last real row and discards padding. Original and swapped calls are separate, then swap-back H/A probabilities are averaged. First original/swapped calls are independently repeated after the full schedule; the vector symmetry check is exact. Prefix-eight and reversed-first-batch comparisons remain diagnostics only. Complete reload repeats all original/swapped canonical queries, with finite normalized probabilities and final state/array immutability checks.

## Runtime closure and execution boundary

I independently called only `source_contract()` and `verify_runtime()` under the private head interpreter with `-B -I -S`. All **9,770** significant runtime-file hashes and **13** symlink records match the complete regenerated inventory, including the copied private Python/stdlib, head runtime, acquisition manifest and explicitly traversed existing package target. The private executable resolves to `data/control_isolated_python_v1/python/bin/python3.12`, SHA256 `ab9c532a75ce81eb96562bda04138e7a0b7528bd242c82a436ebeb1405aa361f`. The head package link resolves to the existing `.venv/lib/python3.12/site-packages` directory; its bytes are bound read-only, and the old shared `.venv` executable is not used.

The declared package metadata is scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1, joblib 1.6.0 and threadpoolctl 3.7.0. This inspection verifies metadata and files, not that those scientific packages import or execute successfully. The retained pre-fix preflight binds the initial reviewed source and plan hashes shown above. Its source-only flags and 9,770-file count agree with the independently repeated checks; no scientific module entered `sys.modules`. This preflight cannot approve the pending launch correction.

Each later worker must verify the freeze and separate matching root admission before any scientific import. The controller checks admission before launching any worker. The admission must specifically assert invented-only R72 scope, R65 closure/pass, stopped other model workers and the reviewed freeze hash. Neither preparation, tests, preflight nor this review grants that admission. The worker disables network access, checks the explicit private package origin/version, limits numerical thread pools to one and rejects Torch.

The external supervisor enforces one serial worker, 6-GiB RSS and 900-second worker time. Its independently read operation marker covers the entire three-inner-plus-final fit operation under 300 seconds and the full query/repeat/symmetry/reload operation under 300 seconds. Resource abort or nonzero exit overrides any success summary; the controller stops on the first failed stage. It requires the exact ordered moderate-first, moderate-rebuild, full-first, full-rebuild identities and successful fresh-refit flags. New output directories and exclusive log/checkpoint creation prevent retrying into an existing run.

No native feature/label/scorer/selected-prediction reader is present in the gate. Array generation and model imports occur only in the admitted worker. The only pickle reads reopen models just written in that same new invented output directory. Source-only inventory reads are restricted to source, metadata and runtime files. There is no package installation or runtime repair path.

## Validation and limits

All **13 pure source-contract tests pass** under the private interpreter with `-B -I -S`. They exercise actual-frame purge, paired roles, uneven groups, empty-context rejection, source-only boundaries, author observer delegation, exact stage identities, stop-on-failure behavior and resource-abort precedence. I also independently rehashed the runtime inventory and all source/preflight/metadata bindings. No old frozen source, dependency file, native artifact or ledger was edited.

The current R72 directory has no `protocol_freeze`, `resource_admission.json` or `runtime_assessment`. This is prepared source only. Scientific import compatibility, complete numerical execution, fit/query duration, memory use, semantic serialization and fresh-refit determinism still require the later separately admitted invented gate. A passing gate would be technical evidence only. It cannot establish possession improvement or authorize native processing, full-match construction or promotion; the later native plan, unchanged selected8 quality safeguards and independent review remain separate requirements.
