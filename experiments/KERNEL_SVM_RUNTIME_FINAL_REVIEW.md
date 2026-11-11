# R72 kernel SVM final source review

**Approved for root source freeze at the exact final hashes below. No unresolved blocking findings remain. This review grants no model resource admission.** R65 retains the active model slot; no R72 scientific worker may start until R65 closes/passes and root records the separately required matching admission.

The earlier withheld review remains unchanged at `experiments/KERNEL_SVM_RUNTIME_INDEPENDENT_REVIEW.md`. Its startup finding is resolved in this unfrozen amendment. This final review and `data/control_kernel_svm_runtime_v1/independent_review.json` are the only new review artifacts. No source, plan, old review, existing evidence, runtime or ledger was edited by the reviewer. All checks below used the private interpreter and standard library only, with no scientific imports, native arrays, labels, model calls, package installation or freeze.

## Final identities

| Artifact | SHA256 |
|---|---|
| `experiments/kernel_svm_runtime_gate.py` | `9dd8f1f4374f538407b11cc5f8c20a8f0828ba6a2820c45bb5784d876d705e8d` |
| `experiments/test_kernel_svm_source_contract.py` | `3058b5342736b84c105023afe7556abfe6c9df54760a241ec449c1985d2b9538` |
| `experiments/KERNEL_SVM_RUNTIME_PLAN.json` | `daab0291d1752adb058b7a9f6db5a32157a479a7666d650ea66cbb6b2ec9451f` |
| `experiments/NEXT_TEMPORAL_LEARNER_RESEARCH_20261031.md` | `f1408fa35d0d8594f237c92a28d13803f4f7db3001434e641369683d4e3cd512` |

## Resolved startup finding

The original `-I` launch would ignore `PYTHONHASHSEED=0` despite the environment string check. The amended supervisor now launches the bound private executable with `-B -P -S` and calls `worker_environment()`, which removes inherited `PYTHON*` variables before applying the fixed environment. Before any scientific import, `initialize()` calls `require_worker_runtime()` to check the actual private prefix, base prefix and resolved executable. It requires effective safe path, no site and no bytecode, rejects environment-ignoring mode, and requires hash randomization disabled. The subsequent exact environment check requires `PYTHONHASHSEED=0`.

I independently launched a standard-library-only child using that same executable, `-B -P -S` flags and `worker_environment()` result. The real prefix and base prefix both resolve to `data/control_isolated_python_v1/python`; the executable resolves to its `bin/python3.12`. Active flags are `safe_path=true`, `no_site=true`, `dont_write_bytecode=true`, `ignore_environment=false` and `hash_randomization=false`. Calling the actual guard succeeds, and no scientific module is imported. These observed fields exactly match the retained `active_runtime_flags.json`. This probe does not call scientific `initialize()` or fit a model.

All **16 pure tests independently pass**. The three added tests accept the intended active runtime, separately reject each wrong prefix/base/executable and each wrong effective flag, and reject inherited Python path/home/user-base/hash-seed redirects. The other 13 source-only contract, split, observer, resource and stage tests remain present. A separate source-only subreview found no remaining startup defect.

## Amendment scope and preserved scientific method

I verified all eight preserved initial files against `before_runtime_guard/snapshot.json`. AST comparison against that actual preserved source finds exactly two changed existing functions, `initialize` and `supervise`, plus the two new functions `require_worker_runtime` and `worker_environment`. Every other existing function is AST-identical, including invented data generation, chronological paired purge, estimator construction, author fit/margin observers, complete semantic state, canonical prediction, reload checks and fresh-refit comparison. The plan changes only its runtime paragraph and added runtime-guard amendment.

The previously inspected method therefore remains one fixed C=1 RBF SVC with gamma=`scale`, effective internal probability false, inner Pipeline imputation/scaling and the pinned author `CalibratedClassifierCV(sigmoid, ensemble=False, n_jobs=1)` path. Its three chronological paired folds apply the 250-actual-frame purge and produce three genuine OOF margin calls followed by one final full-context fit. All three inner fitted pipelines, final pipeline, sigmoid and complete calibrated-model state are retained and checked; fresh-process rebuilds compare semantic learned state, OOF margins and canonical probabilities. The exact 166-column schema, 600/3,000 paired contexts, 7,500 queries, 512-row canonical padding/swap averaging, single numerical thread, four named stages and 300/300/900-second plus 6-GiB bounds are unchanged.

Both retained licenses and the five pinned author source files remain verified through `source_contract()`. The SVC, calibration, preprocessing and query/replay mathematics were not changed to resolve this execution issue.

## Bound evidence

| Evidence | SHA256 |
|---|---|
| `data/control_kernel_svm_runtime_v1/source_preflight.json` | `2d03f40523984a714e679101de71cd241baefe6818b89745c12511dfee60658d` |
| `data/control_kernel_svm_runtime_v1/active_runtime_flags.json` | `329bcdc310d6e0a7b82398f8c9619c4c3960ac9cb7dc28d0acb3b111ffa8032b` |
| `data/control_kernel_svm_runtime_v1/before_runtime_guard/snapshot.json` | `fa38510fd146a6af61b5e6105c0badedc893120906bb8fe5f7188ade94990009` |
| `data/control_kernel_svm_runtime_v1_pure_runtime_guard.log` | `1f527ba673760426d48e5846f5ca28789d838412582b2d387d304d1db1a3159a` |
| `data/control_kernel_svm_runtime_v1_preflight_runtime_guard.log` | `89cb1ddb6224b8a606d87d8da7b9c3c95005c092c1572e9e93508f62360b9be3` |
| `data/control_kernel_svm_runtime_v1/source_metadata.json` | `474061797e8246c76be4f9e95fc0c9b1b32dae5b19dbf3cecb5cdd13d3b858b5` |
| `data/control_kernel_svm_runtime_v1/runtime_manifest.json` | `25117aa192d4625e3cdbc45ae45cab992508fb1ce4beabc1f768935b3492e839` |
| `data/control_kernel_svm_runtime_v1/runtime_symlinks.json` | `cd3043819dd05f0204046e5e85007b5e2f71d3939974efd7197a6b4ce337b9c1` |
| Earlier withheld review | `58d9814ef658e6159e8f61978f8e2577d4275a86b7dd282f3d5a3e543f6bed84` |

The current preflight's exact source/plan bindings match the final identities above. I independently repeated `source_contract()` and `verify_runtime()`: all **9,770 significant runtime files** and **13 symlink records** match the complete regenerated inventory, including private Python/stdlib/acquisition and the explicitly traversed read-only package target. No scientific package entered `sys.modules`.

At review completion, no R72 `protocol_freeze`, `resource_admission.json` or `runtime_assessment` exists. The new review JSON records source-freeze approval only and explicitly denies model admission. Root must freeze these reviewed bytes separately; the existing admission checks continue to require a matching freeze hash, R65 closure/pass and stopped other model workers before any scientific imports or launch.

Actual scientific import compatibility, numerical state serialization, complete four-fit runtime/memory, prediction checks and fresh-refit reproducibility remain unmeasured until that later admitted invented gate. No native possession result or improvement is claimed, and no native/full candidate or promotion is authorized by this review.
