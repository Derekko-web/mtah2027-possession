# Gaussian-process runtime v2 independent source review

**Approved for root source freeze at the exact v2 hashes below. No unresolved blocking findings remain. This review grants no model/resource admission, native processing or technical-pass claim.** The frozen v1 gate remains failed and ineligible. Root must separately freeze/verify v2 and record the required matching v2 admission before any scientific worker starts.

This review inspects a separate prospective execution-admission correction. It changes neither a learner nor a setting selected from model performance: the retained v1 exception occurred before its first author `GaussianProcessClassifier.fit` or posterior invocation. V1 had generated and preprocessed invented data; it did not produce a fitted GP or possession result. Its source, plan, frozen records and failed output remain unchanged. Only this new report and `data/control_gaussian_process_runtime_v2/independent_review.json` are written by the reviewer.

## Reviewed identities

| Artifact | SHA256 |
|---|---|
| `experiments/gaussian_process_runtime_gate_v2.py` | `1d9db995f75eeebe4646360c2103de4b2135ffdde167407aa7c9dac94ee51095` |
| `experiments/test_gaussian_process_source_contract_v2.py` | `7675ba230c349b5f4ff2d3df47b5d8a3d7ece01464b2f503d6f46450d921468c` |
| `experiments/GAUSSIAN_PROCESS_RUNTIME_PLAN_V2.json` | `1900da26c8f0f501f86e3086b27d963a20636d77512ff34a0784ebdd187f88e8` |
| `experiments/GAUSSIAN_PROCESS_LEARNER_RESEARCH_20261101.md` | `e4b06c1c656a356aa30d409978b50850e1fbcf313b3ac9ed6a869ba8181ca224` |
| `experiments/GAUSSIAN_PROCESS_CODE_IDENTITY_DIAGNOSTIC_20261101.md` | `d7e72807e2ffd6219c67198023dfb4780d1a65f6ce196f79413616ed1334d391` |
| V2 `source_metadata.json` | `13edde03b1359b9a09d1099e31528a669050fbdaae3c8cb0fa282234bfa8dc05` |
| V2 retained `source_preflight.json` | `4af3dcad37a9f1c9143d772f75ed9d7aa4345d6532f5dbde7e51fdb8e40ab956` |
| V2 `predecessor_manifest.json` | `794cd6f50295ac70c8849747a62d8b4a08e769b7b11b7071aae223ee9a7dd73f` |
| Captured diagnostic JSON | `a17d4613cadd52cd46b424761fb4d0cc5afa3094e92cf399e0a7e59be73ec825` |

V2 round-relative paths are under `data/control_gaussian_process_runtime_v2/`.

## The preserved v1 failure and narrow correction

The actual v1 top-level assessment is `passed=false`, with one moderate-first stage, exit1 and no successful worker summary. Its retained traceback ends at the original code-identity guard, before the later author-fit call. No `fit_record.json` or `posterior_traces.json` exists in that failed worker. The top-summary SHA256 is `0b392cfd0adf0e976c0ac6013eadfa698041b54767c31904e580c5dcf77a935e`; failure SHA256 is `f80ba68f92472e75717c1912dc3e55b9cef34f99919835ed725a91e22a43f439`. This review neither turns those records into a pass nor reuses the failed output.

I independently decoded the two trusted captured code objects as standard-library metadata only, without executing either. Their raw marshal payloads differ at five byte positions, yet `actual == pristine` is true and their complete canonical records match. The difference concerns reference flags/reference-table indices rather than changed code values. Raw marshal/adaptive representations are retained in a separate nonsemantic audit; they no longer serve as a strict semantic-value byte fingerprint.

One count clarification does not change that conclusion: this pinned interpreter exposes **19 noncallable public `co_*` attributes**, all retained dynamically by v2, plus `co_lines()` and `co_positions()`. The earlier diagnostic prose calls that public count20; its machine-readable values and the current enumeration contain19. The original diagnostic/report are preserved unchanged. The separate private adaptive representation is audit-only. No available public field is omitted to resolve the discrepancy.

## Complete code-value admission

`canonical_code_record()` preserves typed values for every available noncallable public code attribute, recursively supported code constants, both derived line/position sequences and the complete pinned CPython `TYPE_CODE` payload. The parser resolves reference framing while retaining all **16 serialized code data fields**, including hidden `localsplusnames` and `localspluskinds`. It preserves bytecode, constants, filename, name/qualification, flags, arguments, stack, variables/free/cell names, first line and line/exception/deprecated tables. Unsupported constants/tags, incomplete payloads and invalid references fail closed.

`require_authored_code()` requires full actual/pristine canonical-record equality, the expected stable canonical digest, Python code-value equality and no pre-existing trace. It does not replace either code object. The captured equal-code/different-marshal case passes this complete comparison; the stable canonical SHA256 is `e613674e1e2b1e0ebfb60e1a019761efcd89516c5477b6b0b595a53f9d43a952`.

The author source hash remains `90628fca68b26aac2c4f57305b6302c549952d3873e6651209ffcafb2c95ff3d`. AST anchors remain method443, iteration467, condition491 and actual break492. The observer continues to select the exact original `frame.f_code` by identity, delegate the unchanged author method once per invocation, inspect locals without altering them and restore the prior trace in `finally`. Two genuine from-zero posterior calls, finite arrays/likelihoods, contiguous iterations, a real break before100, no delta below `-1e-10`, and final delta in `[-1e-10,1e-10)` remain required.

`code_identity_audit.json` records raw marshal/adaptive hashes separately with `semantic_comparison_allowed=false`. It is locally hashed evidence but is absent from the fresh-refit semantic file list. Source/model/state/trace/probability equality remains mandatory; heap-sensitive container details cannot cause a false semantic refit failure.

## Scientific and resource protocol unchanged

I independently verified the reported AST proof: **23 numerical/data/query/resource definitions are unchanged**, including the fit body after removing exactly one old admission `if` versus exactly three new pristine-check/admission/audit statements. No scientific statement is removed by that normalization. A separate source-only subreview reached the same no-blocker conclusion.

The fixed unit-RBF Bernoulli Laplace GP, training-only preprocessing, variance-derived length scale, optimizerNone, max_iter_predict100, warm_startFalse, full semantic state including RandomState, cached no-argument likelihood getter, raw query mathematics, canonical512 padding/swap average, checkpoint reload, exact fresh refits and all data roles/shapes remain the same. There is no tolerance, kernel, precision, iteration, context, query or quality-setting change based on a learner result.

The unchanged dual UTC/monotonic complete-operation300/worker900 guards, one numerical thread, one worker, 6-GiB cap, exact four-stage order and stop-on-first-failure policy remain. New output directories prevent reusing old failed output. V2 admission additionally requires `predecessor_v1_closed_failed=true`, alongside v2 scope, R73 closed/passed, stopped other model workers and the reviewed v2 freeze hash. Source preparation/preflight/review/freeze do not supply this admission.

## Preservation, validation and limits

All **25 pure tests independently pass**. They reproduce the captured marshal false alarm without executing captured code, reject changed bytecode/constants/path/first-line/flags/name/qualification/stack/arguments/positions, expose hidden-local name/kind mutations, reject nested-code mutation and unsupported constants, and retain the existing trace, clock, resource, state and admission fixtures. Both mutation metadata and pristine author compilation remain non-executing source inspection.

I repeated source-contract and complete runtime verification under the private head interpreter with `-B -I -S`: **9,770** runtime files, **13** symlink records, **26** retained notices and four pinned author files match. Each retained notice matches its installed original. The complete **58-file predecessor/diagnostic manifest** matches the actual preserved files, including v1's frozen source/plan and failed assessment. The current v2 preflight is source-only PASS and binds all five exact `SOURCE_FILES` hashes above. No old file is edited or pass waived.

No NumPy, SciPy, sklearn, Pandas, Torch or joblib is imported by this review; no scientific array/model is decoded, no GP code is executed, and no preflight/freeze/model worker is launched. Trusted captured code objects are read as inert metadata only. At review completion v2 has no freeze, admission or runtime-assessment directory. Actual v2 scientific compatibility, convergence, runtime/memory, probabilities and replay remain unmeasured until the separately admitted gate. V1 remains failed, and no native or full-match improvement is established or authorized.
