# R73 kernel SVM native source review

**Source-only draft reviewed. Freeze approval is withheld while R72's actual four-stage result remains pending/null.** No additional blocking implementation defect was identified in the reviewed draft. This is not an R73 preflight pass, technical pass, resource admission, native execution or performance result. R65 retains the active model resource slot.

This independent review inspected the complete native worker, controller, pure tests, plan, preparation records, inherited source/input/runtime metadata, selected inference interfaces and frozen R72 numerical primitives. A separate source-only replay/prerequisite subreview found no blocking defect in those paths. All checks used the private head interpreter with `-B -I -S` and the standard library only. No scientific module, native array, model or checkpoint was loaded; inherited binary files were streamed only for their declared hashes. No preflight, freeze, scientific import, fit or prediction was attempted. The reviewer writes only this new report and changes no source, plan, old report, frozen file, runtime or ledger.

## Reviewed identities

| Artifact | SHA256 |
|---|---|
| `experiments/kernel_svm_native.py` | `738eea39b36a37387ad9773f20db78ac5846bd6bc1fabe8f8d4fbc21613f6dde` |
| `experiments/research_kernel_svm_native.py` | `a77315b840686feab7d729342aa32433a1236d854c95de650f210adebebb2a0e` |
| `experiments/test_kernel_svm_native.py` | `c0322a487486108971974e05565177446817b13cf916ca3f23b92d1cea6b4ed1` |
| `experiments/KERNEL_SVM_NATIVE_PLAN.json` | `03fbd65015e38825e9719a8267eb03ad7fb77d3f990aee5a3b102d6206b47a48` |
| `data/control_kernel_svm_native_v1/source_preparation.json` | `49119b614871e970bea377df35bf3c08a73348e58de20be2857acb661293e3b9` |
| `data/control_kernel_svm_native_v1/PREPARATION.md` | `ee6d583135b9309112e14fa4fa4418c834e55cc834237bb4d5664b783c7e4e50` |
| R72 source freeze | `5e7d2a6df71dde7782d17b5d3899aab259efd3722289a7ebf467cfa9580fe3b8` |

## Findings and present admission boundary

1. **Required technical prerequisite remains unsatisfied.** The plan has `status=PENDING_ACTUAL_ALL_FOUR_STAGE_PASS`, a null actual-summary hash and an empty technical-evidence map. `technical_gate()` checks this before loading the R72 module. Native verification reaches it first; preflight reaches it before pure checks or any output, and freeze calls that gated preflight first. The pure pending fixture proves that the blocked paths create neither a preflight-pass artifact nor a freeze. The real R73 directory likewise has no `source_preflight.json`, `protocol_freeze`, `resource_admission.json` or screen. No independent freeze-approval JSON is created by this review.
2. **No additional source blocker found in context selection, calibration, selected inference, replay or preservation.** These findings concern the exact draft hashes above. Root must bind a genuine completed R72 result and the retained technical proof map, then review the resulting final plan/source bytes and run the actual prerequisite-gated preflight before any separate freeze/admission decision. The current review does not preapprove those future changed bytes.
3. **Timing is prospective monotonic elapsed time, not an observed UTC elapsed-time result.** R73 `supervise()` uses `time.monotonic()` for its 300-second active-operation and 900-second worker bounds (lines 358–383). Its delegated R72 `operation()` also starts and completes a monotonic interval. The technical-proof validator checks the stored operation/worker durations; it does not reconstruct UTC start/end intervals. No R73 operation has run here. This source review makes no claim that UTC elapsed-time caps were met, and makes no claim about timing, progress or restart of the active R65 process. Any separate root investigation of clock/suspend semantics remains outside this source-only result.

## Scientific interfaces and calibration

The controller calls prerequisite/freeze verification before scientific imports. It retains the exact original 102 ordered columns plus fixed 64 R37 latent columns and checks the complete 166-column order. It uses the same 13 outer folds: five buffered-minute, five purged-spell and three longer-block folds, covering every public frame 11185…18684 once per protocol. Reading only split metadata confirms the actual query groups are 1,500/1,500/2,500 frames respectively, consistent with the worker's canonical-query requirements.

`context_frames()` applies the unchanged 250-frame outer exclusion, ascending public own-training indices, global 5-Hz phase and full real inclusive ±79-frame footprint before the H/A label filter. Empty or oversized contexts reject without resampling. Raw original rows precede their exact swapped copies; targets are H0/A1 and then complemented. The worker validates dtype, C layout, balanced paired labels, original center identity and exact swap bytes. Only five payloads are allowed: raw context, own-training targets, original queries, swapped queries and original context frame centers. There is no query-label payload, D-class target or outer imputer/scaler. Masking labels outside own-training must reproduce the context/target/record. Base centers are included in the retained fitting record and five-payload proof.

The worker delegates the reviewed frozen R72 author estimator, complete three-inner-plus-final fit recorder, canonical predictor, full semantic state and local reload checks. The only estimator parameter amendment sets `svc__random_state` to the fixed head seed before fitting; the declared disabled internal SVC probability fit and explicit calibration remain unchanged. Three actual chronological inner groups keep mirrored mates together and purge by actual frame distance. Each inner imputer/scaler fits inside its own Pipeline clone. The final classifier fits the complete permitted outer context, and sigmoid calibration consumes genuine inner-held-out margins. All four actual fitted Pipelines, inner/assembled OOF margins, sigmoid and complete model state are retained and checked. No parent estimator supplies a fresh fit.

The controller requires all six selected probability slots. The unchanged `factorized_latent.decode_owner()` copies slot0, composes conditional H/A ownership with its original live mass, retains the complete original slot0 vector where latent support is unavailable, passes the other five slots unchanged, and applies the fixed final live overlay. That function asserts both D and X masks exact; the controller also compares the selected reference with its saved seed/protocol OOF labels. The seed42 R37 comparator is reproduced only as a diagnostic and is removed from eligibility.

The main screen must strictly exceed the historical comparable spellF1 `0.7542857142857143`, in addition to every required selected8 primary, mean-fold, macro, short-IoU, prior-match/short-recall, missed/spurious, D-recall, false-D and buffered +.01 check. Seeds43/44 require a passing seed42 screen; all nine exact seed/protocol conjunctions must pass. There is one fixed SVM method and no fallback, best-seed selection or partial-fold rescue. Deterministic heads across seeds are not treated as independent evidence.

## Replay and future technical evidence

The future R72 validator requires the exact ordered moderate-first/rebuild and full-first/rebuild identities, matching frozen source/plan, exit0/noabort, complete stage conjunctions, summary/supervision/log/artifact bindings, four actual fit records, each individual learned-state witness and complete semantic fresh-refit equality. A source freeze or four arbitrary true flags cannot substitute for the missing actual result.

The native finalizer requires exactly the distinct 5/5/3 worker set in every original and replay stage for seeds42/43/44. It validates each job's own five paths and same-seed/protocol/fold original parent, original/replay mode, full four-fit provenance, fresh-context flags, successful supervision, operation/resource durations and every retained worker/stage artifact hash. It compares all five raw payloads, fitting archive and context provenance, plus complete inner/final learned state, calibration, inner/assembled OOF margins, recipe, original/swapped/averaged probabilities, retained repeats, classes and payload immutability evidence. It counts 39 independently replayed workers and 156 actual classifier fit records, rather than inferring proof from directory names.

`replay_complete()` additionally requires exact shared/fold/segment/official-comparison/selection files and public OOF labels/owner probabilities across each original/replay pair. Missing, renamed, self-parented, cross-seed, incomplete-fit and altered/resealed evidence is rejected by the literal fixtures. Full build remains false even in a valid completion marker. A later full-match parity, double construction, validator and independent review remain outside this draft.

## Preservation and verification

All **29 pure fixtures independently pass** with no scientific imports. They cover the prerequisite hard stop, private runtime identities, real ±79/250-frame context rules, empty/oversized rejection, raw inner preprocessing placement, exact R65 pure-interface AST identity, fixed queries, all nine quality pairs and complete 39-worker/156-fit positive and negative replay proofs. The fixtures use invented literal bytes and JSON; they deserialize no model or scientific array.

I independently verified actual file hashes for the 690-entry source closure, 96 inputs, 17 protected selected artifacts, 9,760-entry private/runtime/package map, both five-entry metadata/basis maps and the three prepared sources. All 19 symlink targets match. The inherited bindings contain no active R65 screen/runtime-assessment artifacts and no mutable current full-ledger hash. The current ledger retains the exact 1,612-row preparation prefix: 613,559 bytes, SHA256 `563548a3b551deed8b385f22153fdd9129470e5257b79e83e6992a93f4fc9a7d`. The source-preparation report, plan and metadata hashes also match their recorded values.

| Retained metadata | SHA256 |
|---|---|
| `source_closure.json` | `d2ac081edeee4bce6ce220f2000e71d4d36d227830b137f4bd59617f63fa6975` |
| `head_runtime.json` | `7a072b2f8aebb63b5e18da30ba3d9ecad8e5ca4fac8393809cfc71488da42a41` |
| `protected_artifacts.json` | `e750f9d7cb2e91230867e2052e04741e7914e5a24b59bbd67caffbebf6f352fc` |
| `symlinks.json` | `25c4d3706d676c3769583eec31475b4e71fbc51d1de9341c4d24f926e3c8dd37` |

These are source and retained-identity checks only. R72 scientific compatibility and actual four-stage execution, R73 numerical/resource behavior, native quality, full-match parity and private/full-match accuracy are not established. The 17 adaptively reused public possessions remain dependent development evidence; no improvement or optimum is claimed.
