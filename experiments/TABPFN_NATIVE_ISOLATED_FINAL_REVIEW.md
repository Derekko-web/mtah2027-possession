# R65 final isolated-runtime review — approved for freeze

Independent review **passes and approves the reviewed R65 sources and plan for
freeze**. Actual R70v2 all-four-stage evidence and the final R65 source-only
preflight now pass. This review performs no freeze, native processing or model
execution. Successful frozen verification and the existing resource admission
remain required before native work; every scientific and quality gate remains
unchanged.

Reviewer: `/root/r61_replay_recovery`, with an independent read-only evidence
audit by `/root/r61_replay_recovery/audit_r61_replay`.

## Final identities

| File | SHA256 |
|---|---|
| `experiments/tabpfn_native.py` | `22694bbbc08fe78994566a5f22884fa238cb8e23b7973736615fda51e1f3b739` |
| `experiments/research_tabpfn_native.py` | `526cd5e0cde3b4f7436be996a1d3ec3dff6629f07b6af372951f41f9e938198d` |
| `experiments/test_tabpfn_native.py` | `ea8a2eb0bbfbf6fa8bdf70a2c2da649b499588a56e151cf01960e24fa88a2283` |
| `experiments/TABPFN_NATIVE_PLAN.json` | `fc80e15b7812a7f7f2474c8da8327a0ad2b0504d2f2699a8ca85dcb8007e1825` |

The final core differs from the earlier source review only in human-readable
technical-gate labels and freeze documentation. The plan additionally binds
the now-completed technical evidence. All12 scientific plan fields remain
exactly equal to the preserved unfrozen draft. Sixteen original core functions
are AST-identical; removing the one new worker runtime guard restores its
complete original AST. Replacing the one controller runtime guard restores its
complete original `run` AST; its three other functions remain AST-identical.
The previously reviewed execution-only loader, private-prefix and ledger
changes remain the complete behavioral diff.

The166-column interface,13 folds,5Hz owntrain/79-frame footprint and250-frame
exclusion, paired context cap3000, imputation, checkpoint,8 V2 views,
temperature.9, MPSfloat32, canonical512 queries, original/swapped probability
average, selected live overlay, all selected8 guards, strict historical record,
all9 confirmation conjunctions and39 fresh reconstruction proofs are retained.

## Actual R70v2 evidence

The plan binds exactly117 files: every retained file in the R70v2 evidence
round plus its plan and technical report. All117 hashes verify. The actual
summary SHA256 is
`5ad010c56855e2ec9fad2b4690046b240825ca206291ab592421bd38f210022e`.
The strict stage order is moderate/first, moderate/rebuild, full/first,
full/rebuild; all four have exit0, passed=true, no abort, no monitor failure and
all recorded exactness/resource checks true. Each top-level supervision record
equals its retained stage record. All84 stage artifact hashes and all8
stdout/stderr hashes verify.

| Stage | Context fit seconds | Supervised wall seconds | Peak owned RSS + full driver GiB |
|---|---:|---:|---:|
| Moderate first | .511 | 68.372 | 1.7831 |
| Moderate rebuild | .220 | 62.711 | 1.8486 |
| Full first | 1.989 | 365.242 | 3.9148 |
| Full rebuild | 2.089 | 362.729 | 4.2652 |

Every timed operation completes within300 seconds; the maximum is116.123
seconds. Every stage stays within900 seconds and6GiB. Moderate stages retain11
operations,7 prediction flushes and16128 guarded dispatches each; full stages
retain7 operations,3 prediction flushes and6912 dispatches each. Prediction
flush ordering, exactly one synchronization/cache-release pair, unchanged live
tensors and dispatch shapes/totals verify. No model/context/RNG mutation is
attributed to cleanup.

Each moderate stage retains14 arrays and each full stage12. Across the two
fresh first/rebuild pairs, all26 array comparisons and8 checkpoint,
preprocessing, metadata-initialization and recipe comparisons are byte-exact.
An independent comparison also verifies64 numeric/raw-fingerprint/initialization
artifacts against historical R67. The only historical recipe difference is the
declared unused model-cache directory; new first/rebuild recipes are exact.
The prefix8 shape diagnostic remains unequal by at most1.788e-7, exactly as
historically classified; canonical512 repeat and fresh rebuild checks pass.

The new R70v2 verifier passes without changing any old strict verifier. An
independent invocation of the original R67 `--verify` still fails at the
obsolete shared Python executable, preserving the distinction between the
historical admission and this new private-runtime admission.

## Current provenance and source-only preflight

All9760 private/controller-runtime bindings,690 source-closure bindings,
19 symlink targets,17 protected-artifact bindings and15 retained draft files
verify. The original runtime map loses exactly its two explicitly superseded
executable aliases; all other old bindings remain unchanged. Both original
source links are preserved. The private base/head/configuration/acquisition
bindings are current. The controller and worker require their distinct private
prefixes plus the private base and executable before scientific imports; the
frozen official scorer continues to inherit the private head `sys.executable`.

The preparation ledger's613559 bytes contain1612 rows and remain an exact
prefix of the current ledger, SHA256
`563548a3b551deed8b385f22153fdd9129470e5257b79e83e6992a93f4fc9a7d`.
The plan binds it, source preflight checks the prefix before technical
admission, and freeze includes it in the implementation closure.

The actual final R65 preflight passes in14.033 seconds, binds all three final
source hashes, the final plan and the actual R70v2 summary, and records all27
pure tests passing with no scientific imports, native-array loading or model
forward. Its SHA256 is
`a3a9aa486aa6cfa8e8c1303e372578ff121e8eb766ab61c99a216b361e08d16a`.
The sibling console log ends in successful source/pure preflight and hashes
`4e4aa619e6513494551894d59f1046f49cea35253f56e54ea04d316befc4fd42`.
The independent reviewer also invoked the actual read-only technical verifier
successfully and confirmed no scientific package entered the process.

No unresolved source, provenance or technical-admission finding remains.
Approval is recorded in the new
`data/control_tabpfn_native_v1/independent_review.json`, bound to these exact
sources and plan. Byte hashing was used for integrity checks; no array was
decoded, no native scientific payload was loaded, no model was instantiated
and no freeze was performed by this review. Only this new review and its
approval JSON were written.
