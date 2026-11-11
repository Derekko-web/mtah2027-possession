# R65 isolated-runtime source review — execution and guards only

Independent source review **passes for the execution-only amendment**. This is
not approval to freeze or execute native models. Actual R70v2 all-four-stage
evidence and its final plan-bound metadata remain pending; the current
`technical_prerequisite.actual_all_four_stage_summary_sha256` is null and its
metadata map is empty. A separate final metadata audit is required before
freeze approval.

Reviewed source identities:

| File | SHA256 |
|---|---|
| `experiments/tabpfn_native.py` | `7321c6cb408cccd143f941384fbcae766bf91bd18838d869757e0a135596dd6b` |
| `experiments/research_tabpfn_native.py` | `526cd5e0cde3b4f7436be996a1d3ec3dff6629f07b6af372951f41f9e938198d` |
| `experiments/test_tabpfn_native.py` | `ea8a2eb0bbfbf6fa8bdf70a2c2da649b499588a56e151cf01960e24fa88a2283` |
| `experiments/TABPFN_NATIVE_PLAN.json` | `937292d1d0e8244290e2aa2471e4b438d9a8bf423d1b02f553e6657d5b3e2473` |

## Comparison and scientific invariance

The comparison baseline is the retained15-file unfrozen draft snapshot under
`data/control_tabpfn_native_v1/before_isolated_runtime/`. Every file verifies
against its snapshot manifest, SHA256
`1941a1df48f1a5cdadaafe1eeb9593797b34e6b84cb3d9d825b70633c684eecb`.
No historical frozen source or dependency edit is part of this amendment.

Sixteen of the20 original core function ASTs are completely identical. The
four changed functions are `technical_gate`, `worker`, `preflight` and `freeze`;
their differences concern prerequisite identity, runtime admission and ledger
provenance. `require_private_runtime` is the only added core function.
Removing the worker's one new runtime-guard call restores its complete
original AST. Replacing the controller's one runtime guard restores the
complete original `run` AST; its other three functions are AST-identical.

The plan's model, features, context, imputation, queries, integration,
initialization/immutability, quality gate, confirmation, replay, historical
record and protocol fields are exactly equal to the draft. The fixed166-column
interface,13 folds,5Hz owntrain selection, complete79-frame footprint,
250-frame outer exclusion, paired context of at most3000 rows, original
imputation, all-frame canonical512 queries,8 V2 views, temperature.9,
original/swapped probability average, selected live overlay, strict historical
record and all selected8 guards, all9 confirmation conjunctions and actual39
fresh reconstruction proofs remain unchanged. No numerical/model/decoder
change was found.

## Execution identity and provenance

The prerequisite now binds the exact R70v2 controller, plan and evidence round:

- `tabpfn_mps_isolated_runtime_gate_v2.py`, SHA256
  `16804adff92ba4308dddb693faa53f1763fb8cad4686be18487feaa9a4a67596`.
- `TABPFN_MPS_ISOLATED_RUNTIME_PLAN_V2.json`, SHA256
  `c54e006a10909c57a341c235531bd826f269ab2c0d0ab830bd1f5ccb1a896f73`.
- `data/control_tabpfn_mps_isolated_runtime_v2`.

The technical verifier preserves the strict four-stage order, successful
supervision, complete artifact/log hashes, operation/resource bounds,
allocator-release checks and independent rebuild byte comparisons. Its
prediction predicate correctly comes from the verified original R67 module
through `module.previous().is_prediction`; R70v2 itself does not export that
predicate. R70v2 preservation verification replaces the direct old R63
verification call. The old strict verifier remains unchanged and still rejects
the obsolete shared executable. All21 direct R63 frozen implementation
bindings are present unchanged in R70v2's preservation mapping.

The controller requires the private head prefix, private base prefix and
resolved executable identity before scientific imports. The model worker
requires the private base runtime before runtime-path setup, scientific imports
or NumPy payload loading. Its existing `-P -S -B` flags and fixed environment
checks remain intact. The dynamically loaded R70v2 core supplies the private
model executable; its repaired dependency target and scientific functions stay
the same. The frozen official scorer already launches `sys.executable`, so the
private head controller also routes evaluator subprocesses to the private head
runtime. Plan commands preserve inherited bytecode suppression.

The controller-runtime map retains all8575 non-obsolete old bindings, removes
exactly the two declared old executable aliases, and adds1185 private
runtime/acquisition/configuration bindings for9760 total. Every current bound
runtime hash verified. Both original source links remain unchanged;17 private
links are added, and all19 link targets verify. All17 protected-artifact
bindings remain unchanged. The source closure retains all672 old entries and
adds all16 snapshot files/manifest entries.

The previously unbound preparation ledger is now plan-bound, checked as a
prefix before the technical prerequisite, and included in the later freeze
closure. Its613559 bytes contain1612 rows and hash
`563548a3b551deed8b385f22153fdd9129470e5257b79e83e6992a93f4fc9a7d`.
The current ledger begins with those exact bytes.

## Pure verification and remaining boundary

The retained final pure-test log reports27 passing tests, including the20
existing protocol/replay tests, separate wrong-base and wrong-executable
rejections, private head/base identity cases, and invented failed, incomplete
and duplicate technical-stage records. The invented prerequisite fixture invokes
the real technical conjunction with a mocked source-preservation verifier and
rejects each invalid case before stage payload access. No scientific import or
model execution is part of these fixtures. Log SHA256:
`b378b7579dfd0349298f608bb429b8001a4246980f2bd9ee38e77ea0b5e71c77`.

No further source/guard finding remains from this review. Actual R70v2 stage
results, complete final metadata binding, final pure preflight, independent
freeze review and successful frozen verification are still required. This
review made no native-array reads, scientific imports, model calls or freeze,
and changed only this new review document.
