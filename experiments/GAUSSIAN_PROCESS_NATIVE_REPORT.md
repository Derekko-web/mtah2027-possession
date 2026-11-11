# Native Gaussian-process continuation — completed, rejected

The interrupted Gaussian-process work is closed with a complete public-validation result. All 13 original folds completed successfully: 13 actual author GP fits and 26 observed, genuinely converged posterior calculations. The method fails the fixed quality conjunction on all three protocols and is not promoted. The broader higher-score objective remains unachieved.

| Protocol | GP spellF1 | Selected8 spellF1 | GP macroF1 | GP / selected short matches |
|---|---:|---:|---:|---:|
| buffered_minutes | 0.715151515152 | 0.741176470588 | 0.881910 | 2 / 3 |
| purged_spells | 0.678787878788 | 0.685714285714 | 0.881806 | 2 / 3 |
| buffered_100_seconds | 0.637837837838 | 0.670270270270 | 0.857665 | 2 / 3 |

The GP improves frame macroF1 but recovers fewer whole possessions, including one fewer short possession on each protocol. It fails primary score, mean-fold score, short IoU, preservation of previously matched possessions and short recall. The longer-block protocol also adds a spurious possession. Main-protocol spellF1 is below both selected8 and the prospective strict historical diagnostic ceiling 136/175 (0.777142857143).

## Completed handoff

The prior frozen GP v1 failure and corrected v2 four-stage result remain unchanged. A separate read-only standard-library audit verifies all prior worker/source/runtime/artifact/log identities, actual convergence, full-state digests, normalized probabilities, exact team-swap average and 14 fresh semantic comparisons. It verifies four actual technical fits/eight posterior calculations and conservative observed memory 993,312,768 bytes. See [runtime evidence audit](../data/control_gaussian_process_runtime_v2/actual_independent_review.json).

The native draft was unfinished and never frozen or run: it contained invalid syntax and residual SVM calibration calls/counters. The original three draft sources, plan and five metadata files are preserved byte-for-byte under `data/control_gaussian_process_native_v1/before_actual_binding/`. The completed native integration delegates the unchanged frozen GP fit, query, convergence, code identity, semantic state, reload and nanosecond clock primitives. A scoped public constructor sets the matched job seed; the defining author estimator class and numerical method stay intact. A stable module name supports the local pickle reload audit.

Source-only preflight passes 27 contract/adversarial test cases. Outer context/feature/query/guard functions remain AST-identical to frozen R73; the controller retains the exact 13 folds, full +/-79-frame owntrain footprint, 250-frame exclusion, 5Hz live H/A targets, paired augmentation, 166 fixed features, all six selected8 slots and D/X overlay. The source review is a root continuation review supported by automated checks; no separate reviewer agent is claimed. See [source review](GAUSSIAN_PROCESS_NATIVE_SOURCE_REVIEW.md).

Sources and plan were frozen and verified before separate native admission. The seed42 run finished with exit0. Every worker passes its exact state, raw payload, repeat, symmetry, checkpoint reload, one-thread and resource checks. Maximum worker UTC time is 11.707925 seconds; maximum complete fit/query UTC operations are 1.817636/1.641216 seconds. Conservative observed peak is 611,713,024 bytes (583.375 MiB). Every UTC and monotonic operation remains below300 seconds and every worker below900 seconds; the memory cap is6GiB.

A separate read-only closeout independently recomputes all9 official headline scores, all39 per-fold score rows, complete segment coverage/recall/short-IoU diagnostics, all selected8 guards and the stronger historical ceiling. It verifies all worker/stage/failure hashes, exact D/X preservation, original source/runtime bindings and selected artifacts. No model is unpickled or refit by either audit. See [closeout evidence](../data/control_gaussian_process_native_v1/screen_closeout.json).

Nine method/protocol rows append to the experiment ledger: 1,621 to1,630 rows. All previous bytes remain an exact prefix with SHA256 `bf887fc0b9b2df94c0d0c431ba53976999d79f62ae899a74235d05ad4e9e7760`. The nine rows include selected8 and R37 diagnostic comparators; they represent one new fixed GP method, not nine independent experiments.

## Outcome and preservation

`eligible=[]`, `selected=null`; the fixed method is closed with no retry. Its failed screen does not authorize seed43/44 confirmations, actual fresh replay or a full build. The existing selected8 CSV remains SHA256 `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`. All17 protected selected artifacts and source links verify unchanged. No new candidate is selected.

These scores use the same adaptively reused 7,500 public labels and17 possessions. They do not establish full-match or private-test accuracy. New research would require a separately justified prospective method; this completed GP result must not be retuned or relabeled as a success.

Scores SHA256: `84f56463202d7c913370d5d10ae867e0ab08bc34189dd7917363b3ac143cb05e`. Screen manifest SHA256: `d819b70438928af91bee2b4b8d5988b3f908171b921e3f32073e789ca418c0e8`. Closeout SHA256: `7d74956e7b6f001301397e7275b3c5219f16f9e0ac5b100aa4e5a9b033c4ae28`.

## Reproduction and audit entrypoints

- `data/control_isolated_python_v1/head_runtime/bin/python -B -I -S experiments/gaussian_process_native.py --pure-tests` reruns source-only fixtures.
- `data/control_isolated_python_v1/head_runtime/bin/python -B -I -S experiments/audit_gaussian_process_runtime_v2.py` checks retained invented-data evidence without writing to its sealed round.
- `data/control_isolated_python_v1/head_runtime/bin/python -B experiments/audit_gaussian_process_native.py` recomputes and validates retained public screen evidence. The CLI deliberately refuses to overwrite existing closeout evidence; use its `audit()` function for another read-only inspection.

The original native training command is preserved in the frozen plan and sibling run log. Its retained closed-method guard deliberately prevents retraining or overwriting these results.
