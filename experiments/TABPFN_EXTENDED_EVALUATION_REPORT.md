# TabPFN extended evaluation — completed, current model retained

The user-authorized time extension completed all 13 original evaluation folds. The main **0.777142857143** result remains valid, and the completed purged protocol improves on the current model. The longer-block protocol falls substantially behind. The completed evaluation does not support replacing the current model.

| Protocol | Current spellF1 | TabPFN spellF1 | Difference |
|---|---:|---:|---:|
| buffered_minutes | 0.741176470588 | 0.777142857143 | +0.035966386555 |
| purged_spells | 0.685714285714 | 0.731428571429 | +0.045714285714 |
| buffered_100_seconds | 0.670270270270 | 0.576470588235 | -0.093799682035 |

## Decision follows the user’s scoring priority

The user clarified: “spellf1 is the main score that determines rankings.” A separately frozen ranking-first policy therefore treats spurious-possession counts and the other secondary quality checks as reported diagnostics. The replacement decision is based on **spellF1 itself**: the longer-block protocol declines by **0.093799682035**, or **9.38 percentage points**. Main and purged gains are +3.60 and +4.57 points respectively. Short possessions matched on the longer-block protocol fall from3to1.

The longer-block regression is not a timeout or failed numerical run. All three longer-block workers completed successfully under the extended limits. Its middle100-second fold scores0.485714 versus0.666667 for the reference; its other two per-fold scores improve. The stitched whole-possession score remains the authoritative comparison. No model setting, feature, decoder, seed or threshold was retuned after seeing these results.

All three complete seed42 protocol comparisons were required to support the matched seed43/44 confirmation stage. The longer-block score fails that requirement even under the ranking-first policy. Additional confirmations, independent full main refits and full-match construction were therefore not started. The existing selected CSV remains unchanged. This completes the requested extension and replacement assessment; it does not produce a new selected model.

## What ran and what was reused

Original R65 sources, plan, dependency trees, original15-minute failure and all291 bound predecessor files remain unchanged. The new round is `data/control_tabpfn_extended_v1/`. The nine old successful folds were reused only after fresh regeneration of their owntrain context, target, query arrays, imputer/role records and provenance matched. The failed tenth worker was never reused. It was fitted and queried again; the three previously unrun longer-block folds also ran. Thus13 logical fold models comprise9 verified historical workers and4 fresh workers.

The new tenth worker passes in451.175176 UTCseconds; its probability, checkpoint, preprocessing and scientific-recipe files match the arrays retained from the old timeout byte-for-byte. That establishes consistency of the new result while the original attempt stays failed. The five main-protocol outputs were reused and rescored; this report does not claim a fresh independent refit of all five main folds.

The exact original native worker and numerical primitives were reused. Settings stay fixed:166 features, owntrain +/-79-frame footprint,250-frame exclusion,5Hz H/A context, exact team-pair augmentation, eight-view float32 TabPFN v2, original checkpoint, MPS dispatch,512-row padded query batches, state/checkpoint/repeat tests and unchanged selected decoder/D-X overlay. Both UTC and monotonic clocks now enforce <900seconds per synchronized operation and <3600seconds per complete worker. The6GiB combined ownedRSS+fullMPSdriver cap and4GiB allocator allowance stay unchanged.

All fresh workers pass. The longest is698.882385 UTCseconds, and conservative observed peak across retained workers is4,302,553,088bytes (4.007GiB). Historical reused monotonic timings are not advertised as fresh UTC evidence. Seventeen new execution-only pure tests and five ranking-policy/source checks pass. The ranking driver’s scientific run body is AST-identical after excluding only admission, policy reporting and source metadata statements. Root reviews claim no separate reviewer agent.

## Independent audit and retained artifacts

A separate read-only audit recomputes all9 official headline score rows, all39 per-fold rows, complete ground-truth segment/short-IoU/recall diagnostics, original guards and the explicit ranking-first decision. It verifies worker, cache, log, source, runtime and predecessor hashes, all17 protected selected artifacts, exact D/X preservation and fresh dual-clock/resource records. No model is refit or unpickled by either audit.

- [Completed screen audit](../data/control_tabpfn_extended_v1/screen_closeout.json)
- [SpellF1-first assessment](../data/control_tabpfn_extended_v1/ranking_policy/screen_selection.json)
- [Independent ranking audit](../data/control_tabpfn_extended_v1/ranking_policy/screen_audit.json)
- [Replacement decision](../data/control_tabpfn_extended_v1/replacement_decision.json)
- [Original strict assessment](../data/control_tabpfn_extended_v1/screen/selection.json)

Nine method/protocol/comparator rows append to the ledger, from1,630to1,639, preserving every previous byte. These rows describe one completed extension of the same TabPFN method, not nine independent experiments. Original1630-row prefix SHA256: `4ed8fb8db340327cf60838effc7ef76f00eef9337d25bbe2a80294ff08b39418`.

Selected CSV SHA256 remains `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`. All preserved hardware/acquisition copies, source links and selected artifacts verify unchanged. No candidate was replaced, committed, uploaded or submitted. No model processes remain.

Both the favorable main result and the unfavorable longer-block result use the same adaptively reused7,500 public labels and17 possessions. They do not establish private-test accuracy. The completed evidence supports retaining the current selection while preserving TabPFN as a promising, context-sensitive research result.

Scores SHA256: `89fce2c96f23c51368dbc7f63717a9ec517ccf6a718b033aee4d204eb0f36770`. Screen manifest SHA256: `d1915423e4b5b2ad827a78ea943612a87a816ad77469cb77e3f0c3c6f0d63880`. Decision SHA256: `e406f2fe4e78f6f68782a50ba14694400c8b7f2f5df6bd21b996dec7c484fe6e`.
