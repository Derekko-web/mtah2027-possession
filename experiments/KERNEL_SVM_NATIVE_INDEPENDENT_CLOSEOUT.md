# R73 native kernel SVM — independent closeout audit

The closeout audit passes. The fixed SVM method remains rejected: all three protocol quality conjunctions fail, no method is eligible, and selected8 remains selected. This audit creates only this report and its new JSON record.

All nine retained official reports were independently recomputed from saved OOF labels and released public labels at frames 11185–18684 (7,500 rows, 17 possessions). Every official metric field agrees within absolute tolerance 1e-12. Public labels were padded with X in memory to reproduce the official 199,319-row report fields; no full candidate CSV was read or built.

| Protocol | Method | SpellF1 | MacroF1 | Mean fold SpellF1 | IoU≥.5 matched / spurious / missed | Short matches |
|---|---|---:|---:|---:|---:|---:|
| buffered_minutes | selected8 | 0.741176470588 | 0.870832969753 | 0.758888888889 | 13 / 4 / 4 | 3 |
| buffered_minutes | SVM | 0.697142857143 | 0.886596533559 | 0.708666666667 | 12 / 6 / 5 | 2 |
| buffered_minutes | R37 | 0.754285714286 | 0.876330458544 | 0.751206349206 | 13 / 5 / 4 | 3 |
| purged_spells | selected8 | 0.685714285714 | 0.871604353334 | 0.718000000000 | 13 / 5 / 4 | 3 |
| purged_spells | SVM | 0.633333333333 | 0.886621511673 | 0.658888888889 | 12 / 7 / 5 | 2 |
| purged_spells | R37 | 0.720000000000 | 0.879388968700 | 0.727206349206 | 13 / 5 / 4 | 3 |
| buffered_100_seconds | selected8 | 0.670270270270 | 0.844382596575 | 0.651515151515 | 13 / 7 / 4 | 3 |
| buffered_100_seconds | SVM | 0.655555555556 | 0.858106190334 | 0.644040404040 | 12 / 7 / 5 | 2 |
| buffered_100_seconds | R37 | 0.691891891892 | 0.851308828888 | 0.701144781145 | 13 / 7 / 4 | 3 |

All nine shared score rows, 39 fold rows and 189 segment rows also agree, including six short live spans per method and protocol. Saved selected8 labels exactly match its dead-trim confirmation cache; saved R37 labels exactly match its original representation screen cache. Each OOF table has the exact public frame order and released truth. The SVM retains selected8’s D and X masks exactly.

Every selection flag was independently recomputed:

| Guard | buffered_minutes | purged_spells | buffered_100_seconds |
|---|---|---|---|
| primary | fail | fail | fail |
| mean_fold_primary | fail | fail | fail |
| macro | pass | pass | pass |
| short_iou | fail | fail | fail |
| prior_matched | fail | fail | fail |
| prior_short_recall | fail | fail | fail |
| no_extra_spurious | fail | fail | pass |
| no_extra_missed | fail | fail | fail |
| D_recall | pass | pass | pass |
| false_D | pass | pass | pass |
| incremental_primary_gain | fail | — | — |
| strict_historical_best_gain | fail | — | — |

Across all three protocols, the SVM loses previously matched GT possession 12, A at frames 15799–15888 inclusive (90 frames), and gains no previously unmatched possession. Short IoU≥.5 matches fall from three to two. It also changes the 33-frame H span 15766–15798 from selected8 recall 1 to recall 0, failing prior-short-recall preservation. The JSON retains every GT match flag, one-to-one match pair, short-span recall/IoU, score row and fold score.

The buffered SVM score .697142857143 fails the selected8 gain requirement and the strict strongest-observed ceiling 136/175 = .777142857143. That TabPFN observation remains a closed method because of its 900-second worker limit and spurious-match safeguard failure; it does not authorize promotion or rescue.

The independent artifact review streamed 11,304 distinct files and found no hash discrepancies. Checks include 578 stage-cache entries, 97 input entries, 718 implementation/source entries, 579 method-failure evidence entries, all 17 protected artifacts, all 9,760 runtime files, the technical-gate map and all 19 symlinks. All 13 supervisors exited 0 without abort; 481 worker-specific hash checks included 377 mapped worker artifacts plus job/supervisor evidence.

All 52 recorded SVC fits have the required three-inner-plus-final roles. The review independently recomputed 39 chronological inner fold assignments from retained frame-index metadata and checked paired roles and the exact 250-frame purge. Successful statuses, disabled internal probability fitting, fit-state semantic hashes, final raw-input digests and final/calibrator state composition were coherent. Fresh-replay flags were correctly false with null parents.

All 26 calendar operation records have consistent UTC text/epochs, the correct two-operation sequence and supervisor containment. UTC and monotonic limits both pass: maximum worker durations are 8.625786 / 8.625756 seconds against 900; maximum operation durations are .386165 / .385453 seconds against 300. Maximum retained supervision peak memory is 244,711,424 bytes. The standalone public-label audit peaked at 107,560,960 bytes, below 1 GiB.

The frozen 1,612-row ledger prefix is byte-for-byte preserved (613,559 bytes; SHA-256 563548a3b551deed8b385f22153fdd9129470e5257b79e83e6992a93f4fc9a7d). Exactly nine source-score projections produce 1,621 rows with round73_kernel_svm / prospective_fixed_native_kernel_svm metadata, correct source/digest and blank nonprojected fields. Protected selected-model artifacts match their frozen hashes.

The screen manifest SHA-256 is faaa7ea6cd49b37ddb1d9cded70a0593cb620359cb6c1ec4af09deff782f5944; scores SHA-256 is abdcf28dee47bf79a1ede989298d939edfa557daa395e2807f5e24c67bd06ee5; method-failure SHA-256 is b87039e918a10c5bc075a31b5a5a0eda39aff7533c9f4dad8a525d90e5cd8313.

The public scorer used the private head interpreter with NumPy/Pandas only and no learner imports. The parallel artifact review used the same private interpreter with standard-library-only inspection. No native feature arrays, model deserialization, fits, forwards, private/manual/sealed data or full candidate CSV were used. Recorded fit/query/reload/repeat/raw-payload and semantic-state evidence was verified from retained artifacts; this is not a new execution or native reconstruction. The reused five-minute public window supplies dependent development evidence and cannot establish full-match or private accuracy.

The method-failure record remains closed with no retry or partial-fold rescue. Seeds 43/44, fresh replay, full construction, promotion and submission were neither run nor authorized. Before writing, both new output paths were confirmed absent and unbound in the R73 frozen maps and a broader search of existing manifest/hash/implementation/closure/protected/runtime/plan/freeze/review maps under experiments and data.

Full machine-readable results: `data/control_kernel_svm_native_v1/independent_closeout_audit.json`.

