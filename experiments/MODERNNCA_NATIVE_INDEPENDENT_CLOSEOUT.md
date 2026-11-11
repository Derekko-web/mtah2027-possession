# R68 ModernNCA independent native closeout audit

**Verified rejection.** The retained native result is complete and technically successful, but every protocol fails the frozen possession-quality conjunction. No candidate is eligible; no confirmation, 39-fit replay, full candidate build or selected-model change is authorized.

This independent audit reads persisted artifacts rather than relying on the closeout summary. It uses the isolated head interpreter with `-B`, imports only NumPy/Pandas through the standalone official metric module, performs no learner imports, native fits or model calls, and peaks at 98.0 MiB RSS. Frozen source/input/runtime trees, existing reports and the ledger were not edited. The only deliverables are this new report and `data/control_modernnca_native_v1/independent_closeout_audit_20261030.json` (SHA256 `b74da591a3ff24201cce936c8814bb13e9074211d881a9d70878631b1415eadd`).

## Actual retained execution evidence

The exact 5/5/3 worker set contains 13 distinct workers. Every supervisor has exit 0, `aborted=null`, successful status and matching summary/stdout/stderr hashes. All 13 worker artifact maps match their actual files; the stage manifest binds exactly all 370 retained files, and the closed-method seal binds those files plus the stage manifest (371 entries). All job payload SHA256 bindings match the four correct own-fold paths; features, probabilities and checkpoint bytes were hashed without deserializing them.

Every worker recipe declares final 200 epochs, float64 CPU2, 166 features and the fixed row/batch counts. Every actual training log contains the complete ordered epoch sequence 0…199; every retained epoch-loss array has exactly 200 finite values, and every minibatch-loss array has exactly 200 × declared batches-per-epoch finite values. No worker is aborted or incomplete. The longest fit is 135.919138 seconds; largest worker high-water RSS is 2.874023 GiB. All resource checks pass. Normal-screen `independent_refit_exact=false` is expected: this audit does not claim the unperformed 39-fit replay.

The frozen plan bytes and all map entries match current files: implementation 702, input 95, protected selected artifacts 17, technical-gate evidence 118, isolated head runtime 9760, and source closure 691. All frozen symlink targets also match. Stage source (702) and input (95) maps verify. These are streamed file hashes, not fresh execution of R69 or the native workers.

## Public-label recomputation and selection

Only released frames 11185…18684 are parsed from label/scorer CSVs. Each OOF table has exactly those 7,500 ascending frames and labels. The official metric module recomputes all nine model/protocol reports with bootstrap disabled. Every public-computable report field matches, including scored confusion, per-class metrics, five-IoU spell matching, transitions and counts. Full-file ungraded X padding is excluded from computation; its stored report accounting is checked separately. No full-length CSV is loaded or built.

All shared score fields, 39 fold rows and 189 segment rows independently recompute, including short-possession recall and best same-label IoU. Official comparison counts and scores also agree.

| Protocol | Selected8 spellF1 | ModernNCA spellF1 | R37 diagnostic spellF1 | Short IoU≥.5 matches, S8→new |
|---|---:|---:|---:|---:|
| buffered_minutes | 0.741176470588 | 0.708571428571 | 0.754285714286 | 3→2 |
| purged_spells | 0.685714285714 | 0.491428571429 | 0.720000000000 | 3→1 |
| buffered_100_seconds | 0.670270270270 | 0.616216216216 | 0.691891891892 | 3→2 |

Every gate boolean is recalculated from public labels, frozen folds and retained predictions. The failed checks are:

- `buffered_minutes`: primary, mean_fold_primary, short_iou, prior_matched, prior_short_recall, no_extra_spurious, no_extra_missed, incremental_primary_gain, strict_historical_best_gain.
- `purged_spells`: primary, mean_fold_primary, macro, short_iou, prior_matched, prior_short_recall, no_extra_spurious, no_extra_missed.
- `buffered_100_seconds`: primary, mean_fold_primary, short_iou, prior_matched, prior_short_recall, no_extra_spurious, no_extra_missed.

D recall and false-D checks pass on all three protocols. MacroF1 passes for buffered minutes and longer blocks, but fails purged spells. The main macroF1 of 0.871503457165 does not offset the possession failures. Changed-frame counts are 472/767/547. `selection.json` exactly agrees: `passed=false`, `eligible=[]`, `selected=null`. `method_failure.json` is closed, has no retry or partial-fold rescue, and binds the complete screen. No confirmation or replay stage/final marker exists.

## Selected8 and ledger preservation

For all three public protocols, `current_seed42` is identical to the retained selected8 `motion_dead_trim_seed42` OOF source, and the R37 diagnostic is identical to its retained pretrained OOF source. The ModernNCA D mask and X mask are exactly the selected8 masks (public X count 0). All 17 protected selected-model artifact hashes remain exact.

The ledger has 1,612 rows. Its 1,603-row frozen prefix remains byte-for-byte exact (610,472 bytes; SHA256 `1f912def73c181a6d3880868661abc4646f3844fbb26435e4bbff08a25f1508c`). Each of the nine appended rows matches every column of the declared projection from `screen/scores.csv`, with the correct round, record type, source path and source SHA256. The entire current ledger is also unchanged across this audit. `screen_closeout.json`'s manifest, score, closure and prefix bindings independently match.

## Limits

This establishes retained artifact integrity, complete recorded native execution and an independently reproduced public rejection. It does not repeat the fits, deserialize model tensors, reconstruct feature pools or re-run decoder inference. Worker repeat/reload/state invariance results are verified as successfully sealed evidence, not newly executed here. The 17 adaptively reused public possessions remain correlated development evidence and establish neither private/full-match improvement nor an optimum. The higher-score goal remains active.
