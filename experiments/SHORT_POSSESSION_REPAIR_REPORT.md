# Short-possession repair — 7 November 2026

The selected TabPFN candidate now includes a conservative broadcast-contact boundary refinement. It recovers two additional short possessions at the scorer's IoU0.5 threshold, improving main SpellF1 from **0.777143 to 0.811429**. All nine matched seed/protocol comparisons improve SpellF1, macroF1 and frame accuracy; all39 clipped-fold SpellF1 and macroF1 comparisons are nonregressing.

This is a partial resolution of the requested short-possession task. Main short matches improve3/6→5/6. The71-frame away possession remains completely missed. The19- and33-frame home possessions now cross IoU0.5, but their predicted boundaries are not frame-perfect. The strict complete-resolution feedback loop remains red on its all6shortmatches condition; the nonregression and primary-improvement conditions pass.

| Seed42 protocol | Previous SpellF1 | Repaired SpellF1 | Previous macroF1 | Repaired macroF1 |
|---|---:|---:|---:|---:|
| Buffered minutes | .777143 | .811429 | .888268 | .892006 |
| Purged spells | .731429 | .765714 | .890444 | .894328 |
| Buffered100seconds | .681081 | .691892 | .856936 | .857837 |

The rule reads the unchanged predicted sequence and aligned broadcast-ball/player observations. It only acts on8–250-frame H/A islands with opposite-team live neighbors, complete broadcast mapping/ball observation and finite nearest-player distances. At least3exclusive same-camera contacts must have owner distance<2m and opponent margin>.5m. Each island is intersected with its observed contact envelope plus13frames of padding. Each edge can move inward by at most25frames; at least8frames must remain. It never creates a possession, crosses a decode boundary or changes D/X. Eligibility always reads the unchanged reference.

The existing contact thresholds and temporal scales were reused. A separately frozen5-frame-padding comparator was also tested. Both improve the three seed42 aggregate scores, but13frames yields the higher main SpellF1 and five short matches versus four. An original ranking implementation chose5because mathematically equal minimum gains differed by1.1e-16. That result and its source are preserved. A separate qualification plan fixes only the ranking comparison to12decimal precision before main-score tie-breaking; the inference variants are unchanged. Seed43/44 confirmations follow this corrected choice without further tuning.

The19-frame home's best main IoU improves .463415→.575758; the33-frame home's improves .428571→.634615. Main IoU0.5 matched possessions increase13→15, spurious predictions fall5→3, and missed true possessions fall4→2. Main frame accuracy improves .903867→.908267. The33edited main frames all correct previously wrong predictions. The repair adds no main frame errors.

| Seed | Main SpellF1 | Purged SpellF1 | Longer SpellF1 |
|---|---:|---:|---:|
|42|.811429|.765714|.691892|
|43|.811429|.765714|.745946|
|44|.822857|.744444|.745946|

Production retains prespecified seed42. The repaired full-match CSV has199319rows and SHA256 `e2f550bec2ccaa0a381bfe1e227538f800fdf1b369005c526c70e6e68a8713a5`. It changes347reference H/A frames across16short islands. Every D/X frame remains exact. These347changes are not347verified full-match corrections; private labels remain unavailable. Full and public broadcast feature rows match exactly. Two separate-process full builds are byte-identical in CSV, events and manifest and pass the official validator. No additional TabPFN fitting or native inference is needed.

The new candidate is `data/candidate_tabpfn_broadcast_short_v1/candidate.csv`. The prior TabPFN CSV and package remain at `data/candidate_tabpfn_short_control_v1/`; the older motion candidate also remains intact. The preceding selection document is preserved as `SELECTED_MODEL_BEFORE_BROADCAST_SHORT_20261107.md`.

The remaining away miss was investigated through two additional frozen families. A train-only duration-balanced CatBoost ownership head used inverse-square-root and equal-spell training weights,166existing original/latent features and54fine observation taps. All26outer-fold fits and exact checkpoint reloads complete. Six bounded insertion/refinement variants over three protocols add18score rows, but no candidate passes the nonregression and added-short-match conjunction. A contact-triggered continuation rule tests13- and25-frame stable kinematic reacquisition. It adds false possessions and regresses all three primary scores; both are rejected. None of these rejected models or rules is included in the selected package.

All39new dependent score rows, including failed variants and the broadcast replay, are retained and scored through the official CLI. A separate audit reconstructs all39prediction outputs, reloads all26new rejected checkpoints to exact probabilities, independently implements the successful contact-envelope rule for all9confirmations and the full match, checks39fold comparisons, and validates both full exports. All33affected unit tests pass. No previous source, candidate or historical ledger bytes were overwritten;39new rows append to the ledger.

Evidence is under `data/control_short_repair_v1/`, `data/control_broadcast_short_envelope_v1/` and `data/control_short_continuation_v1/`. The final audit is `data/control_short_continuation_v1/final_audit.json`. The source contains no frame-specific or protocol-specific inference exception, no truth input and no hand-edited output labels. Diagnostic frame ranges appear only in metadata and reports.

To rebuild with the preserved parent and broadcast caches:

```sh
.venv/bin/python experiments/build_broadcast_short_candidate.py --output data/candidate_tabpfn_broadcast_short_fresh
.venv/bin/python evaluation/validate_submission.py data/candidate_tabpfn_broadcast_short_fresh/candidate.csv
```

The evidence still uses the same7500public frames and17true possessions. The fix was designed after inspecting those errors, so the gains are adaptive same-match development results, not independent evidence of private accuracy. Complete short-error resolution and maximum score remain unestablished.
