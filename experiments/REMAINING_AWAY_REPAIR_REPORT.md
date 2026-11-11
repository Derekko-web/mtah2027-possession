# Remaining away-possession recovery — 7 November 2026

The71-frame away possession is now matched at IoU0.5 in **all nine held-out seed/protocol comparisons**, with best IoU **0.689189**. Seed42 main SpellF1 improves **0.811429→0.864865**, purged **0.765714→0.821622**, and longer-block **0.691892→0.748718**. Every aggregate SpellF1, macroF1, accuracy and each of the five per-IoU F1 scores is nondecreasing. Every previously IoU0.5-matched true possession is retained. No SpellF1 or macroF1 regression occurs in39clipped-fold comparisons.

The full-match CSV is **byte-identical to its predecessor**, SHA256 `e2f550bec2ccaa0a381bfe1e227538f800fdf1b369005c526c70e6e68a8713a5`. The full-fit parent already labels70of71diagnostic frames as away after training on all public labels. This fix addresses its held-out failure. The new stage fires in the held-out reconstructions but nowhere in the full-fit output, so no improvement to private submission scores is claimed. Identical full-match predictions necessarily produce identical private scores.

| Seed42 validation | Before SpellF1 | After SpellF1 | Before macroF1 | After macroF1 |
|---|---:|---:|---:|---:|
| Buffered minutes | .811429 | .864865 | .892006 | .897427 |
| Purged spells | .765714 | .821622 | .894328 | .899797 |
| Buffered100seconds | .691892 | .748718 | .857837 | .863238 |

| Seed | Main SpellF1 | Purged SpellF1 | Longer SpellF1 |
|---|---:|---:|---:|
|42|.864865|.821622|.748718|
|43|.864865|.821622|.800000|
|44|.875676|.800000|.800000|

Production retains prespecified seed42. Main short-possession IoU0.5matches improve5/6→6/6. All17true main-validation possessions now have IoU0.5matches, while three spurious predicted possessions remain. The new mean SpellF1 is not a claim of perfect possession segmentation.

The recovered prediction is frames11573–11626inclusive against the supplied true interval11553–11623. It correctly identifies51of71true away frames and extends three frames into the next home possession: intersection51, union74, IoU51/74. The first20true away frames remain labelled home, so the correction is a recovered spell with approximate boundaries, not a frame-exact repair. It changes54held-out labels:51previous errors become correct and3previously correct home frames become wrong, for48net additional correct frames. This is the same observed correction across all9paired checks, not nine independent recovered possessions.

The cause of the inactive broadcast branch was missing strict alignment. The original strict mapping has0/71source correspondences in the target interval. A previously frozen, label-free sub-grid mapping from Round25covers all71. Round25was rejected as an action-clip opportunity because it added only50safe action queries; it nevertheless added454point mappings in the public window. Its unused point geometry supplies three exclusive broadcast away contacts near the tactical feed's six contacts. No alignment threshold was relaxed, new source correspondence hand-picked, or output interval hard-coded.

The added inference stage runs uniformly over decode blocks and both teams. It can insert control only inside a homogeneous reference H/A spell longer than250frames, where the old broadcast mapping is absent throughout the proposed episode and the refined broadcast observations are complete. Initiation requires at least3exclusive tactical contacts with fast-head majority>0.5 and a reference-owner motion kernel<0.5, plus at least3exclusive broadcast contacts within13frames of onset. Existing distance<2m and opposing-distance margin>0.5m gates are reused. Control persists until13consecutive reference-owner frames have motion kernel>=0.5 and fast probability>=0.65, with3broadcast contacts confirming release. The episode must last8–125frames. There is no timeout guess; absence of confirmed release rejects the proposal. D/X and existing short reference spells are protected. All gates read unchanged inputs.

A separately frozen25-frame-release comparator also passes the seed42 floors but has lower primary gains and target IoU0.5930. The13-frame finalist was selected before seeds43/44, without subsequent retuning. Its new-camera gate excludes the tactical-only false contacts that harmed the prior continuation experiment. No frame number, possession truth, protocol name or model seed is an inference feature.

Two actual source-prefix feature builds reproduce all68public columns exactly. Full production reuses the unchanged original sub-grid algorithm over the entire match, retaining the original half/reserved/held-window guards. It executes4126global coarse queries over7971grid positions in approximately285seconds, adds2255full-match point mappings, and preserves every original valid mapping field. Public mapping fields and features match the validated prefixes exactly. This is a full-match observational rebuild, not a public-window patch.

An initial full-export attempt stopped because the old full motion cache contains broadcast-motion fields but no tactical `tc_kernel` columns. Both attempts stopped before creating output CSVs. Their source and failure record remain in `data/control_remaining_away_v1/full_motion_repair/`. The repair rebuilds full tactical kernels from the verified original tracking with unchanged `native_pair_features`, in4096-query chunks. Its public values match the historical tactical kernels exactly. The control rule, probabilities, allOOFpredictions and scoring/independent-replay function ASTs remain unchanged. The original finalization plan and a separatev2full-only schema-repair plan are preserved.

Validation includes12independent OOF reconstructions,12candidate plus9baseline official CLI scores,9complete target/current-score conjunctions,39fold comparisons, and an independent full-label reconstruction using a separately written convolution-based release detector. Both full exports match in CSV, events and manifest and pass the official validator. All27affected tests pass, including200time-shift/jitter/missingness differential cases. No new model fits or TabPFN queries occur. All prior candidates and historical ledger bytes remain unchanged;12dependent new evaluation rows append.

The selected recipe is `tabpfn_new_camera_control_seed42`, packaged at `data/candidate_tabpfn_new_camera_v1/`. Its CSV matches the prior package exactly; the method and validation evidence differ. The preceding selection document is preserved as `SELECTED_MODEL_BEFORE_NEW_CAMERA_20261107.md`. The former broadcast-short, TabPFN-short-control and motion packages remain available.

Evidence is under `data/control_remaining_away_v1/`, particularly `qualification_result.json`, `oof_audit.json`, `final_audit.json`, `diagnostics.csv`, `fold_scores.csv`, `full_inputs/manifest.json`, `full_control_kernel/manifest.json`, and the two preserved finalization plans. Rebuild the candidate from the preserved parent, full source features and tactical kernels with:

```sh
.venv/bin/python experiments/build_new_camera_candidate.py --output data/candidate_tabpfn_new_camera_fresh
.venv/bin/python evaluation/validate_submission.py data/candidate_tabpfn_new_camera_fresh/candidate.csv
```

![Recovered away-possession timeline](/Users/derekko/Projects/more-than-a-hack-2027/data/control_remaining_away_v1/remaining_away_recovery.png)

These results reuse7500public labels and17true possessions from the same match. The rule was designed after inspecting this error; seed/protocol replay establishes robustness to those fitted contexts and reproducibility, not independent private generalization or an unbiased estimate of the gain. Frame-exact recovery and maximum score remain unestablished.
