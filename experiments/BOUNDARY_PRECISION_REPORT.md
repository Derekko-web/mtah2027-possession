# Guarded possession-boundary precision — 7 November 2026

The selected method now refines contact-based boundaries only when the saved ownership posterior corroborates the replacement team, and retrospectively localizes proven two-camera recovery onsets from bilateral approach observations. Seed42 held-out SpellF1 improves **0.864865→0.897297** on the main setup, **0.821622→0.854054** on purged spells, and **0.748718→0.789744** on100-second blocks.

All nine seed/protocol comparisons improve primary SpellF1, macroF1, accuracy and mean best possession IoU. Each of the five per-IoU F1 scores is nondecreasing. Every previously matched true possession at every one of those thresholds is retained. All39clipped-fold primary/macro comparisons are nonregressing. Unlike the preceding recovery-only stage, the full-match CSV changes: **49frames across eight boundary refinements**, with every D/X label and the run-label sequence preserved. Private full-match effects remain unknown.

| Seed42 validation | Previous SpellF1 | Refined SpellF1 | Previous macroF1 | Refined macroF1 |
|---|---:|---:|---:|---:|
| Buffered minutes | .864865 | .897297 | .897427 | .900024 |
| Purged spells | .821622 | .854054 | .899797 | .902419 |
| Buffered100seconds | .748718 | .789744 | .863238 | .865826 |

| Seed | Main SpellF1 | Purged SpellF1 | Longer SpellF1 |
|---|---:|---:|---:|
|42|.897297|.854054|.789744|
|43|.897297|.854054|.830769|
|44|.908108|.831579|.830769|

Production retains prespecified seed42. Main short-possession matches remain6/6. All17main true possessions now match at IoU0.6;15match at0.7, versus13before. Three spurious predicted possessions remain. The33-frame and87-frame home possessions are still below IoU0.7; not every boundary is exact.

Four main-validation spells have better overlap:

| True possession | Previous IoU | Refined IoU | Timing change |
|---|---:|---:|---|
|202-frame home|.860360|.920792|Ending20frames late becomes5frames early|
|71-frame away|.689189|.898734|Beginning20frames late becomes5frames early|
|19-frame home|.575758|.760000|Ending14frames late becomes6frames late|
|401-frame away|.942643|.962594|Beginning14frames late becomes6frames late|

The away interval now covers all71true away frames, with5early and3late home frames included. This is closer timing, not a frame-exact annotation. The19-frame home's predicted duration contracts33→25frames. Main mean best IoU rises .819672→.847564; mean absolute GT start/end error falls19.06→17.71frames, approximately .762→.708seconds. The33main label changes correct28previously incorrect frames and damage5previously correct frames, for23net additional correct frames. These are the same repeatedly observed public examples across protocols and seeds, not independent gains.

The initial frozen screen considered seven uniform variants: three arrival/release envelope paddings using existing5/13-frame scales,13/25-frame approach lookbacks, and two combinations. All topology and D/X decisions were protected. A25-frame approach lookback passed, but larger envelope gains hid a purged regression. The unguarded combined method raised main SpellF1 to .897297 while reducing purged macroF1 and losing previous stricter-IoU matches. It was rejected under the original floors. No gate was relaxed.

The diagnostic separates the failed cut from useful release evidence. The unsafe purged cut proposed replacing25away frames while the saved TabPFN probability for the new home label averaged only .288. Useful19-frame-home release cuts had replacement-away probabilities .545 main, .583 purged and .512 longer. A separately frozen posterior guard therefore accepts a contact-boundary move only when the original conditional ownership head assigns the replacement team mean probability>0.5 over the changed frames. The threshold is the binary majority decision, with ties preserving the reference. No posterior is refitted or calibrated.

The successful rule uses complete refined broadcast observations, same-camera owner distance<2m and opposing-distance margin>0.5m, at least3contacts, leading padding13frames and trailing padding5frames. Existing8–250-frame live islands with opposite live neighbors can contract by at most25frames per edge. The ownership guard rejects proposed cuts unsupported by the independently fitted ownership path.

The onset component only operates on accepted recovery metadata produced by the preceding two-camera stage. Within the preceding25frames, it requires3consecutive bilateral observations with owner distance<3m, opposing margin>0.5m and fast-head majority>0.5. Original broadcast mapping must be absent, refined observations complete, and at least8frames of the preceding opponent run must remain. It places onset at the first qualifying approach, rather than waiting for touch confirmation. Neither component reads truth, absolute-frame exceptions, protocol names or model seeds. Both proposals read the unchanged reference; their union is refused for a decode block if it changes run-label order or violates minimum run lengths.

The guarded release alone and the guarded release plus approach were frozen before seed42scoring. The combined finalist passed all three seed42setups and was selected before43/44confirmation. All nine checks then passed unchanged. These add6screen and6confirmation rows after the initial21screen rows:33dependent comparisons in total. Every failure remains recorded.

The full export reuses the unchanged fitted models, ownership probabilities, refined full-camera observations and prior recipe. No model fitting, new TabPFN queries, media extraction or alignment rebuilding is needed. Both raw distance and all68broadcast public feature columns agree exactly with the validated prefixes. Full ownership/fast probability hashes match the original TabPFN package. Two separate-process builds agree in CSV, events and manifest. An independently written reconstruction agrees with all33OOFoutputs and all199319full labels. All33official CLI scores match the shared metrics; both full CSVs pass the official format validator.

The selected candidate is `data/candidate_tabpfn_boundary_precision_v1/candidate.csv`, SHA256 `8eeb131d343fd34fea9ad77874960ef8047bccdd407a354f59a256f5de0260a0`. All49full changes are outside the labelled public window, so they are not49verified corrections. The parent CSV, SHA256 `e2f550bec2ccaa0a381bfe1e227538f800fdf1b369005c526c70e6e68a8713a5`, and all earlier candidates remain intact. The preceding selection document is preserved as `SELECTED_MODEL_BEFORE_BOUNDARY_PRECISION_20261107.md`.

All40affected unit tests pass, including900new shifted-contact/approach/posterior differential cases. Evidence lives under `data/control_boundary_precision_v1/` and `data/control_guarded_boundaries_v1/`, including the two frozen plans, all33prediction/metric/event sets, official scorer reports,39fold rows, full boundary diagnostics, two full exports and `final_audit.json`. Historical ledger bytes are preserved and33new rows append. Rebuild with the existing parent and verified caches:

```sh
.venv/bin/python experiments/build_guarded_boundary_candidate.py --output data/candidate_tabpfn_boundary_precision_fresh
.venv/bin/python evaluation/validate_submission.py data/candidate_tabpfn_boundary_precision_fresh/candidate.csv
```

Only7500same-match labels and17true possessions underlie these extensively adaptive checks. The posterior guard was designed after seeing the first screen's failures. Paired seeds and replay establish context robustness and reproducibility, not independent private generalization or an unbiased gain estimate. Private score changes and globally optimal boundaries remain unverified.
