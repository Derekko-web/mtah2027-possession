# Selected model: TabPFN with source-aware possession arrivals

The selected recipe is `tabpfn_source_aware_arrivals_seed42`. Its 199319-row CSV is `data/candidate_tabpfn_home_arrivals_v1/candidate.csv`, SHA256 `7c979478291b96b884095cb7a2ad37a877d0790b7855965c48937db5d46c4876`.

Seed42 held-out SpellF1 improves **0.897297/0.854054/0.789744 → 0.918919/0.875676/0.800000** for main/purged/longer validation. All nine seed/protocol primary, macroF1, accuracy, mean best overlap and five per-IoU F1 comparisons are nonregressing. Every prior true-possession match remains at each threshold, and all 39 clipped-fold primary/macro checks pass.

The 33-frame and 87-frame main home overlaps improve **0.634615 → 0.744186** and **0.690000 → 0.711340**. All 17 main true possessions now match at every scored IoU from 0.3 to 0.7; three spurious predictions remain. The rules combine complete-source contact disagreement with posterior refinement restricted to short leading broadcast gaps supported by later owner contacts. Both teams use the same rules, with no truth, protocol or target frame input. D/X and run topology are preserved. No new model fitting or TabPFN queries are needed.

All 26 affected tests pass, including 800 new and 900 retained randomized comparisons. An independent implementation matches all 30 trial OOF outputs and the full candidate. Official scoring verifies all 30 OOF results; both identical full exports pass submission validation. The full file changes 44 frames across five events: eight public and 36 unlabelled. Its labelled full-fit public SpellF1 improves 0.9000 to 0.9125; this is distinct from held-out performance.

The public labels have extensively informed development. Boundaries remain approximate, with weak unchanged target overlaps in some holdouts. Private full-match effects remain unknown; the 36 unlabelled changes are unverified.

See [HOME_BOUNDARY_REFINEMENT_REPORT.md](HOME_BOUNDARY_REFINEMENT_REPORT.md) for evidence, limitations and reproduction. The exact preceding selection document is preserved in [SELECTED_MODEL_BEFORE_HOME_ARRIVALS_20261107.md](SELECTED_MODEL_BEFORE_HOME_ARRIVALS_20261107.md). Every previous candidate remains unchanged. Thirty dependent development score rows append with preceding ledger bytes preserved.
