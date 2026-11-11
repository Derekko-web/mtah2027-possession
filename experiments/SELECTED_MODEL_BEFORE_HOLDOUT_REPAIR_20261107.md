# Selected model: TabPFN with supported stoppage-fragment repair

The selected recipe is `tabpfn_supported_stoppage_fragments_seed42`. Its 199319-row CSV is `data/candidate_tabpfn_stoppage_fragments_v1/candidate.csv`, SHA256 `4b1dc28231b38019644e4e943db39e5142580ba4695e7a3434f93a3e91abfeb4`.

Seed42 out-of-fold SpellF1 main/purged/longer is **1.000000/0.875676/0.821053**, compared with **0.918919/0.875676/0.800000**. All three main-validation seeds now match all 17 true possessions with zero extras at every scored IoU from 0.3 to 0.7. Main seed42 macroF1 improves 0.901583 to 0.932449 and accuracy 0.919600 to 0.942533. Other holdouts retain extras; the main ceiling is not perfect frame accuracy or verified private performance.

All nine primary/macroF1/accuracy/mean-overlap/five-IoU F1 comparisons are nonregressing. Every prior true match remains, all 153 individual true overlaps are nondecreasing, and all 39 fold primary/macro checks pass. Recent 33-/87-frame home overlaps are preserved. Existing D is retained, X is exact, and no false-D frames are added in the labelled comparisons.

The rule removes short unsupported owner flips next to existing stoppages. Entirely unseen flips use bounded visibility/prediction anchors and require eight frames of inferred live continuation; observed fragments require weak dead-head support and mostly stationary ball evidence. Both teams use the same rules; inference has no truth, protocol or target-frame input. It changes topology and may add D, as required to remove false live fragments. No new model fits or TabPFN queries are needed.

Twenty focused tests pass, including 900 randomized independent comparisons across every retained variant. Independent reconstruction verifies all 39 OOF trial outputs and the complete full CSV; 41 official score checks agree. Two identical full exports pass validation. The full file suppresses six unlabelled fragments, changing 327 frames. Public full-fit predictions and SpellF1/macroF1/accuracy remain exactly unchanged at 0.912500/0.991058/0.991200. Those 327 private changes are unverified.

The final guard followed a seed43 overlap failure, so every seed is adaptive development evidence. The 7500 public frames and 17 possessions have extensively informed the work; private effects and generalization remain unknown.

See [SPURIOUS_POSSESSION_REPAIR_REPORT.md](SPURIOUS_POSSESSION_REPAIR_REPORT.md) for methods, retained failures, evidence and reproduction. The preceding selection document is preserved exactly in [SELECTED_MODEL_BEFORE_SPURIOUS_REPAIR_20261107.md](SELECTED_MODEL_BEFORE_SPURIOUS_REPAIR_20261107.md). Prior candidates remain unchanged. All 39 dependent score rows append with preceding ledger bytes preserved.
