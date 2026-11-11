# Current research checkpoint — 7 November 2026

The original higher-score research objective has been achieved relative to its historical .741176/.777143 development benchmarks. The current selected recipe is `tabpfn_physical_continuity_repair_seed42`. All nine retained seed/protocol development comparisons have SpellF1 1.0. A strict improvement over the current 1.0 is impossible under the unchanged official metric.

The continuation resumed from an obsolete GP checkpoint. Current selection and later reports were inspected before launching any model; the already completed GP experiment was not repeated. The initial continuation update about running GP is superseded by this verified state.

## Verified current output

- Pointer: `experiments/SELECTED_MODEL.json`.
- Candidate: `data/candidate_tabpfn_holdout_repair_v1/candidate.csv`.
- SHA256: `400c86edb5ba25bf0f9c40b29a7c8d1d305b0a4d4b7fd3ebc78a0fb04efe415b`.
- Exactly 199319 ordered unique frames, 0 through 199318, with valid H/A/D/X labels.
- All 36 manifest source, input and evidence hashes verified against current bytes, with no mismatches.
- Qualification records cover seeds 42, 43 and 44 and each of `buffered_minutes`, `purged_spells` and `buffered_100_seconds`. Every pair scores 1.0 and passes its retained nonregression checks.
- A separate read-only agent audit independently verified the same candidate hash, both identical full exports, all 36 bindings and all nine ceiling results. At all five official IoUs, each retained result has 17 true possessions matched and no extras or misses.

The retained final audit documents 14 passing focused tests, 30 independent OOF reconstructions, 32 official scoring checks and two passing full-export validators. Those earlier tests, reconstructions and model predictions were not rerun in this continuation. See `HOLDOUT_EXTRAS_REPAIR_REPORT.md`, the candidate's `qualification_result.json`, `final_audit.json` and `manifest.json`.

## Interpretation and next evaluation

The official `evaluation/possession_metrics.py:517` computes precision and recall from nonnegative one-to-one match counts, then their harmonic mean at IoUs .3, .4, .5, .6 and .7. Each F1 and their mean is bounded above by 1. The current development result attains that ceiling; this does not imply exact boundaries or error-free frame labels.

The selected full-fit CSV's separate public SpellF1 remains **0.9125**. It must not be described as the OOF 1.0 result. All nine OOF comparisons reuse the same public 7500 frames and 17 possessions after extensive adaptive development. They are not nine independent matches. The selected file's 832 changed unlabelled frames and private score remain unverified.

Primary-source evaluation research is recorded in `SATURATED_VALIDATION_RESEARCH_20261107.md`. Its actionable conclusion is to keep this selected package fixed and assess the complete pipeline on untouched, match-disjoint labelled data before claiming generalization. New seeds or repartitioning these already examined frames cannot make them untouched again. New test labels, a new match score and an independently established improvement do not exist in this continuation.

No model fits, TabPFN queries, inference-rule changes, candidate replacements, ledger changes, submissions, uploads, commits or external messages were made during this continuation. All prior failures, frozen sources and candidates remain preserved. This checkpoint and the new research note are separate from the frozen experiment evidence.
