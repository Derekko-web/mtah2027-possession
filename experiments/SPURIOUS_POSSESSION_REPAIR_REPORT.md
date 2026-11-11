# Stoppage-fragment repair — 7 November 2026

The selected seed42 recipe is `tabpfn_supported_stoppage_fragments_seed42`. It removes the three spurious possessions in main out-of-fold validation, yielding **SpellF1 1.000000** at every scored IoU threshold from 0.3 to 0.7. All 17 true possessions remain matched. The same result holds for the other two main-validation seeds. This is the ceiling for that development check, not perfect frame labelling or verified private performance.

## Scores against the immediately preceding recipe

| Seed | Protocol | Previous SpellF1 | New SpellF1 | Extras at IoU 0.5, before → after |
| --- | --- | ---: | ---: | --- |
| 42 | Main, buffered minutes | 0.9189189189 | **1.0000000000** | 3 → 0 |
| 42 | Purged spells | 0.8756756757 | **0.8756756757** | 3 → 3 |
| 42 | Longer, buffered 100 seconds | 0.8000000000 | **0.8210526316** | 6 → 5 |
| 43 | Main | 0.9189189189 | **1.0000000000** | 3 → 0 |
| 43 | Purged | 0.8756756757 | **0.8756756757** | 3 → 3 |
| 43 | Longer | 0.8512820513 | **0.8736842105** | 5 → 4 |
| 44 | Main | 0.9189189189 | **1.0000000000** | 3 → 0 |
| 44 | Purged | 0.8421052632 | **0.8648648649** | 5 → 4 |
| 44 | Longer | 0.8512820513 | **0.8736842105** | 5 → 4 |

Main seed42 macroF1 improves **0.901583 → 0.932449**, and accuracy **0.919600 → 0.942533**. All nine primary, macroF1, accuracy, mean best overlap and five per-IoU F1 comparisons are nondecreasing. Extras never increase at any scored threshold, and every previous true-possession match remains. Each of the 153 individual true-possession overlaps is nondecreasing. All 39 clipped-fold primary/macro checks pass. The 33-/87-frame home overlaps remain unchanged at 0.744186/0.711340 in the main check.

Other holdouts still contain extras, as the table shows. The rules decline changes without enough evidence.

## Diagnosis and correction

The three main seed42 extras are home predictions with no overlap with a true home possession:

| Original prediction, end exclusive | Frames | Ground truth within prediction | Correction |
| --- | ---: | --- | --- |
| 12839–12878 | 39 | 12 away, 27 dead | Continue away for 14 frames, then extend the adjacent stoppage over 25 frames |
| 16207–16239 | 32 | All dead | Extend adjacent stoppage |
| 18052–18155 | 103 | All dead | Extend adjacent stoppage |

The first error is an ownership flip while neither camera sees the ball. The other two have no usable broadcast observations, but a visible tactical ball with mostly low displacement and weak dead-ball model support. Treating every short run as false would erase real possessions; absence of broadcast observations alone is insufficient.

The fixed rule considers only original H/A runs of 1–125 frames adjacent to an existing D run, with no usable broadcast ball observations. It has two branches:

1. A wholly unobserved flip between the opposing owner and D continues that owner's possession. The D-facing edge uses the union of a 25-frame allowance from the existing dead boundary and a 25-frame allowance from the actual visibility gap. Visibility lookup stays within 25 frames of the original interval and within the same decode block. The proposal is refused unless at least eight frames of the unseen interval remain as inferred live continuation.
2. Otherwise, the whole fragment may become D only if tactical ball visibility covers at least half of it, movement above the existing 4.5m displacement threshold covers at most half, and at least one saved weak dead-ball head has mean probability above 0.5.

All decisions read the unchanged original labels and observations. The rule applies symmetrically to both teams and at stoppages or restarts. Inference receives no ground truth, seed, protocol or target frame exception. Existing D labels are retained, X is exact, and no extra false-D frames are introduced in any of the nine labelled comparisons. Unlike the preceding boundary-only work, removing false live fragments necessarily changes run topology and may add D.

Implementation is in `spurious_stoppage_fragments.py`, `stoppage_gap_closure.py`, `observed_gap_stoppage.py`, `dual_anchor_stoppage.py` and `supported_gap_stoppage.py`. The final entry point is `supported_gap_stoppage.refine`. There are no new model fits or TabPFN queries; saved weak-D probabilities are reused.

## Preserved failures and adaptation

Five separate frozen rounds retain all trials:

- `control_spurious_possessions_v1`: nine scores. Whole-run invisible continuation removed an extra but extended the neighboring away possession into the stoppage, harming overlap and some fold macro scores. The combined method reached 1.0 main SpellF1 but failed the secondary gates.
- `control_spurious_gap_closure_v1`: six scores. A fixed 13-/25-frame extension from the predicted dead edge retained a longer-holdout overlap regression.
- `control_observed_gap_stoppage_v1`: six scores. Anchoring to the visibility gap instead fixed the longer case, but the 13-frame allowance added false D, and the 25-frame allowance reduced purged overlap.
- `control_dual_anchor_stoppage_v1`: nine scores. The union passed seed42, then a seed43 purged overlap regression rejected it.
- `control_supported_gap_stoppage_v1`: nine scores. The existing eight-frame minimum support criterion makes the final method refuse those nearly all-dead unseen closures. All nine comparisons and 39 folds pass without weakening the original score gates.

These are 39 dependent development score rows: 30 rejected and nine final. Seeds43/44 informed the last refusal, so they are paired development checks, not fresh independent confirmations of the final rule. The previous selected recipe was protected throughout research; promotion preserves its CSV, pointer and selection document, and all preceding ledger bytes. The initial red check with an exact-D policy is retained; the revised monotonic-D/no-added-false-D policy was frozen before the first candidate scores, after diagnosis showed the extras overlap actual stoppages.

## Verification and artifact

All **20 focused tests** pass, including **900 randomized comparisons** against a separately implemented inference function. Cases cover all nine retained variants, both teams, complete/missing/partial camera evidence, movement and likelihood ties, boundary direction, minimum support and multiple decode blocks.

The independent implementation reconstructs all **39 OOF trial files** and the entire full-match output. The official scorer agrees on all 39 OOF results and both preceding/new full files, for **41 official score checks**. All 39 selected fold comparisons pass, and both separate full builds have identical CSV, event, comparison and manifest bytes. Both CSVs pass the official submission validator.

The selected CSV has 199319 rows:

`data/candidate_tabpfn_stoppage_fragments_v1/candidate.csv`

SHA256: `4b1dc28231b38019644e4e943db39e5142580ba4695e7a3434f93a3e91abfeb4`.

The full file changes **327 frames across six fragments**, all outside the labelled public window. Public full-fit predictions are byte-for-byte unchanged, with SpellF1/macroF1/accuracy remaining **0.912500/0.991058/0.991200**. The full-fit model already handles the three targeted development fragments differently, so the OOF repairs do not imply those same changes in production. The six unlabelled suppressions are not verified corrections; private score effects remain unknown.

Evidence is under `data/control_supported_gap_stoppage_v1/`: frozen plans and source copies, `qualification_result.json`, `final_audit.json`, `boundary_diagnostics.csv`, `main_spurious_diagnostics.json`, `fold_scores.csv`, `affected_tests.log`, `official_scoring/` and both full exports. All 39 score rows append to `EXPERIMENT_LEDGER.csv`, preserving the preceding bytes. The previous selection document is saved as `SELECTED_MODEL_BEFORE_SPURIOUS_REPAIR_20261107.md`.

The same 7500 public frames and 17 true possessions have extensively informed development. SpellF1 1.0 on this main check establishes measured development success; it does not establish generalization, exact boundaries, the private maximum or full-match error removal.

Reproduce into a fresh directory:

```sh
.venv/bin/python experiments/build_stoppage_fragment_candidate.py --output /tmp/stoppage-fragment-replay
```

Rerun the acceptance check:

```sh
.venv/bin/python experiments/reproduce_spurious_possessions.py --candidate data/control_supported_gap_stoppage_v1/candidates/supported_gap25
```
