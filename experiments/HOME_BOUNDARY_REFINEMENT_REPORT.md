# Home possession arrival refinement — 7 November 2026

The selected seed42 recipe is `tabpfn_source_aware_arrivals_seed42`. Both requested home intervals exceed IoU 0.7 in the main held-out check, and all nine seed/protocol comparisons preserve the preceding primary and secondary score floors. The previous candidate, selection document and all failed trials remain preserved.

## Measured changes

These are out-of-fold (OOF) scores, compared with the immediately preceding `tabpfn_guarded_boundaries_seed42` recipe:

| Seed | Protocol | Previous SpellF1 | New SpellF1 |
| --- | --- | ---: | ---: |
| 42 | Main, buffered minutes | 0.8972972973 | **0.9189189189** |
| 42 | Purged spells | 0.8540540541 | **0.8756756757** |
| 42 | Longer, buffered 100 seconds | 0.7897435897 | **0.8000000000** |
| 43 | Main | 0.8972972973 | **0.9189189189** |
| 43 | Purged | 0.8540540541 | **0.8756756757** |
| 43 | Longer | 0.8307692308 | **0.8512820513** |
| 44 | Main | 0.9081081081 | **0.9189189189** |
| 44 | Purged | 0.8315789474 | **0.8421052632** |
| 44 | Longer | 0.8307692308 | **0.8512820513** |

| Main seed42 interval | Previous IoU | New IoU | Arrival error before → after | Release error |
| --- | ---: | ---: | --- | --- |
| 33-frame home, 15766–15798 inclusive | 0.634615 | **0.744186** | 9 frames early → 1 frame late | Unchanged, 10 frames late |
| 87-frame home, 17293–17379 inclusive | 0.690000 | **0.711340** | 13 frames early → 10 frames early | Unchanged, 18 frames early |

The main check now matches all 17 true possessions at every scored IoU threshold from 0.3 through 0.7. Three spurious predicted possessions remain. Its mean best overlap increases from 0.847564 to 0.857983. These boundaries are tighter, not exact.

## Evidence and implementation

The 33-frame interval has complete broadcast observations. Broadcast contact evidence favors home, while tactical contact evidence favors away. The previous posterior guard incorrectly vetoes a useful arrival trim because the saved tactical-conditioned owner probabilities also favor home before the actual arrival. For a complete interval with this source disagreement, the new branch uses the first exclusive broadcast contact with a five-frame allowance. It requires at least three broadcast contacts for the current owner and three opposing tactical contacts, with each source's majority pointing in its respective direction.

The 87-frame interval starts with a short gap in the broadcast observations. A contact-only onset would arrive too late. For an incompletely observed interval, a second branch minimizes saved conditional-owner log loss over at most 13 leading frames, subject to the existing mean replacement probability exceeding 0.5. It additionally requires the original onset to be missing, the first observed broadcast frame within 13 frames, and at least three observed contacts for the current owner later in that original interval. Tail-only gaps, long initial gaps and intervals without observed owner control cannot trigger this branch.

Both branches apply uniformly to either team, only to original live runs of 8–250 frames preceded by the opposing live label. They preserve D/X and the sequence of run labels. Inference takes no ground truth, seed, protocol, target frame number or interval-specific exception. There are no new model fits or TabPFN queries. Implementation: `source_aware_arrivals.py` and `leading_gap_arrivals.py`.

## Rejected work and selection

`data/control_home_boundaries_v1/` preserves all 15 initial score rows. The broad incomplete-source rule also moved arrivals with only later source gaps, damaging a following away possession and some target overlaps. It failed the gates and was not promoted.

The separately frozen leading-gap experiment preserves nine seed42 rows. Its owner-only comparator cannot tighten both main targets. Its 25-frame comparator harms the 87-frame overlap on purged/longer checks and fails the unchanged gates. `leading_camera13` was the sole eligible method and was selected before running seed43 and seed44, which contribute six confirmation rows. All 30 dependent development score rows, including rejected variants, append to the ledger. They are not 30 independent statistical trials. The production seed remains the previously specified seed42.

## Verification

- All nine SpellF1, macroF1, accuracy, mean best IoU and five per-IoU F1 comparisons are nondecreasing. Every previously matched true possession remains matched at each threshold. Both requested interval IoUs are nondecreasing in every pair.
- All 39 clipped-fold SpellF1 and macroF1 comparisons are nondecreasing. D/X masks and run topology remain exact.
- An independent implementation reconstructs every one of the 30 trial OOF files and the complete final CSV. All 30 OOF files agree with the official scorer at full precision.
- All 26 affected tests pass. These include 800 new randomized comparisons covering both teams, complete/conflicting sources, initial/tail/long/missing gaps, multiple blocks, float32 and exact likelihood ties, plus the retained 900-case boundary comparison.
- Two separate full builds produce identical CSV, event and manifest bytes. Both pass the official submission validator.
- The exported file's labelled public SpellF1 improves **0.900000 → 0.912500**, macroF1 **0.990378 → 0.991058**, and accuracy **0.990400 → 0.991200**. All five per-IoU scores are nondecreasing. These full-fit public scores reuse training labels and are distinct from OOF validation scores.

Evidence resides in `data/control_leading_gap_boundaries_v1/`: `qualification_result.json`, `final_audit.json`, `fold_scores.csv`, `boundary_diagnostics.csv`, `additional_tests.log`, `official_scoring/`, `full_public_check.json`, and the two full exports. The original red acceptance result remains in `data/control_home_boundaries_v1/baseline_red.json`.

## Selected artifact and limits

Selected CSV: `data/candidate_tabpfn_home_arrivals_v1/candidate.csv`, 199319 rows, SHA256 `7c979478291b96b884095cb7a2ad37a877d0790b7855965c48937db5d46c4876`.

The full file changes 44 frames across five arrival events relative to the preserved `8eeb131d…` parent: eight labelled public frames and 36 unlabelled frames. It uses three camera-disagreement and two leading-gap events. These are not 44 independently verified corrections. The full fit already handles the 87-frame interval differently from the OOF predictions and does not need the same local movement. Private score effects remain unknown.

Some validation boundaries remain poor: seed42's longer holdout leaves the 33-frame overlap at 0.211538; seed44's purged holdout leaves the 87-frame overlap at 0.448276. The current rules decline unsupported changes. The 7500 public frames and 17 possessions have extensively informed development; reused seed confirmations do not establish performance on a new match or private labels.

`SELECTED_MODEL_BEFORE_HOME_ARRIVALS_20261107.md` preserves the exact preceding selection document. Prior candidate packages remain unchanged. Promotion records preserve the preceding ledger bytes and selection pointer.

To reproduce a full export into a fresh directory:

```sh
.venv/bin/python experiments/build_home_boundary_candidate.py --output /tmp/home-arrival-replay
```

To rerun the paired acceptance check:

```sh
.venv/bin/python experiments/reproduce_home_boundary_precision.py --candidate data/control_leading_gap_boundaries_v1/candidates/leading_camera13
```
