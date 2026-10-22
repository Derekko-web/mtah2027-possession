# Weak-supervised stoppage recovery — 21 October 2026

The new candidate substantially improves recognition of visible-ball stoppages
without increasing false dead-ball predictions in any of nine paired development
checks. It retains anchored ownership and adds two small D heads. The gain is in
stoppage recall and MacroF1; buffered and purged spellF1 stay unchanged. The
longer-block primary score improves modestly.

These are adaptive same-fixture results on the same repeatedly inspected public
labels. They do not establish independent or private full-match accuracy.
No private/corrected labels, new annotations or competition media were accessed.
All processing stays local. Nothing was committed, submitted or uploaded.

## Measured gains and their limits

Official scorer, seed 42, compared with the protected anchored default:

| Protocol | spellF1 before → after | MacroF1 before → after | D recall before → after | False D before → after |
|---|---:|---:|---:|---:|
| Buffered minutes | 0.729412 → 0.729412 | 0.844580 → **0.867912** | 66.71% → **74.81%** | 92 → 92 |
| Whole intersecting runs excluded | 0.662857 → 0.662857 | 0.848625 → **0.867623** | 71.61% → **78.49%** | 152 → 152 |
| Buffered 100-second blocks | 0.644444 → **0.659459** | 0.801468 → **0.842747** | 52.14% → **64.94%** | 16 → 16 |

All three seeds 42/43/44 show the same primary scores above. Across all nine
pairs, D recall improves 6.88–12.80 percentage points and MacroF1 improves
0.0190–0.0413. False-D counts remain exactly unchanged in each pair.
Mostly missed stoppages (less than 50% D recall) fall from 1/0/1 to 0/0/0 by
protocol, for every seed. No previously mostly recovered short possession is
lost and short-live recall stays unchanged. Every prediction on a true public
live frame is exactly unchanged, in all nine comparisons; added corrections are
on stopped-play frames.

The final visible-ball stoppage was entirely missed in the longer-block protocol;
its D recall is now 51.93% for seed 42. In buffered minutes the same interval
improves 46.13%→64.64%, and the other visible-ball stoppage improves 52.05%→62.23%.
These remain incomplete recoveries; crossing 50% is a diagnostic convention,
not accurate recovery of every stopped frame or precise boundaries.

There are no clipped-block primary regressions in any of 39 block/seed pairs.
The longer-block final block improves 0.581818→0.633333; the other seed 42 blocks
are unchanged. Longer-block predicted possession count rises 19→20, and the
primary score improves despite that increase. Full fragment counts are unchanged.
All paired counts, block scores and true-run diagnostics are retained.

The 71-frame A and 33-frame H possessions remain missed in buffered validation.
The 19-frame H recovery still ends too late, and the 87-frame H recovery is
unchanged. The final live interval's false-D extension also remains. We did not
obtain a substantial additional primary-score gain for these short-control errors.

The original renewal plan required buffered spellF1 gain>=.04 and positive gains
in the other protocols. **No tested method met that criterion.** After the
stoppage screens, a separate adaptive secondary objective was written and frozen
before the three-seed composite confirmation. It requires every pair to have
unchanged-or-better spellF1, at least +.005 MacroF1 and +.05 D recall, no increase
in false D, no lost short recovery, and zero mostly missed stoppages. All nine
pairs pass that separate objective. This is not a retrospective claim of passing
the original primary-gain rule.

Marginal official 1000-draw 90% possession-bootstrap intervals still overlap.
Buffered intervals are identical 0.576471–0.870588; longer-block intervals are
0.500000–0.778333 before and 0.518919–0.789189 after. Seeds share the same few
possessions and overlapping training sets. No independent, selection-adjusted
significance or new-match generalization claim is made.

## Method

The four anchored models, probabilities and conditional ownership are retained.
Two depth 2,200-tree binary XGBoost heads learn D/live using:

- Team-invariant min/max proximity and activity, player spread and distances to
  the ball; no absolute pitch coordinates.
- Past/future 25/75/125-frame summaries and ball position IQR/visibility at
  radii 25/75/125. The second view adds automatic restart timing.
- Eligible public training targets with class-balanced weights, plus 1554 weak
  samples:1227 live and 327 dead. Weak samples receive total weight equal to half
  the public training mass, class balanced separately.

Weak D means an automatically tagged set piece with observed, spatially quiet
ball evidence. Weak live means sustained existing close-exclusive moving-ball
evidence. Ambiguous overlaps are dropped. Every 10 th qualifying frame is sampled
within the official half bounds. These are fallible algorithmic proxies, not
additional corrected or human ground truth.

The entire public label region plus 250 frames is excluded from weak fitting.
All 24 reserved annotation windows plus 250 frames are excluded too. Their metadata
is read only to enforce exclusions; no annotations or audit media are inspected.
Public validation targets stay out of all fitting, weights and preprocessing.
Weak data outside the public region are the same fixture, which limits transfer
claims even when public labels are properly held out.

At inference, each head is independently averaged over 51 frames inside the
current block/half. A high-confidence core requires D>.8 for 50 consecutive frames,
at least 50% local ball visibility and moving-contact evidence<=.2. Each core can
grow at most 125 frames either way through D>=.5 evidence until visibility or
moving-contact vetoes fail. Take the maximum of the two priors and anchored D,
retain anchored conditional ownership, and decode with the existing penalty 3.
No true interval endpoints, frame IDs or manually patched final labels are used.

`stoppage_consensus.py` is shared by confirmation and full generation. Padded
public features equal full-match features exactly. Physical fit features reach
at most 149 frames into the past (a 125-frame mean of 25-frame ball displacement)
and 125 frames into the future. Automatic restart kernels reach 250 frames either
way; phase-membership metadata can span the complete supplied tagged phase.
These label-free inputs are covered by the existing 250-frame exclusions.
Offline decoding and model-derived interval growth use future observations;
there is no streaming claim.

## Retained failures

Six prospective screen families and one final confirmation add 150 dependent
score rows, bringing the ledger 817→967. All prior 817 rows remain preserved.
These are combinations of feature sets, objectives, decoders, protocols and
seeds, not 150 independent experiments.

| Family | Rows | Finding |
|---|---:|---|
| Contact chronology, receiving-player geometry, phase semantics |39| Six new tree heads; broad replacement loses existing gains, gated proposals mostly do nothing. |
| Directed handoff-event head |15| Small endpoint/frame improvements do not survive the purged protocol; stronger timing weights harm short IoU. |
| Longer stoppage context and excluded weak supervision |27| Public-only heads add false D; weak heads provide useful stoppage cores, but broad replacement adds too much false D. |
| Bounded hysteresis |15| Improves D recall and removes mostly missed stoppages; individual views have complementary gaps. |
| Small neural joint/live classifiers |21| Train-only imputation/scaling and 150 fixed epochs; broad replacement regresses buffered primary, gated proposals do not fix the misses. Convergence-budget warnings are retained. |
| Contact release inside short predicted islands |15| End-only variants do nothing; trimming both ends harms short intervals. |
| Frozen two-view recovery confirmation |18| Nine candidate/reference pairs; all separate stoppage-recovery gates pass. |

Each plan, model/checkpoint where fitted, probabilities, OOF prediction file,
score/fold/segment table and execution log remains under ignored
`data/control_renewal_v1/`. The original source archive, anchored candidate and
confirmation tables were preserved before work. No failed method was deleted.

## Reproduction

Use the existing Python 3.12 `.venv` and pinned
`experiments/requirements-research.txt`, with original data symlinks intact.
First reproduce anchored rescue using [BOUNDARY_REFINEMENT_REPORT.md](BOUNDARY_REFINEMENT_REPORT.md)
if its candidate/prerequisite confirmation is missing. For a fresh stoppage run:

```sh
.venv/bin/python experiments/freeze_annotation_windows.py
.venv/bin/python experiments/prepare_stoppage_recovery.py
.venv/bin/python experiments/research_stoppage_context.py
.venv/bin/python experiments/confirm_stoppage_recovery.py
.venv/bin/python experiments/finalize_stoppage_recovery.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_stoppage_candidate.py
.venv/bin/python evaluation/validate_submission.py data/candidate_stoppage_consensus/candidate.csv
```

New research and full-builder outputs require fresh directories; existing
references are verified rather than overwritten. With completed confirmation,
repeat full generation using:

```sh
.venv/bin/python experiments/build_stoppage_candidate.py --output data/candidate_stoppage_consensus_rebuild
```

The other rejected screens are reproduced by `research_control_renewal.py`,
`research_handoff_hazard.py`, `research_stoppage_hysteresis.py`,
`research_neural_control.py`, and `research_island_endpoints.py`, in that order
after their prerequisite caches exist, each in a fresh checkout/output state.
They are not required to regenerate the selected candidate.

The full builder verifies actual competition-input, source, public-label and
feature/weak-cache hashes; rechecks weak/public/reserved-window exclusion; and
refits both new heads with seed 42. Reloaded new models reproduce every saved
full probability exactly. The four retained checkpoints/probabilities are
unchanged, and disabling the new step reproduces the anchored CSV exactly.

## Final artifact and review

The selected full CSV is `data/candidate_stoppage_consensus/candidate.csv`,
SHA256 `dac48e6be66f4395089cae4eee3cfb5daeadcd7963480f399fe079d871be534c`.
Two fresh refits/builds produce identical CSVs and manifests. The official
validator accepts all 199,319 rows, frames 0–199318. Counts are H=49,429,
A=31,420, D=78,008, X=40,462. Exactly 3,529 labels change relative to the
anchored candidate. It reports 349 possessions versus 348 previously; without
private labels those full-match changes cannot be judged correct.

All 66 tests pass. All 18 confirmation arrays pass shared-inference parity and
agree with the official scorer. The local integrity audit verifies source/input,
cache/model/probability and CSV hashes; both new models reload exactly; retained
models/probabilities and all prior 817 ledger rows remain unchanged. The frozen
annotation metadata hash and every weak-sample exclusion verify.

Independent Standards and Spec reviews found two issues before promotion: an
understated context footprint and an incomplete cache-source guard. Both were
fixed. The original caches, confirmation, full builds, generating scripts and
logs remain under `data/control_renewal_v1/provenance_before/`. Freshly regenerated
features and weak samples equal the originals, all 18 OOF predictions and all
head probabilities match, and final CSV predictions are unchanged. This repair
changes provenance checks and documentation, not model settings or predictions.

[STOPPAGE_CODE_REVIEW.md](STOPPAGE_CODE_REVIEW.md) records the separate review
axes and resolutions. `data/control_renewal_v1/integrity.json` records completed
checks; `verify_stoppage_candidate.py` repeats the local history-specific audit.
Those preserved-history checks are separate from the ordinary fresh reproduction
chain above. Earlier selected entry points and every candidate remain available.
