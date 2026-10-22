# Short-control refinement — 21 October 2026

The new default recovers the previously missed 0.76-second possession and reduces
false dead-ball predictions while retaining the preceding model's gains.
Its full-match file is `data/candidate_control_consensus/candidate.csv`.
The ordered XGBoost candidate and all earlier models remain preserved.

## Held-out results

These are official-scorer results on the same repeatedly inspected public labels.
All training exclusions, 250-frame buffers and padded observations are retained.

| Buffered minute folds, seed 42 | Previous default | Consensus refinement |
|---|---:|---:|
| Stitched spellF1 | 0.6125 | **0.6824** |
| MacroF1 | 0.8341 | **0.8373** |
| Mean clipped-minute spellF1 | 0.6500 | **0.7133** |
| False dead-ball frames | 164 | **140** |
| Dead-ball recall | 66.71% | **66.71%** |
| Mean recall of true live runs <=5 seconds | 34.54% | **51.59%** |
| Short live runs with >=50% recall | 2 of 6 | **3 of 6** |

The primary score improves by 0.0699 (11.4% relative). False dead-ball time falls
by 24 frames, or 0.96 seconds, a 14.6% reduction. This is a modest false-D
improvement; most of that error remains. Dead recall and the number of mostly
missed stoppages are unchanged in every paired seed/protocol comparison.

The true H run at 15346–15364 now has 100% frame recall, up from zero. The predicted
H interval is 15346–15386: its start is recovered but its end extends 22 frames
too far, giving IoU 0.463. It still fails the 0.5 IoU threshold as an individual
short possession. Across all possessions, IoU-0.5 matches rise 11→12 and misses
fall 6→5, while spurious predictions also rise 4→5. Predicted possessions become
17, matching the true count incidentally; this does not mean perfect intervals.

The 71-frame A and 33-frame H possessions remain unrecognized. The 87-frame H
possession improves only from 17.2% to 19.5% recall. This is one recovered brief
possession, not a solution to every short-control error.

| Held-out minute | Previous default | Refined |
|---|---:|---:|
| 1 | 0.6500 | 0.6500 |
| 2 | 0.9500 | 0.9500 |
| 3 | 0.5333 | 0.8500 |
| 4 | 0.4500 | 0.4500 |
| 5 | 0.6667 | 0.6667 |

## Confirmation and limits

Two variants were frozen after the initial screen in
`CONTROL_CONFIRMATION_PLAN.json`: conservative contact attenuation (0.25) and
the preferred stronger attenuation (0.75), with fast-model threshold 0.65.
Both pass the recorded checks. The preferred variant reduces false D more;
the conservative variant has slightly better purged primary scores.

| Seed-42 protocol | Previous spellF1 | Refined spellF1 | Previous mean block | Refined mean block |
|---|---:|---:|---:|---:|
| Five buffered minutes | 0.6125 | 0.6824 | 0.6500 | 0.7133 |
| Five minutes, whole intersecting runs excluded | 0.5750 | 0.6400 | 0.6233 | 0.6840 |
| Three buffered 100-second blocks | 0.5765 | 0.6444 | 0.5660 | 0.6255 |

Across seeds 42/43/44, refined stitched scores are respectively 0.6824 in every
buffered run, 0.6400 in every purged run and 0.6444 in every longer-block run.
All nine paired primary scores improve. Buffered false-D frames fall from
164/167/175 to 140/139/141; purged counts fall from 232/229/226 to 175/174/174.
Longer-block false-D counts are unchanged (16/15/15). D recall is exactly unchanged
in all nine pairs. Mostly missed stoppages remain 1/0/1 across the three protocols.

There are accepted secondary costs: live-run fragments increase 11→12 in the
main comparison; purged minute 5 falls from 0.5333 to 0.5200; longer-block MacroF1
falls from 0.801875 to 0.801468. The frozen tolerance allowed at most 0.005
MacroF1 loss. The longer-block protocol still misses the final visible-ball
stoppage completely. Other block scores and seeds are retained in the tables.

Official 1,000-draw 90% possession-bootstrap intervals overlap: previous
0.4375–0.7875, refined 0.5176–0.8353. These are marginal intervals, not a paired
or selection-adjusted significance test. All protocols reuse the same 17 public
possessions with overlapping training sets. The frozen annotation windows remain
unlabelled and unused. No private/corrected fixture labels were accessed.

The improvements are useful development evidence, not independent confirmation,
statistical proof or a measured full-match accuracy gain. Full-fit public scores
are not used as validation. Nothing has been submitted or uploaded.

## What changed

The current model already assigns low raw probability to the missing brief
possession. Changing smoothing alone cannot recover absent ownership evidence.
A faster live-only model recovers it, but broad use creates many false changes.
Reducing D confidence globally also loses real stoppages.

The selected refinement therefore uses consensus among complementary models:

1. Keep the preceding 102-feature, class-balanced ordered XGBoost model unchanged.
   Its full-match raw probabilities are checked for exact equality to the saved
   preceding model output.
2. Use the older 46-feature ensemble to propose brief H/A runs in its automatically
   decoded sequence. Eligible proposals last 8–125 frames, have surrounding
   opposite ownership in the current model, and are supported by repeated close
   control. Neither true intervals nor frame IDs define these proposals.
3. A live-only binary XGBoost model must independently support that team with
   mean probability >=0.65 over the proposal. It uses immediate geometry/motion,
   without absolute pitch-line distances or long control-history averages.
   H/A-swapped training makes its two target classes balanced. D rows are excluded
   using eligible training labels only.
4. Change ownership probabilities inside qualifying proposals, retain current
   ownership elsewhere, then run the ordinary penalty-3 decoder. Main and proposal
   probabilities use 51-frame smoothing; fast probabilities use 11 frames.

Close control means ball distance <2 m and >0.5 m advantage over the other team.
At least three such observations in a centered 25-frame window establish support;
the proposal needs at least three supported frames. A separate centered 13-frame
fraction of close-control observations with ball displacement >4.5 m/s supplies
soft live evidence. The D probability is multiplied by `1 - 0.75 * evidence`;
remaining mass is assigned using the chosen ownership probabilities. Missing
tracking supplies no contact evidence. This retains all current D recall in the
held-out comparisons while shortening some false dead-ball extensions.

`consensus_control.py` is shared by confirmation and full-match generation.
Training uses only explicit eligible indices. No calibration, weights, fitting
or inference reads held-out targets. Decoding resets at each held-out block or
half, and full output abstains outside the supplied half bounds. This is an
offline predictor using future observations and globally decoded model proposals.

## Rejected experiments and provenance

Every scored alternative remains in `EXPERIMENT_LEDGER.csv` and ignored outputs:

- `research_control_refinement.py`: fast/ordered live-only heads, weaker D-class
  training weights, unconditional D reductions, and contact-based live evidence.
- `research_supported_control.py`: contact-gated fast ownership still fragments
  too many possessions.
- `research_control_blends.py`: broad model blending or fast short bouts alone
  do not preserve the primary gains.
- `research_control_consensus.py`: consensus gates recover the brief possession
  with fewer added transitions. Threshold 0.75 rejects that recovery; 0.60/0.65
  behave similarly in the initial screen.
- `research_consensus_boundaries.py`: extra confidence blending at proposal edges
  does not solve the remaining short boundaries. A 0.5 mixture loses the recovery;
  minor favorable scores at other mixtures were not used to change the frozen
  selected recipe.

The ledger contains the prior 595 rows plus 183 new dependent evaluations
(including references, protocols, decoders and seeds), totaling 778. These are
not independent experiments. The prior source/candidate snapshot is in
`data/control_refinement/preserved/`.

## Reproduce and inspect

```sh
.venv/bin/python -m pip install -r experiments/requirements-research.txt
.venv/bin/python experiments/validate_temporal_candidate.py
.venv/bin/python experiments/finalize_temporal_validation.py
.venv/bin/python experiments/confirm_control_refinement.py
.venv/bin/python experiments/finalize_control_refinement.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_control_candidate.py
.venv/bin/python evaluation/validate_submission.py data/candidate_control_consensus/candidate.csv
```

The first two commands after dependency installation recreate the prerequisite
retained/proposal OOF probabilities. Confirmation rebuilds padded features,
checks equality to those prerequisites, and trains only the fast head. Finalization
verifies all paired gates, exact shared-decoder parity for all 18 candidate OOF
outputs and official scorer agreement for all 27 candidate/reference evaluations.
Forty-five targeted tests pass.

OOF predictions, fast models, split references, segment diagnostics, frozen plan,
promotion checks and official reports are under `data/control_confirmation/`.
OOF files contain X outside the public window and are not full-match submissions.
The full candidate directory contains four model checkpoints, raw probabilities,
source/input/output hashes and the manifest. Earlier full candidates are protected
from accidental overwrite. No competition-derived artifacts enter public history.

Two independently generated full CSVs match byte-for-byte, SHA256
`19dd8ecac4fe0c3fcbdc11d18d41cd9a38d1cf995a0defb05ff816b6057a64f1`.
The official validator accepts all 199,319 rows (frames 0–199318).
Counts are H=51,422, A=32,526, D=74,909, X=40,462. It reports 346 full-match
possessions versus 295 previously; without labels this change cannot be judged
correct. Saved models reload to exactly the saved raw probabilities.

The independent Standards and Spec reviews found no actionable issues; see
[CONTROL_CODE_REVIEW.md](CONTROL_CODE_REVIEW.md).
