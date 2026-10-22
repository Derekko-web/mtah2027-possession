# Expanded improvement report — 21 October 2026

Historical preceding default, retained intact. The subsequent consensus control
refinement is documented in [CONTROL_REFINEMENT_REPORT.md](CONTROL_REFINEMENT_REPORT.md).

The expanded round selects a class-balanced XGBoost model with short control
history and ordered tracking observations. It gives a substantial **public
development improvement**, with better dead/live detection and fewer false
possession fragments. Private full-match performance remains unknown.

The new file is `data/candidate_temporal_xgb/candidate.csv`. The previous model,
25 original Python sources and `data/candidate/candidate.csv` are preserved;
see [PREVIOUS_MODEL.md](PREVIOUS_MODEL.md). No predictions were manually patched,
and nothing has been submitted, published or uploaded.

## Held-out results

All final comparisons rebuild observations through frame 18934, including ten
seconds after the last public label. This removes a future-feature truncation
artifact in the earlier exploratory caches. Only the original 7,500 public
labels, frames 11185–18684, supply training targets and sample weights.

| Method, five buffered minute folds | Stitched OOF spellF1 | MacroF1 | Mean clipped-minute spellF1 | Predicted possessions |
|---|---:|---:|---:|---:|
| Organiser smooth50 | 0.3256 | 0.6207 | 0.3653 | 26 |
| Organiser smooth125 | 0.4242 | 0.5812 | 0.4444 | 16 |
| Previous ensemble, padded context | 0.5231 | 0.7278 | 0.5996 | 22 |
| Selected ordered XGBoost, seed 42 | **0.6125** | **0.8341** | **0.6500** | **15** |

There are 17 true possessions. The primary gain is +0.0894, about 17.1% relative
to the previous model. This is a development score difference, not a claim of
statistical significance. The official scorer uses 1,000 bootstrap draws;
reports, scores and intervals are saved under `data/advanced_validation/`.

| Held-out minute | Previous ensemble | Ordered XGBoost |
|---|---:|---:|
| 1 | 0.4400 | 0.6500 |
| 2 | 0.9000 | 0.9500 |
| 3 | 0.6182 | 0.5333 |
| 4 | 0.4000 | 0.4500 |
| 5 | 0.6400 | 0.6667 |

The mean of clipped minute scores and the stitched score measure different
things: clipping splits intervals, and stitching joins predictions from different
fitted folds. Both are reported rather than substituted.

## Stoppages and accepted regressions

Dead recall rises from **33.29% to 66.71%**; precision falls from **99.59% to
85.66%**. Three of four true D runs reach at least 50% D recall, versus two
previously. The two previously missed visible-ball D runs, 15889–16399 and
18027–18388, now have 52.1% and 46.1% recall. The latter still counts as mostly
missed by the diagnostic threshold; neither has precise recovered boundaries.
The other two D runs have 76.1% and 96.5% recall. Under whole-run exclusion all
four reach 50%, but the longer-block protocol still completely misses the last.

False dead predictions increase from **2 to 164 frames**, an additional 6.48
seconds. The new false-D rate is 2.72% of the 6,031 true live frames. Minute 3
regresses by 0.0848. The real 19-frame H possession at 15346–15364 is lost, and
only 17% of the 87-frame H possession at 17293–17379 is recognized, versus
100% and about 75% previously. The final live possession also has lower frame
recall. Fewer predictions reflects both fewer false changes and some genuine
possessions being suppressed.

At IoU 0.5, both models match 11 true possessions and miss six, while unmatched
predictions fall from 11 to four. This improves possession precision without
reducing matched possession count at that threshold; it does not imply every
formerly recognized possession is retained. The official 90% possession-bootstrap
intervals overlap widely: previous 0.3897–0.6564, selected 0.4375–0.7875. They
are marginal diagnostic intervals, not a paired or selection-adjusted test.

The first round's <=25 additional false-D-frame and <=0.05 worst-minute-regression
gates **do not pass**. After the user requested broader exploration, the expanded
plan explicitly changed that tradeoff. `ADVANCED_VALIDATION_PLAN.json` was frozen
before scoring padded features or longer blocks. It requires primary/mean-block
gains and no MacroF1 regression in all three protocols, fewer missed stoppages
in five-fold checks, and a false-D rate below 5% of true live frames. Seed 42 was
specified in advance; seeds 43 and 44 assess sensitivity rather than supplying a
best-seed selection.

## Stress checks and limits

| Protocol, seed 42 | Previous spellF1 | New spellF1 | Previous mean block spellF1 | New mean block spellF1 |
|---|---:|---:|---:|---:|
| Five buffered minutes | 0.5231 | 0.6125 | 0.5996 | 0.6500 |
| Five minutes, whole intersecting label runs excluded | 0.4923 | 0.5750 | 0.5764 | 0.6233 |
| Three buffered 100-second blocks | 0.5579 | 0.5765 | 0.5330 | 0.5660 |

Each split excludes 250 training frames on either side of validation. Whole-run
exclusion uses labels only to define groups removed from training, never as
inputs, targets, weights or decoding hints. Every eligible frame is predicted
exactly once per method/protocol/seed by a model that did not train on it.

The longer-block gain shrinks to +0.0186, and the final visible-ball stoppage is
completely missed. Its block scores change from 0.4923/0.5250/0.5818 to
0.6545/0.4615/0.5818; the second block regresses by 0.0635. MacroF1 improves from
0.7300 to 0.8019. These results show useful but training-sensitive behavior.

The initial primary finalist was CatBoost, with buffered spellF1 0.6194 and
MacroF1 0.8507. Its three longer-block seeds score 0.5625, 0.5375 and 0.5500,
versus 0.5579 for the previous model. The ordered XGBoost finalist scores 0.5765
for all three seeds. Its five-fold spellF1 range is 0.6000–0.6125, and purged
range is 0.5750–0.5750. Every paired seed improves stitched spellF1, mean block
spellF1 and MacroF1 over the previous model in every protocol. This consistency
supports selecting the secondary finalist over a slightly better headline score.
The choice was made after reviewing stress results and is an adaptive decision,
not a predeclared independent test. No model settings or seed were tuned to those
results. Both finalists pass the original expanded seed-42 gates; the later
robustness comparison is saved separately in `selection_decision.json`.

All protocols reuse the same 17 repeatedly examined public possessions, with
overlapping training sets. Hundreds of method/protocol/decoder/seed score rows
informed the expanded search. Bootstrap intervals do not correct for adaptive
selection. These results establish a useful local development advance, not
statistically proven superiority, new-match generalization or private full-match
accuracy. Do not compare public scores directly with a published private
full-match baseline score. Scores of the final refitted CSV on its own training
labels are not validation and are intentionally not used for selection.

The frozen 24 annotation windows remain unlabelled and unused. An independent
audit or additional representative fixtures is still needed to resolve uncertainty
about visible-ball stoppages and missed short possessions.

## Implementation

`temporal_candidate.py` is shared by validation and full-match construction.
The 102 float features are the previous 46 observations, minus absolute distance
to goal/touch lines, plus ten contact-history features and 48 time taps:

- Contact means ball distance <2 m with >0.5 m advantage over the other team.
  Past fractions cover 1, 3 and 5 seconds. Recent contact decays over one second
  and expires after five; forthcoming contact is bounded to one second.
- Eight visibility, ball-motion and team-control inputs are observed at
  +/-0.2, +/-1 and +/-2 seconds. Their order can distinguish approaching,
  controlling and moving away from the ball; averages discard that distinction.

Missing observations remain missing. Neither feature group uses labels, frame
number, match clock or hand-written boundaries. Speeds use metres and seconds.
Inherited slot-based relative-motion features remain imperfect; the earlier
slot-invariant replacement did not improve the score.

The model uses XGBoost 3.4.1, `multi:softprob`, histogram trees, depth 3,
160 trees, learning rate 0.04, L2 10, minimum child weight 15, subsampling 0.8,
column sampling 0.9, seed 42 and two threads.
There is no early stopping or held-out-label-based preprocessing. Training adds
home/away-swapped rows with swapped targets and class-balanced weights; inference
averages both team orientations. Segment weighting and objective averaging were
tested and rejected for this representation.

Postprocessing retains the previous input-only active-occlusion prior, centered
51-frame probability smoothing and dynamic-programming switch penalty 3. Each
validation block and each match half resets decoding. The full CSV abstains
outside the supplied half bounds. This is an offline predictor using future
observations, not a live causal system.

The builder verifies that full-match public feature values exactly equal padded
validation values. Its manifest records parameters, feature names, public-label
hash, source/input/model/output hashes and limitations. Two independent full
builds produced identical CSV bytes, SHA256:

`bcbd287f437b2404d46a86e6d3a7c0b9fc44c30feb58ed5fdd44f84482d4ec72`

The file has 199,319 rows, frames 0–199318. Counts are H=51,196, A=32,069,
D=75,592, X=40,462. The format validator reports 295 possessions versus 429
previously, but there is no full-match accuracy measurement.

The aggregate ledger contains 595 score rows: 65 historical, eight from the
narrower continuation, 480 expanded research, 27 final method/protocol/seed
checks, and 15 preliminary scratch rows (12 later reproduced by the sequence
runner). This large search increases selection bias; it is not 595 independent
experiments. All failures remain visible. Thirty-seven targeted tests pass.

## Reproduce and inspect

To reproduce this preceding default, use `requirements-research.txt`, run
`validate_temporal_candidate.py`, `finalize_temporal_validation.py` and
`build_temporal_candidate.py`, then validate `data/candidate_temporal_xgb/candidate.csv`.
For separate validation artifacts, pass the same `--out` directory to validation and finalization. The
default candidate builder checks public features against the default validation
directory when that directory exists.

The official scorer can also be run directly, substituting `purged_spells` or
`buffered_100_seconds` for the protocol directory:

```sh
.venv/bin/python evaluation/run_evaluation.py --gt public \
  --results data/advanced_validation/buffered_minutes/scorer_inputs \
  --out data/advanced_validation/buffered_minutes/official --bootstrap 1000 --no-detail
```

`comparison.csv`, `scores.csv`, `fold_scores.csv`, `seed_summary.csv`,
`segment_diagnostics.csv`, `splits.json`, `promotion_checks.json`, `selection_decision.json`, OOF CSVs,
probabilities and model checkpoints live under `data/advanced_validation/`.
OOF files contain X outside public labels and are diagnostic files, not full-match
submissions. The aggregate [EXPERIMENT_LEDGER.csv](EXPERIMENT_LEDGER.csv) and
[EXPANDED_RESEARCH.md](EXPANDED_RESEARCH.md) retain the search and failures.
The narrower first round is preserved in [CONTINUATION_REPORT.md](CONTINUATION_REPORT.md).
