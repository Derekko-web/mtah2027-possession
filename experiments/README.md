# Possession experiment history

For the current anchored ownership and stoppage consensus, results, caveats and reproduction commands,
read [SELECTED_MODEL.md](SELECTED_MODEL.md). Its full-match CSV is
`data/candidate_stoppage_consensus/candidate.csv`. The complete prerequisite and
confirmation chain is in [STOPPAGE_RECOVERY_REPORT.md](STOPPAGE_RECOVERY_REPORT.md);
`build_stoppage_candidate.py` retains four anchored models and fits two additional
weak-supervised D heads. Use `requirements-research.txt`.
The preceding anchored selection and source remain preserved in
[ANCHORED_MODEL.md](ANCHORED_MODEL.md) and
[BOUNDARY_REFINEMENT_REPORT.md](BOUNDARY_REFINEMENT_REPORT.md).
The expanded search, including failed local audio,
video and external-data methods, is in [EXPANDED_RESEARCH.md](EXPANDED_RESEARCH.md).
The preceding consensus model remains in `data/candidate_control_consensus/`;
its reproduction commands and results are preserved in
[CONSENSUS_MODEL.md](CONSENSUS_MODEL.md) and
[CONTROL_REFINEMENT_REPORT.md](CONTROL_REFINEMENT_REPORT.md).

The preceding ordered model remains in `data/candidate_temporal_xgb/` and is
reproduced by `validate_temporal_candidate.py`, `finalize_temporal_validation.py`
and `build_temporal_candidate.py`; see [TEMPORAL_REPORT.md](TEMPORAL_REPORT.md).

The sections below retain earlier experiments and learning history.
`run_selected.py` and `build_candidate.py` reproduce the previous XGBoost model,
whose candidate remains in `data/candidate/`; see [PREVIOUS_MODEL.md](PREVIOUS_MODEL.md).

The 21 October continuation is documented in
[CONTINUATION_REPORT.md](CONTINUATION_REPORT.md). Its three fixed alternatives
were not promoted. [EXPERIMENT_LEDGER.csv](EXPERIMENT_LEDGER.csv) preserves those
65 historical rows and eight continuation evaluations, followed by all expanded
method/protocol/seed evaluations and preliminary scratch screens. These are not
independent experiments. `run_continuation.py` reproduces the narrower round
without replacing either candidate.

Run from the project root:

```sh
.venv/bin/python -m pip install -r experiments/requirements.txt
.venv/bin/python experiments/xgboost_baseline.py
```

The tactical data must be available under `data/tactical/` or `MTAH_DATA_DIR`.

The only features are the nearest home player's distance to the ball and the nearest
away player's distance, in metres. Missing distances remain NaN. Team membership
comes from metadata, not guessed shirt numbers. Frame IDs are converted from
one-based tracking IDs to zero-based video indices.

The public window is split chronologically: 180 seconds for training, 10 seconds
unused, then 110 seconds for validation. The target has three classes: H, A, D.
Frames marked X in the public answer key are never training labels. The model sees
neither frame numbers nor target-derived features. No hyperparameter search or
early stopping uses validation. Parameters and smoothing were fixed before the run.

Compare raw XGBoost and probability smoothing (+/-25 frames) against all four
organiser baselines on the same held-out frames, using the official metric library.
XGBoost smoothing is clipped to validation boundaries; the shipped organiser
baselines were smoothed over the full match. Bootstrap intervals use 1,000 draws
with the official scorer's default seed. The holdout has very few possessions and
may cut possessions at its boundaries. Results are diagnostic, not evidence of
full-match superiority. Do not repeatedly tune against this small holdout.

Generated features/predictions, a trained model and the experiment manifest are in
the gitignored `data/xgboost_baseline/`. `validation_predictions.csv` is a diagnostic
table, **not a valid competition submission**. The model is trained on only the
training block; it has not been refit using all public labels.

## First run: 2026-10-20

| Method | Validation macroF1 | Validation spellF1 | Predicted possessions |
|---|---:|---:|---:|
| Organiser smooth125 | 0.4579 | 0.3273 | 7 |
| Organiser smooth25 | 0.4904 | 0.2750 | 12 |
| Organiser smooth50 | 0.4988 | 0.2500 | 12 |
| XGBoost smooth25 | 0.4698 | 0.1882 | 13 |
| Organiser raw | 0.4794 | 0.0588 | 30 |
| XGBoost raw | 0.4497 | 0.0435 | 42 |

The holdout contains 4 ground-truth possessions. The training block has 4,500
frames; 250 frames are unused; validation has 2,750 frames. Raw nearest-player
predictions reconstructed from the features match the shipped baseline on all
7,500 publicly labelled frames. Training-feature Pearson correlation, using
pairwise observed distances, is -0.0283. About 26.5% of validation rows lack both
distances. Dead-ball examples are 138/4,500 (3.1%) in training versus 1,285/2,750
(46.7%) in validation. This distribution shift and tiny possession sample limit
interpretation; these results do not establish that XGBoost cannot help.

All results above are held out. Do not compare them directly with the README's
scores on the entire public window. No model or smoothing settings were changed
after observing these results.

## Inspecting errors

Run `.venv/bin/python experiments/inspect_xgboost_errors.py`. This reads existing
held-out predictions without retraining. It writes a timeline PNG, plus confusion
matrices and ground-truth segment summaries under `data/xgboost_baseline/`.
Pass `--out /absolute/path` to put the chart elsewhere.

In this validation block, raw XGBoost predicts D on exactly the 730 frames where
both distances are missing, and never on frames with available distances. Missing
inputs include 364 live Athletic frames and 366 stopped-play frames. This is an
observed shortcut, not a valid rule for when play stops. Smoothed XGBoost misses
918 of the 1,285 dead-ball frames and calls 367 of the 1,082 Athletic frames D.
It entirely misses the 18.60-second and 14.48-second stopped-play stretches with
available ball positions. At frame 18100 the closest home player is 1.36 m away,
the truth is D, and raw predicted P(H) is 0.989. At frame 16650 both distances
are missing, truth is A, and raw P(D) is 0.812. These examples motivate examining
motion, restart tags and temporal context rather than merely increasing tree depth.

## Second experiment: add recent ball speed

```sh
.venv/bin/python -m unittest discover -s experiments -p 'test_ball_speed.py' -v
.venv/bin/python experiments/xgboost_baseline.py --with-ball-speed
```

This adds only `ball_speed_past_second_m_s`. Speed is consecutive tracked ball
displacement in metres multiplied by 25 fps, then a trailing 25-frame median with
at least 13 observed steps. Missing positions are not interpolated. A short loss
of the ball can still leave sufficient recent observations for this feature;
longer gaps make it NaN. The one-second lookback fits inside the ten-second gap
between training and validation. Apparent motion includes tracking jitter and is
not a verified physical speed. The supplied speed field is not used.

The same model parameters, split, labels, two original distances and probability
smoothing were retained. Outputs are saved separately to the ignored
`data/xgboost_with_ball_speed/`; the first experiment is preserved.

| Method | Validation macroF1 | Validation spellF1 | Predicted possessions | Dead-ball F1 |
|---|---:|---:|---:|---:|
| Two distances, smoothed | 0.4698 | 0.1882 | 13 | 0.3635 |
| Two distances + recent speed, smoothed | 0.4763 | 0.2667 | 14 | 0.3601 |
| Two distances + recent speed, raw | 0.4537 | 0.0549 | 47 | 0.3549 |

With smoothing, correct H frames rose from 375 to 379 and correct A frames from
539 to 557; correct D frames fell from 367 to 363. At frame 16650, the missing-ball
Athletic example, recent speed is still available from prior observations and
the prediction changes from D to A. Frame 18100 remains incorrectly H during a
stoppage. Thus this run does not demonstrate improved stopped-play detection.

The smoothed speed model's official bootstrap spellF1 interval is 0.1111–0.4444
(5th–95th percentiles), overlapping the two-distance interval 0.0471–0.3529.
Only four ground-truth possessions occur in this validation block. Speed was
chosen after inspecting earlier validation errors: this is now development
feedback, not an untouched final test. No settings were tuned after this run.

Three targeted tests check known-speed units, missing-gap handling, and that
changing future positions cannot alter past speed features. The raw baseline
alignment check still matches all 7,500 public frames.

## Player-motion inspection (no retraining)

Run `.venv/bin/python experiments/inspect_player_motion.py`. Supplied player speeds
are read only from tracks with finite positions and nonnegative finite speed.
The features are the trailing-one-second median of frame-level median player
speed and trailing-one-second mean fraction of observed tracks above 1 m/s.
Each window requires at least 13 valid frames. The 1 m/s threshold is a descriptive
definition chosen before inspection, not a fitted dead-ball decision threshold.
Track slots may duplicate people; these summaries describe observed tracks,
not identified individual players. No frame-differencing of track identities is used.

All 7,500 public frames have these two summaries available. Diagnostic tables
stay under ignored `data/player_motion/`. `--out` selects the chart destination.
The first missed 18.6-second stoppage has median speed 1.015 m/s and median moving
fraction 51.2%; the following 35.72-second Athletic possession has 2.016 m/s and
90.9%. However, frame 18100 is stopped play with recent median speed 2.915 m/s
and moving fraction 100%, while live Barcelona frame 18420 has 0.888 m/s and
38.5%. Motion distinguishes some examples but does not establish a universal
stopped-play rule. This analysis did not fit or change any model.

## Chronological development comparisons

Run `.venv/bin/python experiments/compare_time_blocks.py` to compare all three
feature groups with the same fixed model settings. Public minute 3 is predicted
after training on the first 110 seconds; minute 4 after 170 seconds; minute 5
after 230 seconds. Each split has a ten-second gap and a 60-second validation
block. Features have at most a one-second trailing input window. Earlier
validation frames may become training examples in later folds, as in expanding
window evaluation; each individual prediction uses only earlier training labels.
All original experiments are preserved. Outputs, fold models, predictions and
the split manifest stay in ignored `data/time_block_comparison/`.

| Features, all smoothed | Minute 3 spellF1 | Minute 4 spellF1 | Minute 5 spellF1 | Mean |
|---|---:|---:|---:|---:|
| Distances | 0.1714 | 0.0000 | 0.3636 | 0.1784 |
| + ball speed | 0.1714 | 0.0000 | 0.3077 | 0.1597 |
| + ball speed and player motion | 0.1600 | 0.0000 | 0.2667 | 0.1422 |
| Organiser smooth50 | 0.4000 | 0.1091 | 0.2909 | 0.2667 |
| Organiser smooth125 | 0.5333 | 0.4000 | 0.2222 | 0.3852 |

Each minute has four ground-truth possessions under the scorer's clipped-window
definition. Minute 3 has no D labels. The mean weights minutes equally; it is not
a full-match competition score. The first two training blocks have only 138 D
frames, while the third has 649. The earlier speed improvement was not consistent
across these splits. On minute 5, player motion improves macroF1 from 0.4773 to
0.6256 and D-class F1 from 0.5049 to 0.6912, yet decreases spellF1 from 0.3636 to
0.2667 and increases predicted possessions from 7 to 11 against 4 real ones.
Better frame classification therefore did not yield better possession intervals.

These windows share training data, contain few possessions and have already
informed feature exploration. This is diagnostic development evidence rather
than an independent final test, and does not establish a definitive model ranking.
The organiser references retain their original full-match smoothing context.
The script verifies distance extraction against the shipped raw baseline on all
7,500 public frames and checks split separation before fitting.

## Discouraging unsupported state changes

```sh
.venv/bin/python -m unittest discover -s experiments -p 'test_transition_penalty.py' -v
.venv/bin/python experiments/evaluate_transition_penalty.py
```

`transition_penalty.decode` finds the entire H/A/D sequence minimizing
`sum(-log(probability of selected state)) + 3 * number_of_state_changes`.
It uses dynamic programming with equal costs for all transitions and no minimum
duration. The penalty 3 was fixed before running; it was not swept or tuned.
No labels enter decoding and no XGBoost models are retrained. Future probabilities
within the validation block can influence past states, so this is offline decoding.
The probabilities are correlated and imperfectly calibrated; this is a heuristic
score rather than a fitted probabilistic sequence model.

First, decoding raw probabilities removed many flips but trailed the existing
smoothing reference. We then tested the same penalty after the existing fixed
probability smoothing. This was a sequential development decision after inspecting
the first results, not an independent confirmatory experiment.

| Features | Minute 3 smoothing / + penalty | Minute 4 smoothing / + penalty | Minute 5 smoothing / + penalty |
|---|---:|---:|---:|
| Distances | 0.1714 / 0.1600 | 0 / 0 | 0.3636 / 0.3636 |
| + ball speed | 0.1714 / 0.1600 | 0 / 0 | 0.3077 / 0.4889 |
| + ball and player motion | 0.1600 / 0.1600 | 0 / 0 | 0.2667 / 0.4444 |

All values are spellF1. In minute 5, the five-feature model's predicted possessions
fall from 11 to 5 against 4 real possessions; state changes including D fall from
16 to 9 against 5 true changes. Frames 17999–18007 were incorrectly H for 0.36 s
inside true A possession; smoothing plus penalty correctly changes them to A.
The decoder also suppresses a false 0.92-second D segment inside true H possession.
Other mistakes remain, and fewer transitions alone do not establish correctness.
Penalties can erase real short possessions or stoppages. Minute 4 remains a failure.

Outputs stay under ignored `data/transition_penalty/`; `--out` selects a destination
for the chart and scores CSV. All earlier predictions are preserved. Tests compare
decoding with exhaustive enumeration on tiny sequences, verify zero-penalty
equivalence to framewise argmax, remove a brief weak spike, retain strong H/A/D
changes, and reject invalid probabilities. Original raw frame predictions are
verified against saved probabilities before scoring.

## Why validation minute 4 remains a failure

Its training block (11185–15434) has only one stopped-play segment, frames
12851–12988: 138 frames / 5.52 seconds. The ball distances are missing in 128 of
those frames (92.8%), and recent ball speed is missing in all 138. By comparison,
the missed validation stoppage (15889–16399, 20.44 seconds) has valid distances
and recent ball speed on all 511 frames. Thus the training D class mostly
represents a ball-occlusion situation, not the visible-ball stoppage it must
recognize in validation. Median player speed is similar in both (1.18 versus
1.03 m/s), but the observed five-feature model still misses the stoppage.
This supports investigating representativeness of labelled stoppages before
additional tuning; it does not prove which causal signal the model uses.

## Five buffered offline validation blocks

```sh
.venv/bin/python experiments/compare_time_blocks.py --buffered-blocks
.venv/bin/python experiments/evaluate_transition_penalty.py --buffered-blocks
.venv/bin/python -m unittest discover -s experiments -p 'test_validation_blocks.py' -v
```

Each of the five public minutes is held out once. Training uses other public
minutes, excluding ten seconds on either side of validation. First/last folds
have 230 seconds of training; interior folds have 220. Training labels can come
from later public frames, valid for this offline task. The one-second feature
lookback cannot reach across the excluded buffers. Feature sets, XGBoost settings,
smoothing and the previously chosen switch penalty remain fixed. Training blocks
are recorded separately in the manifest because they can be non-contiguous.
This evaluates interpolation within the same five-minute match window, not
forecasting or transfer to the unseen full match. Prior development already used
these labels, and the folds have overlapping training sets.

| Feature set | Mean minute spellF1, smoothing | Mean minute spellF1, smoothing + penalty |
|---|---:|---:|
| Distances | 0.2092 | 0.2740 |
| + ball speed | 0.1948 | 0.3161 |
| + ball and player movement | 0.1981 | 0.3243 |

Organiser smooth50 and smooth125 average 0.3653 and 0.4444 respectively on these
same five clipped minutes, with their full-match smoothing context retained.
The mean is an equal-weight mean of per-minute scores, not the official full-match
score. The small 0.0082 gap between the two larger feature sets is not strong
evidence for a reliable feature benefit.

For validation minute 4, D training examples increase from 138 to 903 frames.
The five-feature model after smoothing/decoding now recognizes 120/511 frames of
the previously missed stoppage (23.5%), and spellF1 rises from 0 to 0.0800.
Recognition remains incomplete; more diverse examples helped only partially.
The distance-only model recognizes 255/511 stoppage frames but still has spellF1
0 on that minute, illustrating that better D recognition alone does not guarantee
matching whole possessions. No feature set beats the organiser reference means.

Models and predictions are saved under ignored `data/buffered_block_comparison/`
and decoding results under `data/buffered_transition_penalty/`. Tests verify that
every public frame is held out exactly once, training observations and their
one-second input windows avoid validation, and the last buffered fold matches
the last chronological fold. Predictions for that identical fold reproduce the
earlier saved predictions exactly. Original chronological results are preserved.
