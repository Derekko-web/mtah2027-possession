# Continuation report — 21 October 2026

Historical first continuation round. The subsequent expanded search found a
public development improvement; see [SELECTED_MODEL.md](SELECTED_MODEL.md) and
[EXPANDED_RESEARCH.md](EXPANDED_RESEARCH.md). Conclusions below apply to this
earlier three-alternative round.

**No reliable improvement was established in this round. Keep the existing selected
ensemble and full-match candidate.** Three fixed alternatives were tested, all
failed the promotion screen recorded before their scores were seen. The candidate
was rebuilt unchanged and validated; it has not been submitted or uploaded.

## Results and decision

All scores below use held-out public predictions and the official scorer. Training
retains five held-out minutes and ten-second buffers, two shallow XGBoost models,
team swapping, equal probability blending, the occlusion continuity prior and
51-frame probability smoothing. The only changes are those named in the table.

| Method | Stitched OOF spellF1 | MacroF1 | Mean clipped-minute spellF1 | Predicted possessions | Missed stoppages* |
|---|---:|---:|---:|---:|---:|
| Organiser smooth50 | 0.3256 | 0.6207 | 0.3653 | 26 | — |
| Organiser smooth125 | 0.4242 | 0.5812 | 0.4444 | 16 | — |
| Selected, reproduced | 0.5231 | 0.7278 | 0.5996 | 22 | 2 |
| Add robust ball motion | 0.5231 | 0.7278 | 0.5996 | 22 | 2 |
| Replace slot-relative motion | 0.5231 | 0.7277 | 0.5996 | 22 | 2 |
| Increase switch penalty to 25 | 0.5793 | 0.7462 | 0.5981 | 12 | 2 |

*A diagnostic definition: a true D run with less than 50% D recall. There are four
public stoppages and 17 public possessions. This diagnostic is not an official
ranking metric. Organiser smoothing retains its original full-match context.

| Held-out minute | Selected | Robust motion | Slot-invariant motion | Penalty 25 |
|---|---:|---:|---:|---:|
| 1 | 0.4400 | 0.4400 | 0.4400 | 0.4571 |
| 2 | 0.9000 | 0.9000 | 0.9000 | 0.9000 |
| 3 | 0.6182 | 0.6182 | 0.6182 | 0.5333 |
| 4 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |
| 5 | 0.6400 | 0.6400 | 0.6400 | 0.7000 |

The frozen screening rule required at least +0.02 stitched and mean-minute
spellF1, no minute regression worse than 0.05, fewer missed stoppages, at most
25 additional false-D frames, and acceptable performance under whole-spell
exclusion. These are practical development gates, not a significance test.
The motion alternatives provide no possession-score gain. Penalty 25 fails the
mean-minute, worst-minute and missed-stoppage gates.

Its higher stitched score is a precision/recall tradeoff. At IoU 0.5 the selected
model matches 11 true possessions, misses 6 and has 11 unmatched predictions.
Penalty 25 matches 8, misses 9 and has 4 unmatched predictions. It erases the real
0.76-second H possession at 15346–15364, which the selected model mostly labels
correctly. Fewer predicted possessions alone does not demonstrate better control
detection. The official 1,000-draw 90% spellF1 bootstrap intervals overlap widely:
selected 0.3897–0.6564; penalty 25 0.4000–0.7553. They are diagnostic marginal
intervals, not a paired test or a correction for adaptive selection.

## What the error sequences show

The selected OOF sequence completely misses both visible-ball D runs,
15889–16399 and 18027–18388. Their D recall remains zero for every new alternative.
Overall D recall is only 33.36%, but D precision is 99.59%: only two live frames
are incorrectly called dead. The current failure is a conservative and
unrepresentative dead/live detector; stronger temporal regularization does not
supply missing evidence of stopped play.

The robust-motion hypothesis adds seven inputs: ball coordinate interquartile
spread over one and two seconds, observation support, median half-second
displacement in past/future one-second windows, and the fraction of slow steps.
It tests whether tracking jitter hides stationary balls. Missing observations
remain missing. Observation context is bounded within two seconds.

The slot experiment replaces four relative-velocity inputs with nearest
same-team position-set matches at a 0.2-second separation. It tests whether slot
switches corrupt player/ball velocity. Ambiguous equal-distance matches become
missing, independent of slot order. Close teammates can still be mismatched;
these are experimental inputs, not corrected identities. Neither hypothesis
established an improvement with the unchanged learning and decoding pipeline.

## A stricter validation stress check

Original minute folds can leave part of the same true possession in training.
For each held-out minute, the additional protocol removes every complete H/A/D
run intersecting that minute, while retaining the original ten-second exclusion
buffers. Labels define group exclusions only; held-out labels do not enter
features, targets, sample weights, smoothing or decoding.

| Method | Whole-spell exclusion stitched spellF1 | Mean clipped-minute spellF1 |
|---|---:|---:|
| Selected | 0.4923 | 0.5764 |
| Robust motion | 0.4923 | 0.5764 |
| Slot-invariant motion | 0.4923 | 0.5764 |
| Penalty 25 | 0.5793 | 0.5981 |

Selected per-minute scores become 0.4400, 0.9000, 0.5818, 0.4000, 0.5600.
This suggests some sensitivity to training on other portions of held-out spells.
It is still an adaptive stress check on the same previously examined five
minutes, with overlapping training sets. It is not an independent test, a new-match
evaluation, or evidence of private full-match performance. Stitched fold joins
and clipping at minute boundaries continue to affect the two reported scores.

## Evidence needed next

The [official rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
explicitly allow small hand-labelled training/validation sets, while requiring
algorithmic final predictions. The [FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md)
prohibits sending competition material or extracts to hosted/cloud LLMs. Rules
were checked against the current official repository on 21 October 2026.
No competition video or tracking was uploaded; numerical processing ran locally.

`freeze_annotation_windows.py` has frozen 24 local 30-second windows:

- Twelve training windows from the first half, outside the public labels and a
  ten-second buffer: six around automatic restart tags, six random.
- Twelve audit windows from the second half with the same strata, never used for
  training or tuning. Windows do not overlap and have ten-second spacing.

The manifest is `data/continuation_20261021/annotation_freeze/windows.json`, SHA256
`5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c`.
It is a sampling plan only: **no new annotations or independent evaluation
results exist yet**. Selection uses half bounds, automatic tags and seed 42,
without model predictions. The original model checkpoint and CSV remain in the
preserved directory for a future frozen comparison.

A local annotator should review footage without seeing model predictions, mark
control and dead/live transitions, and record ambiguous boundaries. Review a
subset with a second annotator. Keep audit annotations unavailable during model
development; an auditor should evaluate one frozen comparison. Score the two
sampling strata separately, since restart enrichment changes their prevalence.
These human/video labels would be a proxy audit, explicitly separate from the
mechanical corrected-tracking target. They can test visible-stoppage failures and
whether extra training examples help, but cannot establish the private official
score. A local video detector or restart-boundary model becomes worth testing
once representative training examples and that reserved audit exist.

## Artifacts, reproducibility and checks

`EXPERIMENT_LEDGER.csv` consolidates all 65 historical score rows (including
historical reference rows) and eight evaluations in this round: three new
alternatives under two protocols, plus the reproduced/stress-tested selection.
It retains failures, per-minute scores when available, source paths and hashes.
It does not mean 73 independent experiments. Earlier exploratory chronological
protocols remain documented in `README.md` and their preserved outputs.

All new predictions, models, splits, segment diagnostics, manifests and official
reports are under ignored `data/continuation_20261021/`. The selected source/model
snapshot is in its `preserved/` directory. Tracking data links remain intact.
Nothing has been committed, published or submitted; `experiments/` remains
untracked as at handoff. No private/corrected full-match labels were accessed.

Run from the repository root using the existing Python 3.12 environment:

```sh
.venv/bin/python -m pip install -r experiments/requirements.txt
.venv/bin/python experiments/run_selected.py
.venv/bin/python experiments/sequence_diagnostics.py
.venv/bin/python experiments/run_continuation.py
cp evaluation/results/baseline_smooth50.csv evaluation/results/baseline_smooth125.csv data/continuation_20261021/buffered_minutes/scorer_inputs/
cp evaluation/results/baseline_smooth50.csv evaluation/results/baseline_smooth125.csv data/continuation_20261021/purged_spells/scorer_inputs/
.venv/bin/python evaluation/run_evaluation.py --gt public --results data/continuation_20261021/buffered_minutes/scorer_inputs --out data/continuation_20261021/buffered_minutes/official --bootstrap 1000 --no-detail
.venv/bin/python evaluation/run_evaluation.py --gt public --results data/continuation_20261021/purged_spells/scorer_inputs --out data/continuation_20261021/purged_spells/official --bootstrap 1000 --no-detail
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/freeze_annotation_windows.py
.venv/bin/python experiments/build_candidate.py
.venv/bin/python evaluation/validate_submission.py data/candidate/candidate.csv
```

With the existing historical artifacts, regenerate the consolidated ledger using
`.venv/bin/python experiments/update_experiment_ledger.py`. This also checks
agreement between the new in-process metrics and official scorer reports.
The saved historical ledger remains readable without private data artifacts.

`sequence_diagnostics.py --require-stoppages` intentionally exits with failure
until all public stoppages have at least 50% D recall. That observed failure
remains unresolved; it is not part of the unit test pass criterion.

Twenty-seven synthetic/unit checks pass: the original 18 plus bounded motion,
missingness, physical units, arbitrary slot permutations, ambiguous matches,
whole-spell purging, exclusion of held-out fitting labels/weights, sequence
diagnostics and deterministic disjoint annotation sampling. Two-axis code review
found and fixed the slot-order tie bug before the final rerun; no actionable
review findings remain in that scope. The official scorer independently agrees
with the recorded metrics under both protocols.

`data/candidate/candidate.csv` contains exactly 199,319 rows and remains the
existing selected model's file, SHA256
`4e1cfbcf5d346ddead9de0adec420873947298aa4a2bde189d88d9e14b420fa1`.
The rebuild matches the preserved file byte-for-byte. The format validator passes;
this confirms alignment and format, not prediction accuracy. Its public labels
were used for fitting, so its training-fit score is not reported as validation.
