# Selected baseline: reproducible development result

Continuation on 21 October 2026 reproduced this selection and retained it.
Three additional alternatives did not establish a reliable improvement; the
full candidate rebuilt byte-for-byte. See [CONTINUATION_REPORT.md](CONTINUATION_REPORT.md)
for the fixed experiment plan, whole-spell exclusion stress check, failure
analysis, aggregate ledger and frozen plan for additional local annotations.

The selected pipeline materially improves on the provided baselines on the public
development window. It has not established full-match superiority. Only the first
five minutes have public labels, and those same minutes informed repeated feature
and postprocessing choices.

## Results

| Method | Stitched out-of-fold spellF1 | MacroF1 | Mean held-out-minute spellF1 |
|---|---:|---:|---:|
| Organiser smooth50 | 0.3256 | 0.6207 | 0.3653 |
| Organiser smooth125 | 0.4242 | 0.5812 | 0.4444 |
| Selected ensemble | **0.5231** | **0.7278** | **0.5996** |

The selected stitched score is 60.7% above smooth50 and 23.3% above smooth125.
These are relative changes in F1, not percentage-point improvements in accuracy.
The mean-minute gain over smooth125 is 34.9%. The model predicts 22 possessions
over the public window, versus 17 true ones; false changes remain.

| Held-out public minute | Selected | smooth50 | smooth125 |
|---|---:|---:|---:|
| 1 | 0.4400 | 0.3600 | 0.3556 |
| 2 | 0.9000 | 0.6667 | 0.7111 |
| 3 | 0.6182 | 0.4000 | 0.5333 |
| 4 | 0.4000 | 0.1091 | 0.4000 |
| 5 | 0.6400 | 0.2909 | 0.2222 |

The selected method beats smooth50 in every block. It beats smooth125 in four
blocks and ties the fifth. These blocks are chronological positions inside the
same public five minutes, not five independent matches.

Each minute is held out once. Training uses other public frames outside a
ten-second buffer on each side. Features are extracted from inputs only; no
validation labels enter fitting, sample weights, imputation or decoding. Training
sets overlap. The organiser predictions use their original full-match smoothing
context. Stitched predictions come from different held-out models and can have
artificial changes at fold boundaries. Per-minute scores truncate possessions at
minute boundaries; their mean is not the official full-match metric.

## What changed

The candidate is an equal probability blend of two shallow XGBoost classifiers:

- A class-balanced frame model.
- A model weighting contiguous eligible training label segments equally, reducing
  the dominance of long possessions. Stoppage segments also receive segment weights.

Both use 46 tracking features: nearest-player geometry; ball visibility, movement,
acceleration and trajectory coherence; player movement; and ball/player relative
velocity. Shared pitch-location features are invariant to switching ends. No match
clock, absolute frame number, player identity or hand-written frame-specific answer
is a model input. Tactical tags were explored but are not selected model features.
The current feature builder reads the tagging file while constructing its larger
intermediate table, so keep that asset available for reproduction.

Training swaps home/away features and labels to enforce team symmetry; inference
averages original and swapped predictions. A 51-frame centered probability mean
and fixed change penalty of 3 discourage unsupported changes. In active-player
ball occlusions shorter than 9.48 seconds, preceding near-ball team evidence acts
as a soft continuity prior. This is a general input-based rule, not a manually
corrected set of submission frames. All postprocessing supports offline use.

The same regularized XGBoost settings are used for both models: 120 trees, depth
2, learning rate 0.04, minimum child weight 15, L2 regularization 10, row sampling
0.8, column sampling 0.9, seed 42 and two CPU threads. A nested inner validation
check compared four tree configurations without outer labels; all five outer
folds selected the original shallow configuration. This check does not undo the
adaptive feature/postprocessing development on the public window.

## Uncertainty: substantial development gain, limited confirmation

Diagnostic paired bootstrap intervals use 20,000 resamples with seed 42:

| Reference | Paired-minute spellF1 difference, 95% interval | Paired-possession difference, 95% interval |
|---|---:|---:|
| smooth50 | +0.150 to +0.303 | +0.088 to +0.324 |
| smooth125 | +0.051 to +0.288 | **-0.049 to +0.257** |

The minute-level intervals support a positive development difference, but the
possession-level interval against smooth125 includes zero. Only 17 real
possessions are available; folds share training data and this window was used
repeatedly. The experiment ledger contains 65 scored model/postprocessor
configurations across successive protocols, including failures. Those adaptive
comparisons create selection bias. None of these intervals is an independent,
multiple-comparison-adjusted significance test.

It would therefore be misleading to claim proven statistical superiority over
smooth125 or the private full match. Additional independent labels or the official
private evaluation are needed for that claim. The failed visible-ball stoppage
and some brief changes remain clear improvement opportunities.

## Reproduce

Run from `/Users/derekko/Projects/more-than-a-hack-2027`:

```sh
.venv/bin/python -m pip install -r experiments/requirements.txt
.venv/bin/python experiments/run_selected.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_candidate.py
.venv/bin/python evaluation/validate_submission.py data/candidate/candidate.csv
```

On a fresh checkout create `.venv` first with Python 3.12. Supply the competition
tracking metadata/text and Smart Tagging XML under `data/tactical`, or configure
`MTAH_DATA_DIR`. These gated data files must stay outside public Git history.
Dependencies are pinned in `experiments/requirements.txt`. The development
feature cache and all model outputs live under gitignored `data/` directories.

`run_selected.py` prepares label-free public-window inputs, trains both objectives
on each buffered fold, and writes scores, out-of-fold predictions, a timeline,
and diagnostic intervals under `data/selected_evaluation/`.

`build_candidate.py` refits the two selected models using all 7,500 public labels,
predicts the full tactical timeline, decodes halves separately, and writes:

- `data/candidate/candidate.csv`: full-match candidate, 199,319 rows.
- `data/candidate/frame_model.ubj` and `segment_model.ubj`: trained model files.
- `data/candidate/manifest.json`: feature names, parameters, public-label hash,
  output hash, class counts and limitations.

The full-match candidate is trained on the public labels. Scoring it on those
same labels is a training-fit score and must not replace the out-of-fold result.
The official validator confirms file format only, not prediction accuracy.

For official-scorer verification of the development result, use the saved
`data/selected_evaluation/oof_public_only.csv`: its public rows are out of fold,
and all other rows are X. Do not submit that diagnostic file as the full-match
candidate. No files have been uploaded or submitted to the competition.

## Checks

Eighteen targeted tests pass, including physical distance/speed units, bounded
feature context, team-swap symmetry, missing data, validation buffers, segment
weights, occlusion continuity, and decoder optimality against exhaustive search.
The nearest-player feature extraction reproduces all 7,500 shipped raw-baseline
public labels. The official scoring script independently reports selected OOF
spellF1 0.5231; the full candidate passes the official format validator.

## Explored alternatives

Richer long-window geometry, tactical tags, separate dead/live and team models,
random forests, extra trees, soft baseline anchoring, learned Markov transitions,
equal model-family ensembles, contact rules and physical dead-ball priors were
tested. None was promoted solely for improving a single held-out minute. Several
raised frame accuracy while reducing possession-level F1. Their diagnostic outputs
remain in ignored experiment directories, and the delivered experiment ledger
preserves scored configurations rather than showing only the selected result.
