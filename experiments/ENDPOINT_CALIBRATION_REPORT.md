# Nested endpoint calibration — 21 October 2026

The selected six-model candidate remains unchanged. Training-only learned timing
corrections fail the frozen primary-score checks. This round adds 15 dependent
evaluation rows, bringing the ledger to **1,081**, with all preceding 1,066 rows
preserved. All nested fits, calibration cases, failures and comparisons remain
available under `data/control_calibration_v1/`.

The buffered reference is still spellF1 **.729412**, MacroF1 **.867912**, with
3 of 6 short possessions matched at IoU>=.5. Neither the missed short possessions
nor the overlong 19-frame endpoint is fixed by this method. No maximum or private
full-match accuracy has been established.

## Why this differs from the failed endpoint screens

Finer likelihoods and contact-release rules previously failed to locate the
19-frame possession's ending. Applying a manually chosen contraction would
optimize a repeatedly inspected public example without demonstrating a learned
rule. This round instead estimates timing corrections from nested out-of-fold
predictions on eligible training data only.

[ENDPOINT_CALIBRATION_PLAN.json](ENDPOINT_CALIBRATION_PLAN.json) froze three
hypotheses: a repeatable ending bias, different start/end biases, and instability
from a single calibration example. Four methods compare end-only versus separate
start/end corrections, requiring either one or two matched training examples.
The protected source/candidate/confirmation archive was taken before the plan
and experiments. The short-precision diagnostic was run and failed before work.

## Nested fitting and inference

Outer folds are the existing buffered-minute, whole-run-purged and buffered
100-second splits. Each outer training set is subdivided using three fixed
100-second inner chunks. Inner validation is its intersection with the outer
training set. Inner fitting excludes the entire inner chunk plus 250 frames on
each side, and stays inside the outer training set.

All six selected models are refitted for every usable inner fold: ordered
XGBoost, both previous-ensemble objectives, fast control, and both weak D heads.
Their features and settings remain unchanged. Outer-held labels are replaced
with X before fitting or calibration. The existing excluded weak D samples are
unchanged and remain outside the public and reserved annotation windows.

Before nested work, a complete refit of the first outer fold reproduced every
cached raw probability exactly for all six heads. There are then **36 usable
inner fits**, plus three empty-validation skips in the longer protocol. No fold
is skipped for missing target classes. Each disjoint eligible validation
component is decoded independently through canonical selected inference.

Model-derived short H/A islands of 8..125 frames, with opposite live neighbors,
are matched one-to-one to same-label true training intervals by maximal total
IoU. Only matches at IoU>=.3 remain; true intervals touching a component boundary
are excluded because they may be clipped. The targets are true-minus-predicted
start and exclusive-end positions. There are just **13 matched case rows** across
all outer folds, with repeated underlying possessions. These are not 13 new or
independent annotations.

Corrections are coordinatewise medians, clipped to +/-25 frames. Insufficient
cases produce zero corrections. At outer inference, a joint ordered-boundary
optimization applies those learned preferences only to eligible reference
islands. It keeps the entire original run-label sequence, every D decision,
positive run lengths and at least eight frames in each eligible island. It
cannot cross a boundary or create a missing possession. No public endpoint or
frame ID enters inference.

## Results

Seed42, official public scorer:

| Method | Buffered spellF1 | Purged spellF1 | 100-second spellF1 |
|---|---:|---:|---:|
| Selected reference | .729412 | .662857 | .659459 |
| End-only, >=1 example | .717647 | .651429 | .659459 |
| End-only, >=2 examples | .729412 | .662857 | .659459 |
| Both ends, >=1 example | .705882 | .628571 | .648649 |
| Both ends, >=2 examples | .729412 | .651429 | .659459 |

No method adds a short IoU>=.5 match in any protocol. Both one-example methods
worsen buffered and purged primary. Requiring two examples avoids most changes
and losses but provides no primary gain. Buffered mean short IoU falls
.399285→.370919 for one-example end-only and →.393557 for two-example end-only.
The corresponding two-ended values are .346798 and .392424.

The inner-training endings mostly need extension, not contraction. Buffered
outer-fold end medians are **+21, +7, +17, -21 and +11 frames**. Purged medians
are +21, +7, +17, 0 and +17; longer-block values are 0, +22 and +17. Zero here
means no usable example. Matched-case counts by outer fold are 1/2/1/1/1,
1/2/1/0/1 and 0/1/1, respectively. Only one outer fold has a negative ending
correction. The two-example rule therefore acts in only one buffered and one
purged fold, and nowhere in the longer protocol.

This does not demonstrate a universal bias of the selected full-fit model.
Inner models have less training data and can make different errors. It does
show that the proposed correction is not reliably supported by the available
training-only examples. A hand-chosen trim based on the known 19-frame example
would not follow these learned estimates.

Longer-block short mean IoU improves .396764→.452741 with one-example end-only,
and MacroF1 .842747→.844156, while primary stays unchanged. Two-ended calibration
raises short IoU further to .464062 but reduces primary to .648649. These are
another example of a favorable diagnostic average failing to improve the ranking
metric. Every D mask is exactly unchanged. There are eight primary regressions
among 52 method/block comparisons; all block deltas are retained.

## Decision and verification

None passes the prospective requirements: no primary or mean-short-IoU
regression; MacroF1 within .005; no lost previously recovered short run;
identical D masks; buffered primary gain>=.04 with an additional short match;
and positive primary gains in each other protocol. There is no finalist for
seeds43/44, full-match build or promotion.

The audit verifies every nested fitting and matching index against the outer
training pool, including the 250-frame inner exclusions and disjoint component
boundaries. Every training case and correction is regenerated exactly from
saved inner predictions and eligible training labels. All **222 checkpoints**
(216 inner plus six preflight) reload to their exact saved probabilities.
All 36 inner inference outputs and all 15 outer screen predictions replay
exactly. All 15 score rows agree with the official CLI scorer.

Actual competition-input, source, weak-cache and feature identities verify.
The selected CSV and its source hashes are unchanged; all 1,066 preceding ledger
rows are preserved by identity. All **82 tests pass**, including nested coverage
and exclusion, clipped-target matching, median/fallback rules, exhaustive timing
optimization, adjacent-island constraints, D preservation and team symmetry.

The selected full CSV remains SHA256
`dac48e6be66f4395089cae4eee3cfb5daeadcd7963480f399fe079d871be534c`.
The completed history-specific audit is `data/control_calibration_v1/integrity.json`.

With the existing selected feature/weak caches and confirmation present, use
Python3.12 in `.venv`, pinned research dependencies and a fresh output directory:

```sh
.venv/bin/python experiments/research_endpoint_calibration.py --output data/control_calibration_v1/screen_reproduce
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/require_short_precision.py
```

The final diagnostic intentionally remains red. `finalize_endpoint_calibration.py`
audits the original local output and preserved pre-round archive; it is separate
from ordinary fresh screen generation. The output retains nested indices,
training-target/checkpoint hashes, all probabilities and predictions, timing
cases, calibrators, fold/segment/block tables, plans, official reports and logs.
`update_experiment_ledger.py` includes the new 15 rows.

This remains heavily adaptive same-match development on 17 public possessions.
Nested fitting prevents direct outer-target leakage; it does not create an
independent test or correct repeated selection bias. Further generic timing
adjustments need a new reason. Obtaining different ownership information is a
more plausible next investigation than choosing another trim parameter.
No media, private labels, new annotations or reserved audit content were read.
Nothing was committed, submitted, uploaded or published.
