# Reviewed annotation pilot — used, not promoted

Completed 2026-10-24. The user's request to use the work completed so far is
implemented as an immutable **reviewed-subset** training experiment. Reviewed
clips **1, 3 and 4** contribute **2,250 frames / 90 seconds**, spanning two uniform
windows and one restart window. Clip2 was an incomplete, unreviewed draft when
the snapshot was taken and is excluded. Later annotation edits cannot change
this experiment's snapshot.

The added-supervision model regresses all three existing public protocols.
**Selected8 remains the current candidate.** No confirmation seeds, full-match
candidate, promotion or sealed-audit inspection follow the failed frozen gate.

| Public protocol | Selected8 spellF1 | With reviewed labels | Difference |
|---|---:|---:|---:|
| Buffered minutes | .741176 | .720000 | -.021176 |
| Purged spells | .685714 | .666667 | -.019048 |
| Buffered100 seconds | .670270 | .604878 | -.065392 |

## What changed

[HUMAN_ANNOTATION_PLAN.json](HUMAN_ANNOTATION_PLAN.json) was frozen before fitting,
SHA256 `8d2726da4aa58446053d62877f13ed63130f4607c57af979756d4835708971cf`.
Only the original ordered102 current head (slot0) is fitted with added human
supervision. Its existing depth3 /160-tree parameters, seed42, class-balanced
public targets, H/A symmetry and prediction remain. No representation, threshold,
calibration, early stopping or model-family search is added.

Public training-row weights stay exact. Human targets are separately balanced
after team swapping, with total weight mass fixed at **.5 of public mass**.
They comprise 1,035H /890A /325D frames; no U frames occur in these reviewed clips.
The helper excludes U and requires complete coverage, distinct reviewer identity,
frozen media/source-frame provenance and train-window membership. Human video
labels are additional supervision, not official corrected-tracking targets.

All13 original fold checkpoints reload to exact saved probabilities and reproduce
the complete selected8 H/A/D/X sequence **before any alternate fit**. Then13 new
checkpoints cover the existing5 buffered-minute,5 purged and3 longer-block folds.
Only slot0 changes through the unchanged full selected8 pipeline. A frozen final
overlay accepts alternate H/A only where both sequences are live; selected D/X
stay exact. The other seven model artifacts remain untouched. Human annotation
frames never become inference features or direct submission patches.

The existing annotation UI still requires all12 training clips for its final
export. The user's explicit request to use completed work now authorizes this
separate partial snapshot; it does not mark unfinished clips reviewed, relax the
UI gate, or open the12 sealed audit windows.

## Results and decision

All previously matched GT possessions remain matched at official IoU.5, but there
are **no newly matched possessions**. Official short matches remain3 of6. Spurious
predictions increase4→5,5→6 and7→11; missed GT counts remain4 in every protocol.
MacroF1 falls .870833→.862990, .871604→.865097 and .844383→.839945. Mean-fold
primary also regresses everywhere. Buffered/purged short mean IoU decreases;
longer-block short IoU increases but does not recover another short match.
False-D counts remain88/142/16 because final D/X is fixed by design.

The original buffered +.04 /extra-short-match objectives and several secondary
protections fail. No weight sweep, decoder change or seed43/44 confirmation
follows. This rejects **one fixed way of adding these90 seconds**. It does not
establish that the annotations are incorrect, that more human supervision cannot
help, or that any private/full-match score is known. Public protocols still reuse
the same17 adaptively inspected possessions. No fresh human holdout was evaluated.

## Verification and preservation

Two actual feature-prefix lengths, differing by125 rows and both padded beyond
all included windows, reproduce every public and human training feature exactly.
Public values equal the original selected102 cache. The prefix ends before sealed
audit context. A second process independently rebuilds both prefixes; ordered
features and their manifest replay byte-for-byte.

A fresh saved-model replay verifies every training index, augmented target and
weight, reloads all13 new and13 original checkpoints, and reproduces public
probabilities, prediction CSVs, all6 scores, fold/segment diagnostics and official
comparisons exactly. Replacing all non-fitting public labels withX leaves fitting
targets/weights unchanged. All **296 tests pass**, including7 new synthetic
checks for review/provenance, U exclusion, buffers, source weighting, held-label
isolation and ledger append integrity.

Six verified dependent rows append to the ledger, preserving every preceding
byte: **1,483 total =778 inherited +705 continuation**. Repeated references and
protocols are not independent experiments. All17 selected artifacts, the original
annotation freeze, both source symlinks, every prior candidate and preceding
selection/history text remain preserved. The selected CSV SHA256 stays
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.

Evidence is local and ignored under `data/control_human_annotations_v1/`:
`annotation_snapshot/`, frozen plan /319-source archive, both feature builds,
13new checkpoints and fitting records, primary/replay scores and official reports,
test logs and integrity records. No competition images, private/corrected targets,
new manual labels, hosted media, installs, commits, uploads, publication,
submissions or external contacts occur. The annotation server and saved human
annotations are left available for further work.

The fitting / replay entry point is `research_human_annotations.py`; add
`--verify-only` to reload existing models. It requires a fresh `screen_replay/`
destination and deliberately preserves any previous replay. The completed replay
is retained in this workspace. Feature preparation likewise requires a fresh
`--output` folder. This experiment is complete; no follow-up fit has started.
