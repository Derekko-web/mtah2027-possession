# Round 27: DFL ownership supervision — transfer rejected

Completed 2026-10-23. **The external dataset passes feasibility and the external
ownership model passes its frozen validation gates, but the public transfer
experiment fails promotion. Selected8 remains unchanged.**

Buffered spell F1 falls from .741176 to .571429; purged from .685714 to .512821;
longer blocks from .670270 to .536842. Buffered short IoU .5 matches fall from
3 to 2. No seed43/44 confirmation, full-match candidate or promotion follows.

## Protocol and provenance

The user authorized proceeding after the
[next-step review](NEXT_SCORE_STEP_REVIEW.md). The
[primary-source design review](DFL_TRANSFER_DESIGN_RESEARCH.md) preceded execution.
Phase A was frozen before downloading or parsing external XML; Phase B was
separately frozen after all Phase A gates and fresh XML replay passed, before
real shared-feature processing, any fitting or performance inspection.

| Freeze | SHA256 |
|---|---|
| [Phase A plan](DFL_TRANSFER_FEASIBILITY_PLAN.json) | `0ab248d00f1c9e4259a59d0191d46211bb0d12dfff7a8f645b9e4b9e743d533f` |
| [Phase B plan](DFL_TRANSFER_TEACHER_PLAN.json) | `304550b0380bfe68a985f76840f8656bc5f46196d4c0cd5a448c2308cb7dacb5` |

Source: DFL; Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of
spatiotemporal and event data in elite soccer*, Scientific Data12:195.
[Primary paper](https://www.nature.com/articles/s41597-025-04505-y),
[versioned dataset](https://doi.org/10.6084/m9.figshare.28196177.v1),
[CC BY4.0 licence](https://creativecommons.org/licenses/by/4.0/).
The verified Figshare catalog is preserved locally, SHA256
`ab353b23908ddad7cdfb3385a2eed050c69a5f7161d61799d8c67feeb43059f8`.
Fourteen information/position files, 2,622,686,298 bytes, match publisher size and
MD5 and have retained SHA256 identities. Event files and videos are unused.

Exact match IDs were assigned before parsing: train WMX/WN1/WOH/WOY;
development WPY; held out WQQ/WR9, all with prefix `DFL-MAT-J03`.
Both halves stay in their match. No match, split, target conversion, threshold,
architecture or decoder was changed after seeing results.

## Phase A: usable direct labels

Ingestion joins match identity, metadata home/away teams, player membership and
explicit `(half,N)` keys. Canonical grids include missing keys; unknowns remain
X and split spells. Status0 maps to D; status1 with owner1/2 maps to H/A;
other cases remain X. No interpolation, forward fill or label smoothing.

All seven matches pass every frozen support gate:

| Measured support | Result |
|---|---:|
| Canonical frames | 1,002,644 |
| Half lengths | 67,790–78,456 |
| Valid status, active owner, active player geometry, active ball XY, cadence | 100% in every match |
| H frames / spells | 293,931 / 1,539 |
| A frames / spells | 259,174 / 1,522 |
| D frames / spells | 449,539 / 713 |
| X frames | 0 |

Geometry requires at least four finite players per team. Cadence compares paired
observed timestamps with .04 times the difference in frame keys. Each team has
at least 2,500 live frames and10 spells per match; both halves exceed25,000 rows.
The exact gates and denominators remain in the Phase A plan.

The parser rejects conflicting retained keys, body match IDs, team mappings and
paired timestamps. Its audit scope excludes player rows outside the ball frame
domain; a player timestamp at a missing-ball row has no paired ball timestamp to
check. These are explicitly counted/scoped, not claimed as fully validated rows.
The code-only research reviewer found no fatal documented-schema mismatch.

Full label coverage establishes support, not annotation accuracy or identical
loose-ball/restart conventions. Cleaner DFL tracking and the challenge's noisy
automatic tracking remain different observation domains.

## Phase B: one external ownership model

The shared representation contains91 XY-only features: ball visibility and
lag5/25 displacement speeds; anonymous per-team counts, nearest three ball
distances, within1/3/5m counts and soft nearest-team share; bounded past/future
25/75 summaries. Physical metres, half resets and pitch-plus5m validity are
explicit. Native homography-invalid observations become missing; no equivalent
quality flag is supplied externally. Feature reach is at most100 frames.

Inputs exclude ownership/status, supplied speed/acceleration/height, identifiers,
clocks, absolute coordinates and tactical/event tags. Player IDs only place
source observations. The same builder serves both domains, with no fitted
preprocessing. Synthetic tests cover player permutation, team swap, translation,
rotation/reflection, half resets, missing support, bounded reach and exact prefix
parity.

One fixed corrupted view applies label-independent .2m XY noise,20% player-slot
dropout per50-frame block and a ball visibility Markov schedule with stationary
visibility .6. Corruption occurs before feature derivation. Clean and corrupted
training views are both used; no view or schedule is selected by performance.

The four training matches supply60,389 known live rows at stride5. Two views and
team-swap augmentation give241,556 fitting rows. One depth2/120-tree binary XGB
model uses equal match/class weights, seed42, native missing values and symmetric
prediction. No calibration, early stopping or parameter search.

The frozen gate requires masked balanced accuracy >=.70 **and** a gain >=.02 over
the same-view soft nearest-team baseline, separately in all three validation
matches. Missing baseline evidence gives .5. All three pass:

| Match / role | Clean teacher / nearest balanced accuracy | Masked teacher / nearest balanced accuracy | Masked teacher / nearest conditional spell F1 |
|---|---:|---:|---:|
| WPY / development | .877660 / .776073 | .786986 / .614719 | .237396 / .064313 |
| WQQ / held out | .899995 / .773915 | .801085 / .620668 | .268719 / .057090 |
| WR9 / held out | .891995 / .796266 | .825785 / .644989 | .290815 / .068434 |

Brier, log loss,10-bin calibration, class recall and short matches are preserved
per match/view in the ignored external summary. These are conditional H/A
metrics with reference D/X and half separators, not full H/A/D inference.
The external frame gate does not establish adequate short-possession matching.
The held-out results have now been inspected for this one acceptance decision;
they are no longer untouched evidence for future model selection.

## Public transfer: fails all three promotion conjunctions

Only after external gates and70 fresh replay artifacts passed did public fitting
begin. Two identical binary H/A heads use either the91 shared features or those
features plus the paired frozen teacher probability. Each uses only eligible
live public training targets, class balancing and symmetric augmentation and
prediction. The13 existing folds retain250-frame training exclusions. This fits
26 new public checkpoints; the external teacher never fits competition targets.

Both heads use the one frozen live-replacement decoder. Selected8 supplies the
D mask exactly; existing smooth5/switch3 ownership decoding resets at D and
evaluation boundaries. Its dynamic program uses observations throughout its
evaluation block; the100-frame bound describes feature construction, not a
claim of globally local offline decoding.

| Protocol | Selected8 spell F1 | Matched XY-only head | XY + teacher | Selected8 / new macro F1 | Selected8 / new short IoU .5 matches |
|---|---:|---:|---:|---:|---:|
| Buffered minutes | .741176 | .571429 | .571429 | .870833 / .844830 | 3 / 2 |
| Purged spells | .685714 | .512821 | .512821 | .871604 / .842988 | 3 / 2 |
| Buffered100 seconds | .670270 | .533333 | .536842 | .844383 / .818533 | 3 / 2 |

The teacher meets the matched-control primary check, with only a .003509 gain
in longer blocks and unchanged buffered/purged spell F1. It fails selected8
primary, macro, mean short IoU, previous short recovery and official
spurious/missed constraints in every protocol. Buffered substantial gain and
extra-short criteria also fail. D/false-D/stoppage checks pass because the D
mask stays exact across all six new sequences.

Official IoU .5 matched/spurious/missed counts, selected8 -> teacher:
buffered13/4/4 ->11/7/6; purged13/5/4 ->12/10/5;
longer13/7/4 ->12/9/5. All nine shared scores agree with the official scorer.
The71-frame miss and33-frame short possession have zero correct raw argmax
frames in both new heads across all protocols. New external supervision has not
resolved those ownership failures in this representation.

This tests one XY-only ownership head and one fixed teacher integration. Richer
broadcast/selected ownership features remain outside these new heads; selected8
still supplies D. The result cannot identify whether feature limitations,
observation/target domain differences, public adaptation or integration cause
the failure. It establishes no generic rejection of external supervision.
There is no retuning, second method or further round in this execution.

## Verification, execution history and preservation

- Phase A: both full XML parses reproduce17 artifacts byte-for-byte, including14
  parsed half caches; target conversion, canonical axes and every quality
  aggregate are recomputed. Publisher MD5/SHA256 identities verify all14 files.
- External:70 artifacts reproduce byte-for-byte in a fresh process:28 feature
  files,24 checkpoint probability arrays,17 fitting/provenance files and the
  metric summary. One checkpoint reload; no replay refit.
- Public: all26 checkpoints reload, all training rows/targets/weights reconstruct,
  all three selected references and six decoded sequences agree, and D remains
  exact. Both actual18935/19060 feature prefixes agree on public rows. A separate
  process reproduces51 feature/probability/provenance/prediction/score artifacts
  byte-for-byte, including9 scorer CSVs and all26 fitting records.
- All261 production tests pass. Source/input/cache identities,17 protected
  hashes and both original source symlinks verify. One external plus26 public
  models were fitted; no seed43/44 or full-public/candidate fit follows rejection.

Failures remain preserved. A synthetic timestamp test first exposed host-timezone
dependence; naive source timestamps now use an explicit reference timezone for
relative clock comparisons. The repair preceded XML parsing and kept labels,
splits and gates fixed. The initial slow transport was intentionally stopped
(exit143);574,619,648 partial bytes remain exact. The replacement verified126
disjoint HTTP ranges and complete publisher checksums. A duplicate-key error in
stage-metadata construction was repaired before external feature processing or
fitting, with unchanged protocol/source freeze. Final actual ingestion, training,
replay, scoring and verification dispatches all terminate with exit0.

The ledger appends exactly9 verified rows through `detector_ledger.append_score_rows`:
1,468 ->1,477 (778 inherited +699 continuation). Every preceding byte is exact.
The original SHA256 was
`c7c2b0383f799e3f192285b230254a28860f955b36511ddbeee6687c2b3cfaa1`;
the new ledger SHA256 is
`b7d629707c163df277c6b535004ba7150a603c3e96d2af1d6891b61087c34091`.
These remain dependent, adaptively reused within-match evaluations.

Selected `data/candidate_motion_dead_trim/candidate.csv` remains199,319 rows,
SHA256 `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
No prior candidate, model, failure or source artifact is removed. Only provided
public competition targets are used; no corrected/private targets, manual labels,
reserved/held sample inspection, competition image processing, uploads, commits,
submissions, publication, contacts, installations or toolchain changes occur.

Evidence is under `data/control_dfl_transfer_v1/`: Phase A/B freezes and source
archives, raw checksum manifest, paired XML assessments, external/public trials,
fresh replays, execution repairs/logs and integrity/closeout records. Entry points
are [ingestion](prepare_dfl_transfer_data.py),
[external trial](research_dfl_external.py), [public trial](research_dfl_public.py)
and [verification](finalize_dfl_transfer.py). Stage runners use exclusive output
directories to retain evidence. The full frozen chain, including ignored audit
scripts, is preserved locally; reproduction requires fresh outputs and the
matching pinned inputs/environment.

Round27 is complete and rejected for possession-score promotion. Data feasibility
and external prediction gains are measured; private/full-match improvement and
maximum score remain unverified. Existing promotion requirements remain intact.
