# WASB native opportunity: execution repaired, signal rejected

Completed 28 October 2026. The unchanged native optical opportunity fails its
frozen continuation gate. No possession-model fitting, official possession-score
evaluation, ledger row, candidate or promotion follows. Selected8 remains exact.
See the [native plan](WASB_NATIVE_PLAN.json), the original
[detector report](WASB_REPORT.md), and the added execution wrapper
[wasb_native_repair.py](wasb_native_repair.py).

## Preserved execution repair

The original preparation compared a raw prefix ball array against a cached ball
array whose producer already applied pitch/validity cleanup. Both actual prefixes
(18,935 and 19,060 rows) disagree with the cache on exactly 26 public rows; every
disagreement is outside the producer's existing pitch-plus-five-metre bounds.
All jointly finite ball values, player positions and team metadata agree exactly.
Applying the producer's existing cleanup makes both prefix witnesses exact.

The separate wrapper changes only that preparation witness. It does not modify
the cached tracked ball, original source file, failed preparation, frame schedule,
image processing, detector, confidence, component selection, contact readout or
gates. The old failure and freeze remain under `data/control_wasb_native_v1/`;
the repaired run has its own folder and source/failure hash binding.

Physical preparation passes with 1,499 of 1,500 uniform centers eligible and
99.92% selected-live coverage in every protocol. All three independently decoded
original-timestamp RGB witnesses are byte-identical. No images are displayed or
sent to a remote service.

## Verified detector execution

The fixed adjacent-frame and repeated-center arms complete on MPS in 287.22
seconds, with CPU peak 0.289 GiB and MPS driver allocation 1.169 GiB. The frozen
900-second, 2.5-GiB CPU and 2-GiB MPS limits pass. All 32 fresh-model reload
witnesses match exactly. Inference and contact extraction finish before the
public target join.

| Label-free coverage over all 1,500 centers | Temporal | Repeated center |
|---|---:|---:|
| Physically eligible | 1,499 | 1,499 |
| Finite detected pixel points | 209 | 57 |
| Finite projected points | 209 | 57 |
| Within the frozen pitch bounds | 12 | 4 |
| Positive contact denominator | 12 | 4 |
| Strong contact | 4 | 1 |

Most finite optical points therefore fail the existing pitch-bound check. The
external localization results do not establish useful native contact evidence.
This is a measured support failure, not evidence that another threshold or
component/ROI rule would repair it; no such variant is run.

## Frozen signal audit

Only the temporal arm is eligible. In every protocol its observed editable
true-live sample has 12 supported centers and balanced accuracy .20. Qualified
decisions correct zero reference centers and damage four, for precision zero and
net minus four. No previously missed short possession gains an IoU .5 match.

| Protocol | Editable true-live centers | Temporal supported centers | Corrections / damage | New short matches | Diagnostic spurious before / after |
|---|---:|---:|---:|---:|---:|
| Buffered minutes | 1,190 | 12 | 0 / 4 | 0 | 4 / 4 |
| Purged spells | 1,179 | 12 | 0 / 4 | 0 | 5 / 5 |
| Buffered 100 seconds | 1,204 | 12 | 0 / 4 | 0 | 7 / 7 |

All old IoU .5 GT matches are retained, but balanced accuracy, correction count,
precision, net corrections and the common-new-short condition fail. Both the
conditional observed metric and missing-counted-wrong accuracy are retained;
the latter is about .0033 across protocols. Every truth, short, missed and
spurious case remains in the saved audit.

The repeated-center control has four supported editable true-live centers and
zero corrections versus one damaged center. Its balanced accuracy is undefined
because one class lacks support, and that condition fails rather than being
omitted. The provided-tracked-ball control has 1,112/1,101/1,126 supported centers
but only 5/7/5 corrections versus 41/41/37 damage; its spurious counts increase
to 14/15/13. Neither control is eligible for selection or rescues the temporal
failure.

## Replay and preservation

A fresh process recomputes contacts and the complete audit from the immutable
input/detection caches. All 16 contact/audit artifacts, including all nine
diagnostic sequences, summary and manifests, replay byte-identically. Every
frozen source/input/output/contact/audit hash, all 17 selected artifact hashes,
the earlier ledger prefix and source links verify. No existing Python source is
modified by this continuation. No new possession-score rows are appended.

Ignored evidence is under `data/control_wasb_native_repair_v1/`, including
`preparation_diagnosis.json`, `resource_checks.json`, `contact_coverage.json`,
`signal_audit/summary.json`, `fresh_replay_verification.json` and `closeout.json`.
The independent contact/audit recomputation is under
`data/control_wasb_native_repair_replay_v1/`.

This one fixed native detector/readout avenue is rejected. The earlier pilot
remains failed for insufficient support; the disjoint external confirmation
remains passed on its own terms. Neither establishes native possession value,
private full-match accuracy or a maximum achievable score. The broader research
goal remains unfinished.
