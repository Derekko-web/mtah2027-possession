# Round 28: frozen DFL teacher signal and pipeline capacity — opportunity rejected

Completed 2026-10-23 (America/Toronto). **The diagnostic fails its prospective
continuation gate. No integration is trained, no score row is added and selected8
remains unchanged.** The teacher has good average native ownership accuracy but
provides little reliable correction evidence on the selected model's errors.

The [advisory recommendation](POST_DFL_NEXT_STEP_REVIEW.md) and
[code-only primary-source design review](DFL_SIGNAL_DESIGN_REVIEW.md) preceded
execution. The user authorized the diagnostic and conditional integration.
[DFL_SIGNAL_PLAN.json](DFL_SIGNAL_PLAN.json), SHA256
`40584d1eefb22d3b4d6b8e4d9296bab2de7af08651637a7c0d8e7790be577790`,
was frozen before target-conditioned inspection. Its criteria, teacher,
probabilities, model roles and oracle recipe remain unchanged after results.

## What was measured

The audit reloads the immutable external teacher and verifies its cached native
XY features/probabilities. It covers all17 full-public ground-truth possessions,
all six short possessions, all matched controls, all four IoU .5 unmatched GT
cases and every spurious reference prediction in each protocol (4/5/7).
These are repeated same-match diagnostics, not independent examples.

Raw native balanced accuracy uses all6,031 true-live public frames. Teacher
decisions use pH>=.5 for H. Confidence disagreements use true-live AND
selected-live rows, teacher confidence>=.65 and a different teacher/reference
class. C counts teacher-correct/reference-wrong frames; H counts
teacher-wrong/reference-correct frames. Precision is C/(C+H), undefined if empty.
These are hypothetical frame decisions, not actual corrections deployed.

The separate fixed teacher diagnostic copies selected D/X and applies existing
smooth half-window5 (11 frames) and DP switch penalty3 within each selected-live
and evaluation segment. Matching uses the official gap-aware, abstention-filled,
same-team greedy IoU>=.5 policy on the assembled public sequence. Whole GT case
IDs and lengths stay fixed across protocols; evaluation-block clipping cannot
create new short cases. No new possession F1 or candidate CSV is produced.

The continuation gate requires, in every protocol:

- native balanced accuracy>=.70;
- at least25 confident potentially corrected frames and correction precision>=.55;
- one **common** originally unmatched GT case, at most125 frames, matched by the
  fixed teacher diagnostic with mean raw correct-team probability>=.60;
- that same case recoverable through one declared model slot under the fixed
  oracle, retaining every previous GT match and adding no spurious prediction.

Slot4 has prospective priority over slot0 only if its complete conjunction passes
all protocols. No mixing different cases or slots between protocols is allowed.
Passing would only justify a later separately frozen append-only fit. This is a
sufficient opportunity screen, not a necessary condition for all possible uses
of this teacher.

## Teacher signal: average accuracy does not supply useful error corrections

Raw native balanced accuracy is **.854971**: H recall .905906, A recall .804036.
Those values are identical across protocols because the teacher and true-live
rows are unchanged. The native accuracy condition passes, but both correction
requirements fail everywhere:

| Protocol | True-live and reference-live rows | Confident disagreements | C / H | Correction precision | Required |
|---|---:|---:|---:|---:|---|
| Buffered minutes | 5,943 | 160 | 14 /146 | .087500 | C>=25 and precision>=.55 |
| Purged spells | 5,889 | 170 | 24 /146 | .141176 | C>=25 and precision>=.55 |
| Buffered100 seconds | 6,015 | 120 | 24 /96 | .200000 | C>=25 and precision>=.55 |

The fixed teacher diagnostic retains11/11/10 of the13 previously matched GT
possessions, adds zero matches among the four unmatched GT cases in every
protocol, and matches2 of6 short possessions versus3 for selected8.

All four unmatched GT cases are included, not just familiar failures:

| GT length | Mean raw correct-team probability | Correct raw argmax fraction | Teacher best same-team IoU |
|---|---:|---:|---:|
| 202 frames | .409068 | .356436 | .185567 |
| 71 frames | .347892 | 0 | 0 |
| 19 frames | .336745 | 0 | 0 |
| 33 frames | .252230 | 0 | 0 |

These raw summaries are protocol-invariant. In particular, the raw teacher
already chooses the wrong team throughout all three unmatched short cases,
before any public adapter consumes it. This supports an absent-correct-decision
diagnosis for these cases; it does not prove the probabilities contain no
information a different learner could exploit. The19-frame case is unmatched
under IoU .5 despite selected8 recalling its true frames with an overlong span;
unmatched does not mean zero selected frame recall.

## Pipeline capacity: both actual branches traced, no clean common route

There are56 target-aware oracle probes per process: two slots times12 missed-GT
protocol cases plus16 spurious-prediction protocol cases. They are diagnostic
label injections, **not model predictions or achievable score estimates**.

For one case at a time, only its true H/A rows change. Original slot D and H+A
mass stay fixed; conditional ownership allocates99/1 toward the true team. Other
rows and all other slots stay pristine. The unchanged full selected8 pipeline
runs, followed by a fixed overlay accepting alternate H/A only where both outputs
are live. Selected D/X stay exact. This constrained oracle does not test every
possible joint change to ownership, dead-ball probability or multiple intervals.

Every probe regenerates both actual closure contexts: the seven-model reference
and the native-motion alternative. Traced branch assembly agrees with the
original shared decoder. Gate reasons are first evaluated rejections, not unique
causes or evidence that all later predicates were tested. Global official
matching is recomputed after each probe; new matches cannot hide lost matches
or extra spurious predictions through greedy reassignment.

| Model role | Missed-case oracle observations |
|---|---|
| Broadcast-joint slot4 | No label changes for any of the four missed GT cases in any protocol; no additional GT match. |
| Ordered current slot0 | The71-frame case becomes matched at IoU .835294/.835294/.855422, but each probe adds one spurious possession. It fails the collateral constraint. Other missed cases remain unmatched. |

This distinguishes **a reachable ownership change with collateral error** from
a completely blocked route. It is incorrect to claim slot0 cannot express the
71-frame possession. The19- and33-frame extent problems persist under the
declared probes. Neither role supplies a common clean recoverable short case,
and the teacher diagnostic supplies none of the missing short matches anyway.

All4/5/7 spurious cases were probed, including those with no wrong true-live rows.
Only2/2/4 spurious cases contain wrong true-live frames (113/114/183 such frames
respectively). Ownership-only probes leave true D/X rows untouched. Slot4 never
reduces total spurious count in these probes. Slot0 reduces it for one longer-
block probe, without yielding a whole-conjunction pass. Detailed per-case
collateral matches, prediction IDs, branch hashes and ordered gate traces remain
local under ignored data.

## Execution repair and correction to Round27 prefix evidence

The initial run stopped **before reading the public target CSV or performing
target diagnostics**: a new guard incorrectly assumed the decoder cache and
native observation cache had the same length. The actual lengths are18,935 for
the decoder/fitting cache and19,060 for native observations.

The failed source/freeze, partial output and log are retained in
`data/control_dfl_signal_v1/execution_failure_01/`. A regression reproduces the
wrong length assumption (red exit1) and passes after separating decoder, short
and long source lengths (green14 tests). Execution revision1 changes only the
new diagnostic helper, test and runner; the frozen plan and numerical criteria
are byte-identical. No target-conditioned result preceded the repair.

This exposes a historical evidence error: Round27's native parity implementation
used `len(raw)==18935` for its full feature build and also used18935 for the
short build. Its claimed19060-vs18935 native check was therefore unverified.
The old report, evidence and scores remain preserved; this paragraph corrects
that claim explicitly.

Round28 now actually reconstructs features from both **19,060 and18,935 native
source rows**, independently predicts with the frozen teacher, and verifies exact
public feature/probability equality. The18,935-row reconstruction also reproduces
the complete frozen Round27 native feature table and teacher probabilities.
The corrected check passes; no old feature, model, score, candidate or ledger
byte changes. This is a verification repair, not a measured score gain.

## Verification and decision

- All twelve identity controls per process reproduce selected8's entire padded
  validation-prefix H/A/D/X sequence: unchanged and self-conditional transfers
  in slot0 and slot4, across all three protocols. Selected candidate-file
  identity is checked separately; no new full-match candidate is generated.
- Both independent processes repeat actual feature-prefix, teacher reload,
  identity, complete case, oracle and branch-trace diagnostics. All13 output
  artifacts, including native short/long feature and probability caches, replay
  byte-for-byte. All59 baseline/oracle probe groups have both closure branches.
- All56 oracle probes per process preserve D/X and pristine parent arrays;
  full traces reproduce the original decoder. Matching/denominators/common-case
  gates are checked independently by the verification entry point.
- All275 production tests pass, including14 diagnostic regressions covering
  official matching, gaps/abstentions, oracle scope/mass, undefined precision,
  common IDs, role priority, collateral matching and distinct prefix lengths.
- Source/input/cache identities,17 protected hashes and both symlinks verify.
  Every1,477 ledger row and byte stays exact; SHA256 remains
  `b7d629707c163df277c6b535004ba7150a603c3e96d2af1d6891b61087c34091`.

Selected8 remains `data/candidate_motion_dead_trim/candidate.csv`, SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
Its retained public spell F1 remains .741176/.685714/.670270. No new research fit,
possession-score row, candidate CSV, confirmation or promotion occurs. The
conditional integration is stopped at the declared gate; no thresholds are
relaxed and no replacement model is tried.

Evidence and immutable source snapshots are under `data/control_dfl_signal_v1/`.
Successful outputs are `analysis_A_02/` and `analysis_replay/`; the original
`analysis_A/` and failed log remain untouched. Entry points are
[the audit runner](research_dfl_signal.py) and [verification](audit_dfl_signal.py).
Stage outputs are exclusive to preserve prior evidence.

No private/corrected targets, manual annotations, reserved/held sample inspection,
new media, external validation reuse, uploads, commits, submissions, publication,
contacts, installations or toolchain changes occur. Public error diagnostics
remain adaptive and cannot establish private accuracy or a maximum score.
Round28 is complete; this specific teacher integration opportunity is rejected.
