# DFL native-signal diagnostic: design review

Prepared 2026-10-23. Source/aggregate-report review only. No data files,
observations, arrays, caches, media or target rows were opened; no tests,
processing, fits or scores ran. This advisory note freezes no protocol and
changes no existing file. The research skill is applied by the already-dispatched
background reviewer.

## Assessment

The proposed conjunction is a reasonable **sufficient opportunity gate for a
later investigation**, with the qualifications below. It does not establish a
score improvement, validate an oracle as a model or prove that the teacher is
useless if it fails. Round27 evaluated newly fitted91/93-column heads, not a
standalone native-teacher error-and-route audit.
[research_dfl_public.py:61-64,84-99](research_dfl_public.py#L61-L64),
[DFL_TRANSFER_REPORT.md:148-157](DFL_TRANSFER_REPORT.md#L148-L157).

Retain the proposed numerical requirements without tuning: exact selected8
identity in all three protocols; native conditional live balanced accuracy
>=.70; high-confidence disagreement precision >=.55 with >=25 potentially
corrected frames; a previously unmatched short ground-truth possession <=125
frames matched at official IoU>=.5 by the fixed teacher diagnostic, with mean
correct-team probability >=.60; and that same case recoverable by the declared
99% target-aware intervention in a particular existing slot, preserving all
previously matched GT cases and adding no official spurious predictions. One
common full-public GT case must satisfy the whole case conjunction in all three
protocols. Prefer slot4 only
if its entire conjunction passes all protocols, otherwise slot0 if it passes
all, otherwise stop without fitting. These choices are prospective design
decisions, not empirical calibration of the cutoffs.

## Denominators and fixed teacher diagnostic

Native balanced accuracy should use **all public true H/A rows**, not only rows
where selected8 already predicts live or where the teacher is confident. Require
both true classes and finite probabilities; an empty class fails as undefined.
The teacher's existing binary convention assigns exact .5 ties to H. Freeze raw
versus smoothed probability use explicitly; raw native probabilities are the
natural choice for this gate.
[dfl_transfer_teacher.py:29-33,40-53](dfl_transfer_teacher.py#L40-L53).

For disagreement precision, restrict to true H/A AND reference H/A, raw teacher
confidence `max(pH,pA)>=.65`, and different teacher/reference decisions. Binary
disagreements then partition into potentially correct changes C and harmed
previously correct frames H. Use `C/(C+H)` with zero denominator failing, and
report C, H, excluded rows and class support. C is a hypothetical frame count,
not demonstrated corrected output. Do not add already-agreeing correct frames
to its numerator. This frame gate can still favor one long interval while
damaging several brief possessions; that is why the possession and route checks
and collateral diagnostics remain necessary.

Fix the mean-correct-team-probability denominator to the entire ground-truth case
span, with raw/smoothed convention declared, rather than selecting correct or
high-confidence frames after seeing outcomes. Missing or undefined groups fail.

The diagnostic decoder should copy selected8 and decode teacher H/A only inside
each contiguous selected8-live segment, smoothing with existing half-window5
(11 frames) and switch penalty3. Reset smoothing and DP at reference D, X, half
and evaluation-block boundaries. Preserve selected8 D/X exactly. Existing
`ownership_prediction` accepts only H/A/D codes, so X separators need explicit
handling rather than passing them into that function. Its DP is global within
each supplied segment; do not claim a local prediction dependency bound.
[weak_ownership.py:73-85,101-102](weak_ownership.py#L73-L85),
[control_refinement.py:66-67](control_refinement.py#L66-L67),
[transition_penalty.py:12-47](transition_penalty.py#L12-L47).

Whole-public native accuracy may be identical across protocols because neither
teacher probabilities nor true-live rows change. Repeating that gate three
times is not independent evidence. Reference disagreements and segment decoding
can differ with the protocol's selected8 sequence and block boundaries.

## Official matching and case identity

Construct the complete missed-ground-truth, spurious-prediction, short and
already-correct control universes from the original selected8 sequence, with
stable global case IDs. Use the full graded-public ground-truth case length for
the <=125 rule; do not manufacture short cases by clipping at evaluation-block
edges. Decode blocks independently, then assemble the public prediction before
the official stitched comparison. Keep block-clipped diagnostics separately.

The official matcher uses same-team overlaps, descending IoU, then ascending GT
and prediction indices, with greedy one-to-one consumption and inclusive >=.5.
A case's best overlap reaching .5 is insufficient if that prediction is consumed
by another GT case. Recompute the complete matching after every teacher/oracle
intervention. Preserve the official graded-run and predicted-X filling policy;
do not substitute Hungarian matching or match each case independently.
[possession_metrics.py:259-307,464-514,654-671](../evaluation/possession_metrics.py#L464-L514).

The executing agent settled the quantifier before target-conditioned diagnostics:
**one common full-public GT ID must satisfy the entire case conjunction in all
three protocols for one slot**. The teacher and oracle qualify that identical
GT ID, and every protocol-specific condition must pass. Different successful
cases across protocols do not suffice; neither does combining slot4 in one
protocol with slot0 in another.

Greedy reallocation can newly match a case while losing another. The declared
oracle gate therefore also retains **every previously matched GT** and allows
no increase in official spurious predictions. Recompute those global flags/counts
for each intervention. Report matched prediction identities, lost matches and
collateral changed/spurious/correct possessions. A successful oracle is still
not a learned-model gain claim. Report all baseline
missed/spurious cases and controls, including undefined or unsuccessful cases;
the known71/33-frame failures must not determine the case universe.

## Oracle and route-trace hazards

The declared oracle is ownership-only: preserve the slot's original D mass and
original H+A mass, allocating the latter .99/.01 toward the case's true team on
its true-live rows. Every other row and slot stays pristine. Handle zero live
mass explicitly without division by zero; it supplies no new live evidence.
Do not use target=.99/others=.005 vectors, which would also alter intermediate D.

Each oracle run starts from pristine selected arrays, changes only the chosen
slot and that case's half-open GT span, and executes the unchanged full pipeline
through every evaluation block. Never accumulate oracle interventions across
cases. The final fixed overlay may take alternate H/A only where both selected8
and the alternate are live, retaining selected8 otherwise. Its identity control
must reproduce the full H/A/D/X sequence exactly, not merely D or the score.
[research_detector_slot.py:53-72](research_detector_slot.py#L53-L72),
[motion_dead_trim.py:9-16](motion_dead_trim.py#L9-L16).

Slot4 affects learned agreement only inside observed, physically generated
broadcast proposals; physical confidence, contacts, reference protection,
duration and flank gates remain. Slot0 can alter the underlying consensus and
therefore change downstream reference-dependent gates. Static coverage masks
cannot prove either route is blocked or usable.
[broadcast_consensus.py:38-46](broadcast_consensus.py#L38-L46),
[broadcast_intersection.py:15-44](broadcast_intersection.py#L15-L44),
[stoppage_consensus.py:27-54](stoppage_consensus.py#L27-L54).

Trace both actual broadcast-closure contexts: selected8's reference and the
motion alternative. The latter can use a different current-head replacement.
Regenerate traces for each oracle, and connect any recovery to the output's
actual branch. A single baseline closure trace is incomplete.
[motion_dead_trim.py:9-14](motion_dead_trim.py#L9-L14),
[pair_candidate.py:14-19](pair_candidate.py#L14-L19),
[broadcast_refinement.py:11-18](broadcast_refinement.py#L11-L18).
The trace's reason is the **first evaluated failure**, not a unique physical
cause; unavailable later predicates must not be reported as tested failures.
[broadcast_intersection.py:31-45](broadcast_intersection.py#L31-L45).

## Scope and interpretation

Use lower-level official spell construction/overlap/matching helpers to produce
counts, flags and case identities. Do not run the broad evaluator merely to
discard its F1/bootstraps afterward. No candidate CSV, new possession F1 score,
model fit or ledger row is required. Preserve selected8, the teacher, all17
protected artifacts and every current ledger byte; retain source/cache/prefix
and fresh-process replay evidence in a new ignored diagnostic directory.
[possession_metrics.py:517-539,644-671](../evaluation/possession_metrics.py#L644-L671).

Passing this conjunction would justify freezing one later append-only fitting
hypothesis in the selected slot. That later integration would transfer only the
new head's conditional H/A into the original slot's live mass, retaining original
slot D, just as this diagnostic does. Its conditional identity control must
reproduce selected8's complete labels; preserve original identity rows exactly
to avoid avoidable renormalization rounding. It would not demonstrate that the new head can
learn the oracle intervention, preserve all correct spells or pass the original
substantial-gain/extra-short-match and confirmation gates. Failure closes this
specific sufficient test without a conclusion about every possible teacher,
representation or model. No actual diagnostic result is claimed here.
[DFL_TRANSFER_REPORT.md:138-157,211-213](DFL_TRANSFER_REPORT.md#L138-L157).
