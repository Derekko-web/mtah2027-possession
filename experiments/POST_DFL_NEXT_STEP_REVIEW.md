# After DFL transfer: advisory next-step review

Prepared 2026-10-23. This is a source/report review, **not a frozen experiment or
authorization to execute another round**. No data files, saved observations,
arrays, media or target rows were opened. No fits, tests, scores, downloads or
processing ran. Only this new note was written. The research skill is applied
by the already-dispatched background reviewer.

## Recommendation

**The strongest immediate next action is one bounded, no-fit audit of the frozen
teacher's native signal and the selected pipeline's gates.** Do that before
another supervised fit. Round 27 establishes that a 91-feature ownership head
with teacher input fails; it does not isolate the cause or establish whether
the immutable teacher itself supplies useful signal on competition errors.
[DFL_TRANSFER_REPORT.md:117-157](DFL_TRANSFER_REPORT.md#L117-L157).

The producer computes the teacher probability, adds it to shared XY inputs,
then fits two new binary heads and replaces live ownership. It evaluates those
heads, rather than a native-teacher diagnostic. Both new heads miss the known
brief controls in raw argmax, but that finding does not answer whether the
teacher signal was absent or discarded by public adaptation.
[research_dfl_public.py:61-64,84-99](research_dfl_public.py#L61-L64),
[DFL_TRANSFER_REPORT.md:148-156](DFL_TRANSFER_REPORT.md#L148-L156).

Nor is the loss against selected8 evidence that dropping rich features alone
caused failure: the comparison also changes the learner, binary versus joint
training and ownership integration. The XY-only control correctly isolates
adding teacher inputs **within that weaker pipeline**; exact D does not make
its H/A behavior an identity control for selected8.
[research_dfl_public.py:63,89-99](research_dfl_public.py#L89-L99).

## One diagnostic scope, frozen before looking at targets

A separately authorized protocol should hash and reuse the teacher and existing
probabilities, features and selected references. Freeze diagnostics and a
numeric continuation gate before examining new target-conditioned results.
Do not fit a teacher again, change corruption or use this audit to tune a
threshold. The external held-out matches have already been inspected; they
cannot become untouched validation for another selected model.
[DFL_TRANSFER_REPORT.md:100-115](DFL_TRANSFER_REPORT.md#L100-L115).

Cover **every public missed and spurious possession**, all short live possessions
and the complete public live interval under the existing protocols and official
matching rules. Do not select only the familiar 71-/33-frame failures. With fixed
existing class decisions and probability summaries, distinguish:

- Teacher signal absent or wrong before public fitting.
- Useful teacher signal lost by the newly fitted Round 27 head.
- Useful signal outside broadcast observation/proposal eligibility.
- A potential proposal already present but vetoed by learned agreement or another
  unchanged gate.

Include denominators, ambiguity/undefined groups, already-correct controls and
potential damage to existing recoveries. Any target-conditioned opportunity or
oracle count is diagnostic, not an achievable score estimate or inference rule.
Keep output aggregate, with detailed evidence local and ignored. No production
prediction changes or possession-ledger rows follow from this diagnostic alone.

The continuation gate must require useful incremental signal across the fixed
error universe and an identified route through a specific model role, while
accounting for false opportunities on correct reference predictions. If that
conjunction fails or is undefined, stop this teacher avenue. The exact numeric
criteria need a prospective diagnostic protocol; this review does not invent
them from unseen results.

## Why raw102 and joint170 are different downstream choices

| Append-only choice | What it tests | Structural limitation |
|---|---|---|
| Original ordered102 current head, array slot0 | Whether the frozen teacher helps the current head inside the full existing consensus, rescue, broadcast closure and motion pipeline | Broad influence on reference H/A and D through consensus; more interactions and risk. It is not equivalent to binary live replacement. |
| Existing ordered102 + broadcast68 joint head, array slot4 | Whether the teacher helps the existing broadcast model's role without discarding its rich inputs | Broadcast use requires observed ball and physical source proposals. A better learned head cannot create a proposal that this source never generates. |

Slot identities and assembly are explicit in
[research_pair_motion.py:25-33](research_pair_motion.py#L25-L33) and
[research_ownership_graph.py:24-30](research_ownership_graph.py#L24-L30).
The joint view is `ordered.join(combined)` and was trained with the frame-objective
depth3/160-tree fitter, unlike Round 27's new binary heads.
[research_broadcast_ownership.py:45-49,103-109](research_broadcast_ownership.py#L45-L49),
[research_control_renewal.py:32-40](research_control_renewal.py#L32-L40).

Slot4 reaches learned broadcast inference only where `bc_ball_seen` is true.
Closure proposals come from observed source shares, contact, duration, confidence,
reference protection and flank rules; the learned head contributes an agreement
check inside those proposals. Therefore the proposal map is a necessary part of
the diagnostic, not a reason to assume richer augmentation will repair a miss.
[broadcast_consensus.py:38-46](broadcast_consensus.py#L38-L46),
[broadcast_refinement.py:11-20](broadcast_refinement.py#L11-L20),
[broadcast_intersection.py:15-44](broadcast_intersection.py#L15-L44).

If fitting later becomes justified, choose **one** slot prospectively from the
diagnostic and append only the paired frozen teacher probability to its original
inputs. Keep its original model settings, targets, weights, columns, fold
exclusions, seed and all other selected arrays unchanged. Do not add the 91 XY
features simultaneously or try both slots as an undeclared screen. This is a
conditional downstream choice, not a committed fitting plan.

## The control must reproduce all selected8 labels

An unchanged checkpoint placed in its original slot must reproduce selected8's
complete H/A/D/X sequence, including every existing recovered possession, on all
protocols. Matching only D or headline scores is insufficient. Verify feature
schema, checkpoint/raw probability identity and the original fitting recipe;
cache reuse is valid only with that provenance. Existing detector-slot research
already demonstrates the appropriate identity-control pattern.
[research_detector_slot.py:53-72](research_detector_slot.py#L53-L72).

For slot4, replace only `arrays[4]` and run the same selected8 decoder. The
existing joint-slot implementation does this; broadcast closure preserves the
six-model D mask and motion trimming is applied through the same path.
[research_broadcast_action.py:92-115](research_broadcast_action.py#L92-L115),
[motion_dead_trim.py:9-16](motion_dead_trim.py#L9-L16).
Keep an exact selected8 D/X assertion. If slot0 is chosen and the alternate path
changes D, a separately frozen live-only overlay can accept alternate H/A solely
where selected8 and the alternate are both live, retaining selected8 otherwise.
Apply the identical rule to the identity control; do not use the blanket
`live_replace` decoder as a substitute for pipeline identity.
[stoppage_consensus.py:27-54](stoppage_consensus.py#L27-L54),
[provided_speed.py:114-126](provided_speed.py#L114-L126).

This distinction is already material in the history: contact heads extended
the same170 view, yet their live-replacement control regressed while every
joint-slot variant remained label-identical to selected8. Detector slot changes
also left labels unchanged despite changed probabilities. That supports careful
role preservation and warns that slot4 may simply have no actionable route.
[CONTACT_HEADS_REPORT.md:9-43](CONTACT_HEADS_REPORT.md#L9-L43),
[DETECTOR_SLOT_REPORT.md:14-24,33-44](DETECTOR_SLOT_REPORT.md#L33-L44).

## Alternatives and limits

A new external temporal learner could address fragmented external ownership
sequences, but it changes another hypothesis before establishing native transfer
signal. High external balanced accuracy and conditional spell F1 .24-.29 are
different achievements; neither proves useful competition possession intervals.
Another generic endpoint adjustment also cannot create a wholly missing
possession. These are reasons to prefer the bounded diagnostic first, not proof
that a richer model cannot help.
[DFL_TRANSFER_REPORT.md:104-115](DFL_TRANSFER_REPORT.md#L104-L115),
[SHORT_PRECISION_REPORT.md:62-69](SHORT_PRECISION_REPORT.md#L62-L69).

Preserve the teacher, all failures, selected8, the17 protected artifacts and every
current ledger byte. Any later candidate still needs the original substantial
gain, extra short match, secondary protections, confirmation seeds, official
replay and independent review. An append-only result could show incremental
utility in that pipeline; it would not prove why Round27 failed or establish
private/full-match accuracy. No next round starts with this advisory note.
[DFL_TRANSFER_REPORT.md:186-213](DFL_TRANSFER_REPORT.md#L186-L213),
[PAIR_MOTION_REPORT.md:147-163](PAIR_MOTION_REPORT.md#L147-L163).
