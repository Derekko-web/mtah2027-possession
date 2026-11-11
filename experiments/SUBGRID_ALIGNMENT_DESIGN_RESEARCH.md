# Sub-grid formation refinement: source-based design review

Prepared 2026-10-23 for the proposed Round 25 metadata investigation. This review
read local source, plans, tests and aggregate reports only. It did not inspect
competition arrays, media, XML, targets or per-frame extracts, and performed no
observation processing, inference, fitting or scoring. It applies the research
skill as the already-dispatched background reviewer. The separately frozen
execution plan is authoritative and must freeze the definitions below before
observations.

## Recommendation

Test one bounded change: independently refine each original global-grid winner
inside its integer half-cell before applying the strict neighborhood checks.
Preserve the original reflection and directed geometry. Use the resulting
candidate only to fill missing original strict mapping rows, preserving all old
mapping fields exactly. Report the candidate alone and this extension separately.
The prior report identifies 18 own-geometry-passing, neighbor-cost-vetoed anchors;
it explicitly does not establish that any can be recovered correctly.
[ANCHOR_FAILURE_REPORT.md:101–106](ANCHOR_FAILURE_REPORT.md#L101-L106),
[ANCHOR_FAILURE_REPORT.md:197–205](ANCHOR_FAILURE_REPORT.md#L197-L205).

This changes the search procedure even when numeric thresholds stay unchanged.
Improved minimum geometry is expected from searching more candidates and cannot
establish better synchronization. Freeze opportunity, availability and unused
checkpoint gates before computing new results, with no fallback variant or
threshold adjustment if the primary fails. The earlier metadata protocol makes
the same distinction between engineering opportunity and possession accuracy.
[broadcast_alignment.py:103–110](broadcast_alignment.py#L103-L110),
[ALIGNMENT_COVERAGE_PLAN.json:41–53](ALIGNMENT_COVERAGE_PLAN.json#L41-L53).

## Preserve and reconstruct the original baseline first

Recreate the unchanged original anchors using both original profiles: the fine
prefilter uses the joint maximum cost 3.5 and minimum margin .25, followed by the
strict 2.5/.5/span-10 mask. Retain exact coarse/fine arrays, NaNs, positive-infinite
margins, allowed masks, mapping fields, rounding and inclusive overwrite order.
Do not replace a permissively computed original fine table with a strict-only
reconstruction. This baseline is distinct from the new refined candidate.
[broadcast_alignment.py:112–161](broadcast_alignment.py#L112-L161),
[ANCHOR_FAILURE_REPORT.md:123–137](ANCHOR_FAILURE_REPORT.md#L123-L137),
[test_anchor_failures.py:30–47](test_anchor_failures.py#L30-L47).

The source usability rule remains supplied homography validity and at least four
finite player observations per team. The geometry is broadcast-to-tactical
same-team nearest-neighbor distance, capped at 10 metres before squaring, averaged
within each team and equally across teams, then square-rooted. Slot identity is
not used. Do not substitute reciprocal error, balanced RMS or one-to-one matching
while reusing 2.5: those statistics have different visibility behavior, documented
in the existing diagnostic report.
[broadcast_alignment.py:65–89](broadcast_alignment.py#L65-L89),
[test_broadcast_alignment.py:45–53](test_broadcast_alignment.py#L45-L53),
[ANCHOR_FAILURE_REPORT.md:108–113](ANCHOR_FAILURE_REPORT.md#L108-L113).

## Winner and far-runner refinement

The original tactical and broadcast anchor lattices are zero-based multiples of
25. Coarse search stacks all four reflections and takes the first flattened
minimum. Its far set is **strictly more than 250 source frames from the original
winning broadcast frame**, across all four reflections. Preserve that exact mask
and tie order when extracting the original runner identity and reflection. A saved
scalar margin does not identify the runner. If no far candidates exist, keep an
explicit absent-runner sentinel and positive-infinite margin, rather than
mistaking infinity for an unusable neighborhood.
[broadcast_alignment.py:96–110](broadcast_alignment.py#L96-L110),
[anchor_failures.py:27–40](anchor_failures.py#L27-L40),
[test_anchor_failures.py:40–47](test_anchor_failures.py#L40-L47).

For each original finite winner, search exactly the integer source frames
`old_winner + [-12, ..., +12]` at its original reflection, using the same
single-tactical-frame geometry and source usability. Choose the earliest source
frame on a tie. This is a half-cell refinement, not a trajectory smoother. No
refined neighbor may alter another anchor's candidate set or select its winner.
Freeze boundary clipping explicitly, and retain candidate-count and unavailable
counts; the old usable center must remain an included comparison. Because that
center is included, the refined minimum cannot exceed its old single-frame cost.
This is an algorithmic consequence, not evidence of synchronization accuracy.
[broadcast_alignment.py:96–110](broadcast_alignment.py#L96-L110),
[broadcast_alignment.py:122–133](broadcast_alignment.py#L122-L133).

Refine the **original far-coarse runner** independently inside its own +/-12
half-cell and original reflection. Retain both the original global far margin
`old_runner_cost - old_winner_cost >= .5` and an additional refined-runner margin
`refined_runner_cost - refined_winner_cost >= .5`. Apply both across each
three-anchor neighborhood. The runner can improve more than the winner, so the
second inequality does not follow from the first. Inference from the 25-grid and
strict far rule: the original far runner is at least 275 source frames away;
moving both centers by at most 12 leaves them at least 251 apart.
[broadcast_alignment.py:96–110](broadcast_alignment.py#L96-L110),
[broadcast_alignment.py:139–142](broadcast_alignment.py#L139-L142).

Refining only the original winning far alternative does **not** exhaustively
resolve sub-grid global ambiguity: another coarse alternative, another
reflection or an unavailable coarse center's surrounding usable source frames
could contain a lower sub-grid residual. Describe the refined margin as an
additional check against that original alternative. The original global lattice
margin remains a retained safeguard, not a certified continuous-time margin.
This limitation follows directly from the original coarse candidate set and the
proposed bounded two-candidate refinement.
[broadcast_alignment.py:97–110](broadcast_alignment.py#L97-L110).

## Refined neighborhood, fine search and mapping

Use the refined single-frame costs <=2.5, effective margin equal to the minimum
of the retained original and refined-original-runner margins >=.5, constant
original reflection across three anchors, and **refined** source increments in
the inclusive range 0..50. Keep these predicates separate in the trace so a
neighbor veto is distinguishable from a center failure. Require the refined
single-frame uncertainty gate on all three coarse neighbors: full span of
candidates within best+.05 <=10. Freeze boundary clipping and report any
boundary-censored spans. The existing fine span uses
last-minus-first across all qualifying candidates, including disconnected sets;
it is not a confidence interval.
[broadcast_alignment.py:115–133](broadcast_alignment.py#L115-L133),
[anchor_failures.py:54–64](anchor_failures.py#L54-L64),
[ANCHOR_FAILURE_REPORT.md:131–132](ANCHOR_FAILURE_REPORT.md#L131-L132).

After this new prefilter, run the original fine objective over integer source
centers within +/-25 of the refined center, at the same original reflection.
Keep the original source-center bounds, tactical/broadcast comparison offsets
-25/0/+25, per-comparison usability masks, squared-cost mean, first minimum,
cost<=2.5 and full best+.05 span<=10. Then check both paired tactical/broadcast
offsets -12/+12 at the chosen fine center, requiring both usable and each forward
residual<=2.5. Check only the center's own fine winner at these offsets: do not
recursively require neighbors' fine winners or their extra-offset gates, which
would broaden the tactical dependency. These extra offsets are decision evidence
once gated, so they must not later be presented as unused validation. Fine
recentering can remove a
previous fine optimum from the new +/-25 range; unchanged thresholds therefore
do not imply the candidate preserves old mappings.
[broadcast_alignment.py:121–139](broadcast_alignment.py#L121-L139),
[anchor_failures.py:223–238](anchor_failures.py#L223-L238).

Map only adjacent allowed refined endpoints with equal reflection and fine
source/tactical rate in [.8,1.2]. Retain original rounding, inclusive knot order
and final broadcast usability mask. Do not interpolate across a rejected endpoint,
force bridges or add a tactical-interior validity condition to the original
baseline. At exact knots, evaluate both incident intervals when attributing
absence. Single-anchor acceptance is not equivalent to interval or row support.
[broadcast_alignment.py:137–161](broadcast_alignment.py#L137-L161),
[anchor_failures.py:107–147](anchor_failures.py#L107-L147),
[test_anchor_failures.py:49–74](test_anchor_failures.py#L49-L74).

Construct the primary extension row by row: where original `frames>=0`, copy its
frames, reflection, cost and margin exactly; elsewhere use the candidate's row.
Keep original and candidate anchor-allowed arrays separately, because a per-row
union has no single original-or-refined anchor derivation. Do not manufacture an
allowed-anchor mask implying every union row came from one consistent trajectory.
The bounded inverse sampler must still break components on missing rows,
reflection changes, decreasing source times or source steps above 2. Test old/new
joins; filling a row does not itself establish clip eligibility.
[broadcast_alignment.py:144–161](broadcast_alignment.py#L144-L161),
[alignment_coverage.py:38–64](alignment_coverage.py#L38-L64),
[test_alignment_coverage.py:28–41](test_alignment_coverage.py#L28-L41).

## Unused checkpoints and coverage guards

Use fixed tactical checkpoints `f % 25 in {6,19}`, at each mapping's exact rounded
source frame and reflection, without an additional source search. Matching uses
tactical residues 0; the new +/-12 gate uses residues 12 and 13. Thus 6 and 19
are distinct tactical observations from those decision residues. They remain
correlated same-source consistency evidence, not correspondence ground truth or
an independent match. The earlier review explicitly describes this quarter-
interval diagnostic at offsets 6 and 19.
[broadcast_alignment.py:96–97](broadcast_alignment.py#L96-L97),
[broadcast_alignment.py:125–133](broadcast_alignment.py#L125-L133),
[ANCHOR_FAILURE_DESIGN_RESEARCH.md:240–253](ANCHOR_FAILURE_DESIGN_RESEARCH.md#L240-L253).

Freeze a minimum usable-checkpoint denominator and explicit empty-group failure
before observations. Report eligible lattice points, mapped points, both-source-
usable comparisons and geometry counts separately for baseline, candidate and
extension, plus **newly filled rows** as their own stratum. Give missingness its
own denominator; a small favorable subset must not conceal missing checkpoints.
Report candidate changes against baseline on overlap without altering the
extension's preserved rows. Do not tune gates, select subgroups or modify mapping
using checkpoint outcomes. The earlier protocol already requires fixed label-free
windows and separates observation coverage from source/clip counts.
[alignment_coverage.py:130–150](alignment_coverage.py#L130-L150),
[ALIGNMENT_COVERAGE_PLAN.json:41–53](ALIGNMENT_COVERAGE_PLAN.json#L41-L53).

Keep opportunity gates conjunctive and choose the extension as the single primary
in advance. The chosen proposal requires >=3,000 safe public query frames and
>=750 net additional safe query frames against Round 23 strict inverse-25 support
(reported baseline 2,328), exact preservation of every old mapped row, >=.98
old-query retention, and all source/prefix/replay guards. The simultaneous total
and net thresholds therefore require at least 3,078 query frames if the reported
baseline reproduces. This is a fixed engineering opportunity requirement, not a
power calculation or accuracy guarantee. Freeze the reported baseline in the
plan and verify it before comparing the new result. The previous frozen protocol
requires the same types of opportunity and parity conditions but compared to a
different original mask.
[ALIGNMENT_COVERAGE_PLAN.json:41–53](ALIGNMENT_COVERAGE_PLAN.json#L41-L53).

For the chosen extra consistency gate, require >=50 usable unused checkpoints
on newly mapped extension rows, and their directed residual-above-2.5 fraction
no more than the baseline fraction plus .02. Freeze the baseline checkpoint
universe and usability rule, and fail empty/undefined denominators. Availability
and fractions remain descriptive same-source comparisons; the threshold does not
calibrate synchronization correctness. These values are the proposed Round 25
protocol choices, not measurements calculated in this review.

Report candidate and union mapping coverage, safe
inverse-25 associations, distinct source centers, observed queries, additions,
losses and fixed windows separately; do not select whichever yields the best
headline. A positive metadata opportunity still does not authorize fitting.
[ALIGNMENT_COVERAGE_PLAN.json:8–20](ALIGNMENT_COVERAGE_PLAN.json#L8-L20),
[ALIGNMENT_COVERAGE_PLAN.json:41–53](ALIGNMENT_COVERAGE_PLAN.json#L41-L53),
[alignment_coverage.py:130–150](alignment_coverage.py#L130-L150).

## Dependency, exclusions and reproducibility

Independent half-cell searches inspect only the current tactical anchor. Refined
neighborhoods use its adjacent anchors; fine search uses tactical offsets +/-25;
the extra gate uses +/-12. For a mapped row between a and a+25 the full endpoint
dependency remains a-25..a+50, and at a knot the incident-interval union is bounded
by row +/-50. Neither a far-runner search nor per-row extension adds a tactical
smoothing dependency. Global broadcast tracking still enters the original winner
and far-runner choice. This is a tactical-axis bound, not all-source locality.
[broadcast_alignment.py:93–133](broadcast_alignment.py#L93-L133),
[broadcast_alignment.py:139–158](broadcast_alignment.py#L139-L158),
[ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md:89–126](ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md#L89-L126).

Retain the bounded inverse scan +/-125, alignment context 50, query radius 25 and
summary radius 50: total tactical dependency <=250. Eligibility and missingness
must inspect no mapping outside the bounded inverse scan. Retain +/-175 schedule
decision windows inside a single official half and actual tactical prefix, with
sealed/held intervals expanded by 250 and inclusive intersections. Check all 199
intervening source frames for each 100-frame stride-2 clip and actual video bounds,
not just sampled frames or tracking length. Source-center deduplication must
retain every tactical association.
[ALIGNMENT_COVERAGE_PLAN.json:22–40](ALIGNMENT_COVERAGE_PLAN.json#L22-L40),
[alignment_coverage.py:16–64](alignment_coverage.py#L16-L64),
[test_alignment_coverage.py:37–60](test_alignment_coverage.py#L37-L60).

Actually reconstruct baseline, original runner identities, refined anchors,
candidate mapping, extension, schedules, queries and diagnostics independently
from 18,935 and 19,060 tactical prefixes; compare identical public rows plus
required context. Do not treat slicing one full computation as independent
reconstruction. Require byte-exact second-process artifact replay and exact
original baseline/candidate preservation audits. Metadata parity does not certify
actual action-feature parity, which belongs to a later separately frozen stage.
[ANCHOR_FAILURE_REPORT.md:147–155](ANCHOR_FAILURE_REPORT.md#L147-L155),
[ALIGNMENT_COVERAGE_PLAN.json:40](ALIGNMENT_COVERAGE_PLAN.json#L40),
[test_anchor_failures.py:97–105](test_anchor_failures.py#L97-L105),
[test_alignment_coverage.py:74–83](test_alignment_coverage.py#L74-L83).

Add synthetic cases for a true non-grid offset, an independently refined
neighbor, retained coarse ambiguity, a runner that improves more than the winner,
absent far runner, all four reflections, exact 250 versus 275 exclusion, inclusive
0/50 increments, equal-cost ties, unavailable/boundary half-cell candidates,
disconnected span sets, fine recentering, extra-offset missingness, rate limits,
shared-knot overwrite, old/new joins and bit-exact baseline extension. Preserve
the existing far-copy rejection, reflection/permutation, source-cut and prefix
contracts. Store detailed source identities and traces under ignored local data;
the public report should contain aggregate evidence only.
[test_broadcast_alignment.py:27–62](test_broadcast_alignment.py#L27-L62),
[test_anchor_failures.py:30–119](test_anchor_failures.py#L30-L119),
[test_alignment_coverage.py:20–105](test_alignment_coverage.py#L20-L105),
[ANCHOR_FAILURE_PLAN.json](ANCHOR_FAILURE_PLAN.json).

No result from this investigation establishes synchronization truth, private
accuracy, possession improvement or maximum score. Preserve the selected model,
ledger and existing promotion requirements; close the metadata round before any
new image processing, action heads, fits or scores.
[ANCHOR_FAILURE_REPORT.md:207–213](ANCHOR_FAILURE_REPORT.md#L207-L213),
[ALIGNMENT_COVERAGE_PLAN.json:52–54](ALIGNMENT_COVERAGE_PLAN.json#L52-L54).


## Frozen execution clarification

The authoritative [Round 25 plan](SUBGRID_ALIGNMENT_PLAN.json) uses the
hash-verified Round 24 original anchors for each actual prefix and reconstructs
the original strict map exactly. New global coarse replay and runner extraction
occur only at anchors with safe +/-25 context; excluded tactical samples receive
no new geometry. Thus original full-prefix coarse reconstruction is inherited,
while safe coarse search and the new candidate are independently reconstructed.
The [completed report](SUBGRID_ALIGNMENT_REPORT.md) distinguishes these proofs
and records the execution-only whole-grid rejection-counter comparison repair.
No method or threshold changed during that repair.
