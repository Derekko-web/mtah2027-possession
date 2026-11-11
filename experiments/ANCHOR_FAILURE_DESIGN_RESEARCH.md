# Strict formation anchor failure: source-based diagnostic design

Prepared 2026-10-23. This is a design investigation from the six requested local
source, plan, test and report files. It did not inspect media, tracking arrays or
XML, boxes, model heads, possession targets or per-frame extracts; it ran no
observation processing, inference, fitting or scoring. It follows the research
skill's primary-source workflow as the already-dispatched research agent.

## Recommendation and limits

Instrument the **existing computation**, preserving its complete original
strict-plus-relaxed profile set and all thresholds. Record each gate's inputs,
attempt status and result, then derive both a declared mutually exclusive
first-failure waterfall and overlapping predicate counts. Separately trace final
frame absence through all intervals that could supply that frame. A rejected
anchor, a rejected interval and an unmapped public frame are different units.
These diagnostics explain executed rules; they do not identify incorrect real
correspondences or show which rule can safely be relaxed.
[broadcast_alignment.py:92–161](broadcast_alignment.py#L92-L161).

The latest report establishes a useful question, not its answer: it records
3,359/7,500 public rows with strict mapping and 4,141 without it. The 1,031 strict
mapped rows lacking complete safe clip support belong to a later scheduling
problem, so do not attribute them to anchor rejection. These are reported
historical aggregates; this investigation did not recalculate them.
[ALIGNMENT_COVERAGE_REPORT.md:24–29](ALIGNMENT_COVERAGE_REPORT.md#L24-L29).

## Baseline that must be reproduced

The original preparation program passes **both** original profiles into one
`formation_anchors` call. Consequently the fine search's neighborhood prefilter
uses the maximum coarse cost **3.5** and minimum far margin **.25**, while strict
acceptance later uses fine/coarse cost **2.5**, far margin **.5** and fine span
**10**. Running formation search with only the strict profile changes which fine
rows are computed, even if the resulting strict mapping happens to agree. Such a
run cannot faithfully explain why a cached original fine row is absent.
[prepare_broadcast_alignment.py:38–48](prepare_broadcast_alignment.py#L38-L48),
[broadcast_alignment.py:112–142](broadcast_alignment.py#L112-L142),
[BROADCAST_ALIGNMENT_PLAN.json:14–18](BROADCAST_ALIGNMENT_PLAN.json#L14-L18).

Coarse anchors lie on each source's zero-based 25-frame lattice, excluding a
25-frame margin. Each tactical query searches all eligible broadcast lattice
anchors and four reflections. A source frame is usable if its supplied homography
validity is true and each team has at least four finite player coordinates.
`load_source` first sanitizes nonfinite or out-of-pitch-with-five-metre-tolerance
coordinates, whereas `frame_usable` itself only tests finiteness and team counts.
Do not substitute a newly invented validity predicate for either stage.
[broadcast_alignment.py:30–55](broadcast_alignment.py#L30-L55),
[broadcast_alignment.py:65–110](broadcast_alignment.py#L65-L110).

If there are no eligible broadcast lattice anchors, the baseline raises globally;
it does not create a per-tactical-row rejection. Preserve and report that abort
separately. An unusable tactical center leaves its coarse row NaN. Fine search
does not attempt the first or last tactical lattice anchor, even if its coarse
row is valid.
[broadcast_alignment.py:96–102](broadcast_alignment.py#L96-L102),
[broadcast_alignment.py:114–117](broadcast_alignment.py#L114-L117).

## Ordered anchor attribution

Use the original tactical lattice as the denominator. The following order is a
**declared reporting precedence**, matching the computation's stages; when
multiple predicates fail, the primary label is not a unique physical cause.
Keep the underlying booleans and unknown/not-attempted states as well.
[broadcast_alignment.py:100–142](broadcast_alignment.py#L100-L142).

| Order | Exclusive outcome | Evidence and qualification |
|---:|---|---|
| 1 | Coarse center source unavailable | Original `usable_t[frame]` is false. Record homography invalid, team 0 below four and team 1 below four as overlapping flags. |
| 2 | Fine lattice edge | Center is the first or last tactical lattice anchor; the fine loop never visits it. |
| 3 | Coarse neighborhood unavailable | Any of the three neighboring rows has a nonfinite broadcast frame, orientation or cost. Preserve left/center/right source-availability flags; unexpected missing rows with usable source are an integrity finding. |
| 4 | Coarse geometry prefilter failed | Any of those three costs is greater than 3.5, using the original shared-profile prefilter. |
| 5 | Far ambiguity prefilter failed | Any of those three margins is less than .25. |
| 6 | Neighbor orientation failed | Three coarse orientations do not all equal the center orientation. |
| 7 | Neighbor increment failed | Either broadcast increment differs from 25 by more than 25. |
| 8 | Fine candidate/source failure | Fine was attempted but no finite best score exists. Distinguish no candidate with all three valid source comparisons from unexpected nonfinite geometry despite valid comparisons. |
| 9 | Strict fine cost failed | Fine exists but its cost is greater than 2.5. |
| 10 | Strict fine span failed | Fine exists but its uncertainty span is greater than 10. |
| 11 | Strict coarse cost failed | Fine passed its own gates but any of the three strict coarse neighborhood costs exceeds 2.5. |
| 12 | Strict far margin failed | Earlier gates passed but any strict neighborhood margin is below .5. |
| 13 | Strict anchor allowed | The original `anchor_allowed` is true. This is not yet an accepted interval or final mapped frame. |

Orders 3–7 correspond to the original fine-search preconditions in their original
order. Orders 9–12 split the original strict allowed-mask expression and its
following neighborhood checks in a fixed, explicit order. A scalar diagnostic
must reproduce the vectorized booleans exactly, including NaN comparisons. A
nonfinite fine row when no recorded original gate explains it must be classified
as an integrity discrepancy rather than silently assigned a plausible reason.
[broadcast_alignment.py:115–142](broadcast_alignment.py#L115-L142).

The source masks should separate tactical and broadcast validity at fine offsets
`-25,0,+25`. In an unmodified ordinary run, the three tactical comparison frames
are exactly the coarse neighborhood already required to be usable; therefore a
new tactical-invalid fine failure after that gate is unexpected. Broadcast
validity varies over candidate-plus-offset frames and can genuinely eliminate
every candidate. Do not infer that distinction from the saved all-NaN fine row
alone; record the masks while the baseline evaluates its candidates.
[broadcast_alignment.py:95–102](broadcast_alignment.py#L95-L102),
[broadcast_alignment.py:115–133](broadcast_alignment.py#L115-L133).

## Overlapping counts and unavailable evidence

In addition to the exclusive waterfall, publish aggregate counts for each raw
failure predicate, its evaluated denominator and unknown count. Include center
and flank usability; strict and shared-profile neighborhood cost/margin failures;
orientation and each increment failure; fine-attempted status; number of bounded
candidates; number with all three comparisons usable; each fine validity offset;
strict fine cost/span; and final allowed status. Joint counts for cost-plus-margin,
orientation-plus-increment, and fine-cost-plus-span can show overlap without
inventing unique causes. Count center-only failures separately from failures in
either neighbor, since the actual acceptance rule is neighborhood-based.
[broadcast_alignment.py:115–142](broadcast_alignment.py#L115-L142).

Do not label a blocked fine search as a fine geometry or span failure: no such
original score exists. If an extra diagnostic later evaluates a fixed rejected
candidate, mark it as a separately declared diagnostic observation; it must not
fill the original fine table or become an accepted anchor. Existing preparation
diagnostics record only coarse/fine availability and medians, so they cannot by
themselves supply this causal attribution.
[prepare_broadcast_alignment.py:65–71](prepare_broadcast_alignment.py#L65-L71),
[broadcast_alignment.py:114–134](broadcast_alignment.py#L114-L134).

An empty fine candidate set is not a normal missing-row reason: the original
`argmin` would raise. A valid original coarse winner is itself within the bounded
fine candidate range, so that set should be nonempty. Keep an invariant check and
report an actual violation as an abort/integrity discrepancy instead of silently
turning it into an ordinary rejection.
[broadcast_alignment.py:97–110](broadcast_alignment.py#L97-L110),
[broadcast_alignment.py:122–131](broadcast_alignment.py#L122-L131).

## Interval and public-frame attribution

An interval between adjacent tactical anchors proceeds only if both original
allowed flags pass, then their coarse orientations agree, then their fine
broadcast displacement divided by 25 lies in `.8..1.2`. Interpolation is inclusive
of both endpoints, uses `np.rint`, and is processed in increasing interval order.
Finally only the rounded **broadcast** source frame's usability is rechecked.
There is no new interior tactical-frame usability check in `map_anchors`.
[broadcast_alignment.py:137–161](broadcast_alignment.py#L137-L161).

Keep an interval table with a primary outcome in the order: endpoint not allowed;
orientation mismatch; rate outside the inclusive range; interpolated. Also keep
overlapping endpoint, orientation and rate predicates wherever the required
values exist. A 25-frame coarse increment guard permits broadcast increments
**0..50 inclusive**, while an accepted final interval needs fine displacement
**20..30 inclusive**; these are distinct guards with different inputs.
[broadcast_alignment.py:119–120](broadcast_alignment.py#L119-L120),
[broadcast_alignment.py:147–150](broadcast_alignment.py#L147-L150).

For every public tactical row, enumerate every incident lattice interval. Most
rows have one; an exact lattice anchor can be supplied by either adjacent
interval. Define nested sets of incident intervals that pass endpoint, orientation
and rate gates, then assign this exclusive frame outcome:

1. Outside the lattice's interpolation domain: no incident interval exists.
2. No incident interval has two allowed endpoints.
3. Some interval has two allowed endpoints, but none also passes orientation.
4. Some interval passes endpoints and orientation, but none also passes rate.
5. At least one interval interpolates, but the final mapped broadcast row is
   unusable. Record homography/team-count failures as overlapping flags.
6. Final strict mapped frame.

This is an OR over possible supplying intervals, not rejection of a nearest
anchor. Replay forward overwrite order at shared knots and compare the complete
original mapping arrays, since `margin` can be overwritten from a different
neighboring pair even where mapped frame, orientation and fine cost agree. Report
both interval and frame units; do not multiply rejected anchors by 25 to estimate
missing frames.
[broadcast_alignment.py:146–161](broadcast_alignment.py#L146-L161).

## Source-code pitfalls and unsupported interpretations

- **The cost is directed, not reciprocal.** For each team it asks how close each
  visible broadcast player is to its nearest tactical player, caps distances at
  10 metres, averages squared distances within that broadcast team, then gives
  the two teams equal weight. Several broadcast players can use the same tactical
  nearest neighbor, and tactical players with no broadcast counterpart do not
  directly penalize this direction. The existing symmetry test establishes slot
  order and team-label invariance, not exchangeability of camera directions.
  [broadcast_alignment.py:72–89](broadcast_alignment.py#L72-L89),
  [test_broadcast_alignment.py:45–53](test_broadcast_alignment.py#L45-L53).
- **Far margin is not a full ambiguity probability.** The runner searches all
  orientations only at broadcast lattice times strictly more than 250 frames
  from the selected best time. A time exactly 250 away and alternative
  orientations at nearby times are excluded. If no far time exists, the runner
  and margin are positive infinity, which passes the original margin threshold.
  The finite-neighborhood test deliberately covers only the first three coarse
  columns; an instrument that requires finite margin would change semantics.
  [broadcast_alignment.py:103–118](broadcast_alignment.py#L103-L118).
- **Ties and span have exact implementation meanings.** Coarse `argmin` chooses
  the first flattened orientation/time minimum; fine `argmin` chooses the first
  minimum in its ascending candidate list. Fine span is the last-minus-first
  candidate within `.05` of the best cost, even if that set is disconnected. It
  is not a confidence interval, an alternative count or correspondence truth.
  [broadcast_alignment.py:103–110](broadcast_alignment.py#L103-L110),
  [broadcast_alignment.py:122–133](broadcast_alignment.py#L122-L133).
- **The stored mapped margin omits some acceptance evidence.** It is only the
  minimum of the two central coarse margins. Each allowed endpoint additionally
  consulted its own three-anchor neighborhood, so the interval's complete gating
  evidence spans four coarse anchors. Final mapping fields alone cannot recover
  all rejection reasons.
  [broadcast_alignment.py:139–156](broadcast_alignment.py#L139-L156).
- **Rule failure does not identify a camera event.** High cost can indicate poor
  geometry, visibility, wrong timing or another mismatch; low far margin can
  indicate repeated formation rather than a verified replay. Orientation or
  increment failure is not proof of a camera cut. The synthetic cut and far-copy
  tests show intended behavior on constructed cases; no supplied real
  correspondence truth exists.
  [test_broadcast_alignment.py:13–42](test_broadcast_alignment.py#L13-L42),
  [test_broadcast_alignment.py:55–62](test_broadcast_alignment.py#L55-L62),
  [BROADCAST_ALIGNMENT_PLAN.json:41](BROADCAST_ALIGNMENT_PLAN.json#L41).

## Independent consistency diagnostics without another alignment

Freeze the statistics, source positions and aggregation strata before reading
new counts. The parent protocol selects the **existing coarse winner and finite
fine winner**, with fixed `-12,+12` offsets and reciprocal geometry; the
far-runner and quarter-interval checks below are alternatives for a separately
frozen investigation, not additional observations authorized by that protocol.
Record unavailable comparisons. Do not search for
a replacement time, add an anchor, sweep thresholds or select a favored subgroup
after observing results. Keep accepted and each rejected stage as fixed strata;
these are descriptive comparisons, not calibrated correctness labels.
[broadcast_alignment.py:103–133](broadcast_alignment.py#L103-L133),
[BROADCAST_ALIGNMENT_PLAN.json:41](BROADCAST_ALIGNMENT_PLAN.json#L41).

1. **Reciprocal and balanced same-team geometry.** At each existing candidate and
   its original orientation, report the unchanged broadcast-to-tactical score,
   the reverse tactical-to-broadcast score using the same cap and equal team
   weighting, and the square-root mean of the two directed squared scores. Also
   report visible counts per team, reciprocal-nearest-neighbor fraction and
   nearest-neighbor collision counts. This can expose directional subset matches
   and many-to-one crowding. Wider tactical visibility can legitimately raise
   reverse cost, so disagreement is a diagnostic concern, not automatic proof
   that an accepted time is wrong. A fixed one-to-one assignment residual with
   explicitly specified unequal-count handling is an optional alternative
   descriptive statistic; it must not become a new matching objective.
   The rationale follows from the existing directed cost's exact omissions.
   [broadcast_alignment.py:72–89](broadcast_alignment.py#L72-L89).
2. **Fixed interior offsets on unused timing positions.** The parent protocol's
   fixed `-12,+12` tactical/broadcast same-offset checks at original coarse and
   finite fine winners can describe local candidate consistency without a new
   trajectory. Report original-direction and reciprocal residuals and source
   availability; do not search neighboring broadcast frames. The original fit
   used lattice anchors and `-25,0,+25` offsets, so these add timing positions
   absent from that fitting objective. A separately frozen alternative is a
   quarter-interval check: for each originally accepted 25-frame interval,
   inspect offsets 6 and 19 from its left endpoint, using exactly the original
   interpolated/rounded broadcast frames and orientation. Never call a
   hypothetical rejected trajectory an existing mapping.
   [broadcast_alignment.py:96–97](broadcast_alignment.py#L96-L97),
   [broadcast_alignment.py:125–133](broadcast_alignment.py#L125-L133),
   [broadcast_alignment.py:148–156](broadcast_alignment.py#L148-L156).
3. **Candidate-versus-original-runner comparison.** Preserve the original far
   runner's identity and orientation while evaluating the unchanged cost matrix;
   its saved scalar margin alone does not identify it. Evaluate the same fixed
   reciprocal and quarter-offset statistics at that existing alternative. This
   can show whether an ambiguity is shared by both metrics, without picking a
   new winner or treating the alternate as truth.
   [broadcast_alignment.py:103–110](broadcast_alignment.py#L103-L110).

The original 2.5-metre threshold can be used as a fixed descriptive reference in
reciprocal reports, but it has not been calibrated for the reverse statistic's
different visibility behavior. Passing or failing it does not justify a changed
acceptance decision. Require each extra candidate-centered observation check's
entire `t +/-50` tactical dependency window to satisfy the frozen half, prefix,
sealed-window-plus-250 and held-frame-plus-250 guards; suppress unavailable checks
rather than shrink the guard after seeing counts. This follows the conservative
dependency and exclusion protocol already established in the previous review.
[ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md:89–126](ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md#L89-L126),
[ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md:151–160](ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md#L151-L160).

Ball disagreement is another complementary observation because ball position is
absent from formation cost, but it was already used in original profile feasibility
selection. It is therefore reused evidence, not a pristine independent holdout.
None of these same-source consistency checks is independent synchronization ground
truth or an estimate of possession accuracy.
[broadcast_alignment.py:72–89](broadcast_alignment.py#L72-L89),
[prepare_broadcast_alignment.py:49–64](prepare_broadcast_alignment.py#L49-L64),
[BROADCAST_ALIGNMENT_PLAN.json:20](BROADCAST_ALIGNMENT_PLAN.json#L20),
[BROADCAST_ALIGNMENT_PLAN.json:41](BROADCAST_ALIGNMENT_PLAN.json#L41).

## Tactical dependencies and prefix audit

For a row between `a` and `a+25`, the endpoint fine searches and their coarse
neighborhoods use tactical observations from `a-25` through `a+50`. At an exact
lattice knot, determining absence must consider both adjacent intervals, so the
safe per-row bound, including rejection/missingness, is **tactical +/-50**. The
proposed fixed interior/quarter checks do not require a broader tactical window.
However, the coarse winner and far-runner search retain global broadcast-tracking
dependencies. A tactical bound must not be described as an all-source local
bound.
[broadcast_alignment.py:115–159](broadcast_alignment.py#L115-L159),
[ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md:89–126](ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md#L89-L126).

Before accepting instrumentation, require exact original coarse/fine arrays and
all strict and relaxed mapping arrays, including NaNs, integers, costs, margins
and allowed masks. Preserve source/profile hashes. Add synthetic fixtures for
each gate and its inclusive boundary, simultaneous predicate failures, missing
center versus flank source, infinite far margin, both valid incident intervals
at a shared knot, one surviving incident interval, and broadcast-invalid
interpolated interiors. These checks extend the existing offset/reflection,
permutation, cut and far-replay contracts without changing their meaning.
[test_broadcast_alignment.py:27–62](test_broadcast_alignment.py#L27-L62).

Run actual independent-prefix reconstruction with the original joint profile set,
then compare public mappings **and diagnostic attribution**, rather than comparing
a sliced full mapping with itself. The original preparation already rebuilds a
shorter tactical prefix; the latest report records independent 18,935/19,060
prefix mappings and exact metadata/synthetic parity. New reason arrays should
agree wherever their complete declared local dependency window is present.
Global aggregates over unequal prefix lengths need not agree; compare identical
public-row or anchor universes. Include exact absence categories at knots, not
only accepted mapped frames.
[prepare_broadcast_alignment.py:72–85](prepare_broadcast_alignment.py#L72-L85),
[ALIGNMENT_COVERAGE_REPORT.md:104–115](ALIGNMENT_COVERAGE_REPORT.md#L104-L115).

Keep detailed gate traces, candidate identities and per-frame tables in ignored
local artifacts; publish aggregate counts and source citations. Preserve the
selected candidate and ledger. Closing this diagnostic stage should produce a
reasoned aggregate report even if no single rule dominates; it does not authorize
new synchronization, media extraction, fitting, scoring or promotion.
[ALIGNMENT_COVERAGE_REPORT.md:6–7](ALIGNMENT_COVERAGE_REPORT.md#L6-L7),
[ALIGNMENT_COVERAGE_REPORT.md:72–73](ALIGNMENT_COVERAGE_REPORT.md#L72-L73),
[ALIGNMENT_COVERAGE_REPORT.md:140–142](ALIGNMENT_COVERAGE_REPORT.md#L140-L142),
[ALIGNMENT_COVERAGE_REPORT.md:165–171](ALIGNMENT_COVERAGE_REPORT.md#L165-L171).


## Execution protocol authority

The executed diagnostic follows the category order frozen in
[ANCHOR_FAILURE_PLAN.json](ANCHOR_FAILURE_PLAN.json), including prefix-grid edge
before source unavailability. The table above recommends the opposite precedence
for those two conditions; this distinction affects only edge attribution, not the
production arrays. The executed extra checks are the fixed -12/+12 offsets and
reciprocal geometry in that plan; other suggested checks were not run. Results
are recorded in [ANCHOR_FAILURE_REPORT.md](ANCHOR_FAILURE_REPORT.md).
