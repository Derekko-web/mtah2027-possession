# Bounded strict-alignment coverage research

Prepared 2026-10-23 from repository source, plans, reports, tests and official
challenge documentation only. This investigation did not read competition media,
tracking arrays/XML, boxes, raw model outputs, possession target rows or derived
per-frame extracts, and did not run observation processing, inference, fitting or
scoring. The research skill's primary-source workflow was used; this is the
already-dispatched background research task, so no further agent was created.

## Recommendation

Freeze **one metadata-only protocol** with two nested, predetermined schedules:
the existing 50-frame center grid and the same range on a 25-frame center grid.
For both, replace the blanket complete `c +/-150` map requirement with an exact
source-footprint test in the center's strict-map component, **searching only
`c +/-125`**. Preserve the cached strict mapping and all of its thresholds. Count
the resulting associations, distinct clips and actual dense-query coverage, with
no choice of new thresholds after seeing counts. This can test whether the
previous scheduler discarded usable partial components; it does not establish
that coverage will improve or that any new feature predicts possession better.

There is one material safety correction: the old 225-frame calculation bounds
image content, not all alignment-dependent feature missingness. The new protocol
must account for the tactical observations used to decide whether an eligible
clip exists. A bounded `125 + 50 + 25 + 50 = 250` dependency budget can do so.
An unbounded search of a whole alignment component cannot make that guarantee.

## What the primary code establishes

1. **The clip is fixed; the 150-frame completeness window is a scheduler rule.**
   The model's source indexes are `b - 100 + 2 * arange(100)`, ending at `b +98`.
   The scheduler independently asks for every mapped tactical row in `c +/-150`
   to be present, monotone, orientation-constant and in one half/prefix; it also
   requires every source sample to be within one broadcast frame of a local
   mapped row and requires source endpoints inside the mapped range.
   [sports_action_sampling.py:19–20](sports_action_sampling.py#L19-L20),
   [broadcast_action_sampling.py:6–25](broadcast_action_sampling.py#L6-L25).

2. **Strict alignment already contains the correspondence guards to preserve.**
   The strict profile is maximum formation cost 2.5, minimum far-runner-up margin
   .5 and maximum fine uncertainty span 10. Fine matching compares offsets
   `-25,0,+25`; mapping interpolates only between adjacent accepted 25-frame
   anchors at rates `.8..1.2`, with a common orientation, and rechecks mapped
   broadcast validity/player counts. The mapper never fills rejected gaps.
   [BROADCAST_ALIGNMENT_PLAN.json](BROADCAST_ALIGNMENT_PLAN.json),
   [broadcast_alignment.py:92–161](broadcast_alignment.py#L92-L161).
   The synthetic tests explicitly reject cuts/missing players and ambiguous far
   replays. [test_broadcast_alignment.py:37–62](test_broadcast_alignment.py#L37-L62).

3. **Dense-query support is narrower than a clip association count.**
   The existing feature code first selects the nearest eligible tactical center
   within 25, choosing the earlier center on ties, then requires the query and
   center to be in the same strict-map component. It interpolates at the query's
   mapped broadcast frame inside that clip. Components break on invalid mapping,
   orientation changes, reverse or greater-than-two-frame mapping steps and half
   boundaries. Summaries use only contiguous observed rows in one component and
   cover inclusive past/future radii 25 and 50. Therefore an interval painted
   `c +/-25` is only a preliminary upper bound on coverage; execute the actual
   selector/component/broadcast-range rules when counting observed queries.
   [broadcast_action_features.py:10–62](broadcast_action_features.py#L10-L62),
   [broadcast_action_sampling.py:46–58](broadcast_action_sampling.py#L46-L58),
   [sports_action_features.py:71–91](sports_action_features.py#L71-L91).

4. **Tracking and image indexes have different final bounds.**
   `games.py` describes 164,488 broadcast tracking records, but the checked video
   container has 164,487 frames. Its valid image indexes are **0..164486**. A
   source sample equal to 164487 must be rejected, even though the old component
   helper's `<164488` test would accept it. The observation adapter already obtains
   the actual count from `validate_video_metadata` and passes it to the scheduler.
   [games.py:78–89](../games.py#L78-L89),
   [broadcast_action_sampling.py:46–63](broadcast_action_sampling.py#L46-L63),
   [prepare_broadcast_action_observations.py:75–84](prepare_broadcast_action_observations.py#L75-L84),
   [test_broadcast_action.py:33–49](test_broadcast_action.py#L33-L49).
   The earlier feasibility report's assertion of 164,488 **video** frames is
   superseded by the later execution correction.
   [BROADCAST_ACTION_REPORT.md:48–55](BROADCAST_ACTION_REPORT.md#L48-L55).

5. **Parity and source reuse are separate from correspondence quality.**
   Broadcast centers may be odd or even irrespective of tactical-grid parity.
   Existing extraction builds the union of separate odd/even stride-two runs,
   and inference deduplicates broadcast centers while retaining every tactical
   association. Support peaks count unique source clips, not associations.
   [broadcast_action_sampling.py:29–43](broadcast_action_sampling.py#L29-L43),
   [broadcast_action_features.py:65–85](broadcast_action_features.py#L65-L85),
   [test_broadcast_action.py:15–40](test_broadcast_action.py#L15-L40).

## Dependency accounting: images, mapping and missingness

For a tactical mapping row `t` between adjacent anchors `a` and `a+25`, fine and
coarse eligibility of both endpoints can inspect tactical anchor observations
from `a-25` through `a+50`. At an exact anchor, either adjacent interval can supply
the row; proving its absence must consider both. Thus **`t +/-50` is a safe
conservative tactical-observation dependency bound**, including invalidity. This
follows from the three-anchor coarse checks, three fine offsets and adjacent
endpoint interpolation, rather than the mapped image indexes alone.
[broadcast_alignment.py:115–159](broadcast_alignment.py#L115-L159).

In Round 22, selecting or rejecting a candidate can inspect mapping throughout
`c +/-150`. Even if the RGB content is confined there, the decision's underlying
tactical observations can reach `c +/-200`. With a center up to 25 from a dense
query and a summary query up to 50 from the final feature row, this gives a
conservative **275-frame eligibility/missingness bound**. The report's stated
`150 +25 +50 =225` remains an image-content bound. Its observed full/short prefix
equality is valuable empirical evidence but does not prove a general 225-frame
bound on every feature dependency.
[broadcast_action_sampling.py:8–24](broadcast_action_sampling.py#L8-L24),
[BROADCAST_ACTION_REPORT.md:37–45](BROADCAST_ACTION_REPORT.md#L37-L45),
[BROADCAST_ACTION_REPORT.md:69–75](BROADCAST_ACTION_REPORT.md#L69-L75).

Restricting **all candidate map inspection**, including failed searches, to
`c +/-125` bounds its tactical-observation dependencies by `c +/-175`, and final
features by `125 +50 +25 +50 =250`. This bound must cover the availability of a
closer candidate, an absent candidate, `seen`, the chosen source and summary-run
edges, not just the accepted source samples. A scheduler that searches arbitrarily
far along a component before finding a small final witness interval does not
qualify. Likewise, schedule decisions must not depend on total component length
or remote component endpoints. Component-number equality may be computed globally,
but its local connectivity interpretation must use only edges on the relevant
bounded path.

This is a **tactical temporal bound**, not an all-source observation bound. Coarse
matching and the far-runner-up margin search all eligible broadcast anchors;
broadcast alignment has global broadcast-tracking dependencies. The new metadata
protocol reuses that unchanged mapping. It cannot claim that broadcast tracking
outside a local clip was unused or that fallible correspondences are synchronization
ground truth. [broadcast_alignment.py:96–110](broadcast_alignment.py#L96-L110).

## One concrete metadata-only protocol

Freeze the code and protocol hashes before any new counts. Use the following
fixed steps for both schedules; the 25-frame schedule is a nested sampling
comparison, not a sweep over alignment or support thresholds.

1. **Whitelist inputs.** Use only the preserved strict mapping metadata and its
   manifest; already-produced independent mappings for actual tactical prefixes
   18,935 and 19,060; official half constants; the 24 reserved-window intervals;
   the 60 held weak-image frame indexes; and the previously verified container
   metadata. Keep schedules, source indexes and all per-frame metadata in a new
   ignored data directory. Do not load source positions/speeds/ball arrays, raw
   media, model heads, feature caches or possession labels. The existing prefix
   program explains how the independent cached maps were derived, but rerunning
   it would read tracking arrays and is outside this metadata-only phase.
   [check_broadcast_action_prefix.py:18–44](check_broadcast_action_prefix.py#L18-L44).

2. **Use two fixed grids.** Enumerate `11000..18950` inclusive at steps 50 and 25.
   Do not move centers around known errors or choose regions from score reports.
   The candidate ceilings are 160 and 319 associations respectively, before
   guards and source-center deduplication. The initial range is inherited from
   the existing source scheduler. [broadcast_action_sampling.py:8](broadcast_action_sampling.py#L8).

3. **Guard the full dependency window conservatively.** Require `c +/-175` inside
   the actual tactical prefix and one official half. Reject it if it intersects
   any reserved interval expanded by 250 or any held frame expanded by 250,
   using inclusive intersections and preserving the exact interval/frame set.
   This conservatively excludes both image witnesses and the tactical observations
   supporting their eligibility; static exclusion metadata can reject early.
   Do not shrink the existing 250-frame exclusions. Official halves are derived
   per half, not by a single global offset.
   [games.py:197–208](../games.py#L197-L208),
   [broadcast_action_sampling.py:10–15](broadcast_action_sampling.py#L10-L15).

4. **Construct the exact local witness.** Inside `c +/-125`, find the strict-map
   component containing `c` using only bounded local edges. Require the center
   and candidate source samples to be in actual video bounds `<164487` and use
   exactly `clip_frames(mapping.frames[c])`. Preserve the source-endpoint-in-map-
   range test. For every one of the 100 stride-two source indexes, find a mapped
   tactical witness in this component at distance at most one broadcast frame;
   choose the minimum absolute distance, then the earlier tactical row. The hull
   of these witnesses and `c` is the explicit image footprint. Every tactical
   row in that hull must remain in the same valid, orientation-constant,
   nondecreasing component with steps at most two. Reject absent witnesses,
   clipping outside source bounds, cuts and reversed mappings. Do not impute
   holes or extrapolate. Neither finding witnesses nor rejecting a candidate may
   inspect mapping outside `c +/-125`.

   Checking all 199 intervening broadcast indexes as well would be additional
   conservatism, rather than a requirement of this 100-image model input. If
   adopted, freeze it explicitly and use the same bounded mapping metadata;
   do not read broadcast player arrays to introduce a new validity criterion in
   this phase. The clip definition and old local-witness tolerance are owned by
   [sports_action_sampling.py:19–20](sports_action_sampling.py#L19-L20) and
   [broadcast_action_sampling.py:20–23](broadcast_action_sampling.py#L20-L23).

5. **Count actual query coverage.** Retain every tactical association and infer
   nothing. Deduplicate broadcast centers for clip counts; count both source
   parities and the exact source-frame union. Replay the existing nearest-center,
   earlier-tie, 25-frame, component and clip-range logic to obtain the structural
   observed-query mask. Preserve missing queries as missing. Do not choose a
   farther center to bypass the existing nearest-center component rejection.
   Record the chosen source and bracketing source indexes/weights locally; no raw
   heads are needed. Report aggregate public observed frames, gained/lost/retained
   frames relative to the inherited scheduler, number of observed runs and gap
   lengths, associations, unique sources, duplicate associations, parity counts,
   rejection reasons, maximum witness radius and certified dependency radius.
   If short-interval coverage is counted, use a frozen set of uniform sliding
   windows independent of target segments, with fixed lengths/coverage rules.
   Such counts are not short-possession recovery, action peaks or recognition
   accuracy. [broadcast_action_features.py:18–54](broadcast_action_features.py#L18-L54).

6. **Verify prefix and summary structure.** Run the complete metadata selection
   against both already independently rebuilt actual-prefix mappings. Compare all
   public rows `11185:18685`, including selected source, interpolation metadata,
   seen/missing mask and the component/observed-run membership of every inclusive
   past/future 25/50 summary window. A useful synthetic check additionally drives
   all 121 feature columns with deterministic dummy clip values and verifies
   exact NaN/value equality. Do not mistake comparing a truncated full mapping
   with itself for an independent prefix-map replay. The short prefix ends at
   18,934, exactly the last public row 18,684 plus the 250-frame conservative
   bound. Actual neural-feature parity remains a later-stage requirement, since
   this phase has no new neural outputs.
   [check_broadcast_action_prefix.py:30–47](check_broadcast_action_prefix.py#L30-L47),
   [test_broadcast_action.py:83–98](test_broadcast_action.py#L83-L98).

7. **Close the phase after counting.** Verify hashes and absence of ledger,
   selected-candidate or protected-artifact mutations. Freeze an aggregate
   report whether counts improve, regress or remain equal. A denser schedule is
   a coverage proposal only; the new 50-frame schedule is not guaranteed to
   retain all previous associations because the dependency exclusions are more
   conservative. Do not automatically decode images, run action models, fit or
   score. Any later observation/model round needs its own frozen protocol and
   must preserve the original substantial-gain, short-match, paired-protocol,
   D/macro/official, seeds 42/43/44, reproduction and independent-review gates.
   [BROADCAST_ACTION_PLAN.json](BROADCAST_ACTION_PLAN.json).

## Blocking conditions and limits

Do not certify the 250-frame tactical dependency bound if any candidate decision
reads outside `c +/-125`, the mapper implementation changes its tactical context,
the required independent-prefix metadata provenance is missing, or local/summary
parity fails. Stop before model processing rather than weaken an alignment,
exclusion or promotion requirement. The metadata calculation alone can identify
more safely scheduled images; it cannot supply missing correspondence truth.

All detailed artifacts must stay ignored. Official rules prohibit redistribution
of the competition assets and reconstructive derived exports, private ground
truth use and manual output correction.
[RULES.md:35–59](../RULES.md#L35-L59),
[RULES.md:70–81](../RULES.md#L70-L81). No competition-specific accuracy or maximum
score claim follows from this source-code research.
