# Temporal global formation matching: design preflight

Prepared 2026-10-23 from local source, plans, synthetic test source and aggregate
reports only. This is an advisory review of one proposed Round 26 method, not a
frozen protocol or an executed experiment. No competition observations, media,
tracking/XML, targets, reserved samples, derived per-frame arrays or caches were
opened; no observation processing, tests, inference, fitting or scoring ran. The
research skill is applied by the already-dispatched background reviewer.

## Recommendation

The proposed method is feasible as one bounded metadata investigation, provided
the frozen implementation establishes the contracts below before processing
observations. Use temporal RMS for the global coarse search, retain the selected
individual-frame cost checks, use the production mapper, and preserve original
valid rows in a final extension. Reduce only the candidate experiment's bounded
inverse scan from 125 to 100 to pay for the new alignment context. Report an
unchanged historical comparator and a separate strict-map comparator using that
same 100-frame scan. Checkpoint eligibility must increase from radius 50 to 75.

This tests a plausible explanation for rejection, not an established cause.
Round 25's 454 extra mapped rows produced only 50 extra usable queries; its
report explicitly leaves temporal global search as an untested hypothesis.
[SUBGRID_ALIGNMENT_REPORT.md:3-8,76-80,198-211](SUBGRID_ALIGNMENT_REPORT.md#L198-L211).
Earlier broadcast action heads also failed promotion, so improved alignment
coverage would establish an engineering opportunity only.
[BROADCAST_ACTION_REPORT.md:87-112](BROADCAST_ACTION_REPORT.md#L87-L112).

## The dependency repair is sufficient under explicit conditions

The following bounds are deductions from the existing three-anchor acceptance,
adjacent interpolation, bounded inverse selection and summary implementation;
they are not measurements from a new run.
[broadcast_alignment.py:115-159](broadcast_alignment.py#L115-L159),
[alignment_coverage.py:18-59,68-83](alignment_coverage.py#L18-L59),
[sports_action_features.py:71-91](sports_action_features.py#L71-L91).

| Decision or value | Required tactical observation context |
|---|---:|
| New coarse winner at anchor `a`, paired offsets -25/0/+25 | `a +/-25` |
| Acceptance of endpoint `a`, including neighbors' coarse triplets | `a +/-50` |
| Mapped row `f`, considering either adjacent segment and knot overwrite | `f +/-75` conservatively |
| All clip eligibility decisions using mapping only within `c +/-100` | `c +/-175` |
| Dense query, considering every eligible center within 25 | query `+/-200` |
| Final inclusive summary of radius 50 | feature row `+/-250` |

The endpoint bound depends on evaluating only the center's own fine search and
own extra offsets. Requiring neighboring fine searches or their extra gates
recursively would invalidate this proof. Existing fine evidence uses tactical
offsets -25/0/+25, and the proposed extra offsets -12/+12 lie inside that range.
[subgrid_alignment.py:62-75,112-135](subgrid_alignment.py#L62-L75).

The mapped-row bound includes missingness: an exact anchor can be supplied by
either neighboring segment, and the later segment can overwrite that row. Both
adjacent endpoints' acceptance decisions matter even when the final row is
missing. The production mapper also requires the same orientation, rate .8..1.2,
rounding and final broadcast usability.
[broadcast_alignment.py:137-161](broadcast_alignment.py#L137-L161).

Therefore `100 +75 +25 +50 =250` repairs the prior 275-frame conflict. Every
successful or failed inverse search must remain inside `c +/-100`; choosing a
short witness after searching an entire component would not satisfy the bound.
Keep scheduling's conservative `c +/-175` half/prefix/exclusion guards. Potential
image content has the smaller bound `100 +25 +50 =175`; no images are processed
in this investigation. The current scheduler illustrates the required local
component and absence logic with a 125-frame scan.
[alignment_coverage.py:18-59](alignment_coverage.py#L18-L59).

Global broadcast formation search remains a global broadcast-tracking dependency.
The 250-frame claim concerns the tactical timeline; it must never be described as
an all-source local-observation guarantee.
[broadcast_alignment.py:96-110](broadcast_alignment.py#L96-L110).

## Freeze the objective, candidate domain and failure behavior

Use the existing directed same-team geometry: for each team, each visible
broadcast player is compared with its closest tactical player, distances cap at
10 metres, team mean squared errors receive equal weight, and the result is RMS.
The source is usable only with supplied validity and at least four finite
players per team. These definitions are owned by
[broadcast_alignment.py:65-89](broadcast_alignment.py#L65-L89).

At tactical anchor `t` and broadcast center `b`, define the new coarse score as
`sqrt(mean(cost(t+d, b+d, orientation)^2))` for `d in {-25,0,25}`. Use complete
usable triplets only, the declared 25-frame broadcast grid and all four
reflections. Preserve the existing source-center bounds
`25 <= b < len(broadcast_tracking)-25`, if that is the chosen frozen metadata
domain. Do not silently replace it with a clip-center bound or change the
historical comparator. The original coarse and fine candidate bounds are in
[broadcast_alignment.py:96-110,122-133](broadcast_alignment.py#L96-L110).

Compute the global temporal argmin first, with reflection-major then ascending
source ordering matching NumPy's existing flattened first argmin. After choosing
that winner, require each of its three individual costs to be finite and <=2.5.
Failure rejects that coarse winner; it must not trigger a search for the next
winner satisfying the cutoff. The same ordering applies to the selected fine
winner. Individual checks prevent a good temporal mean from hiding one poor
frame, but they must not remove otherwise usable candidates from the search,
far-runner comparison or fine-span calculation. The existing source uses an
unfiltered argmin followed by acceptance gates.
[broadcast_alignment.py:103-110,129-133,137-142](broadcast_alignment.py#L103-L110).

For the temporal winner's ambiguity margin, evaluate the same temporal objective
over every complete usable grid candidate in every reflection satisfying
`abs(b - new_coarse_winner) >250`. The cutoff is strict; an alternative exactly
250 away is excluded. Do not reuse the original spatial winner's far mask or
Round 25's refined original-runner approximation. If the winning score is finite
and no finite far runner exists, a positive-infinite margin is valid; if no
finite winner exists, record unavailable rather than allowing `inf-inf` or a
fabricated first candidate. Freeze the zero-candidate behavior explicitly.
Original far-mask semantics are in
[broadcast_alignment.py:103-110](broadcast_alignment.py#L103-L110), and the
single-original-runner approximation being replaced is in
[subgrid_alignment.py:95-110](subgrid_alignment.py#L95-L110).

Require all three neighboring coarse winners to pass their cost, individual
offset and temporal margin >=.5 gates, use a common reflection, and have source
increments inclusive 0..50. Fine search is only the center winner's integer
`+/-25`, with the same complete paired triplet objective, first argmin, selected
individual costs <=2.5, aggregate cost <=2.5 and full best+.05 span <=10. The
span includes all qualifying scored candidates, including disconnected near
minima; it is not just the connected basin around the chosen winner. Retain the
center's own usable -12/+12 checks, each cost <=2.5. Do not reintroduce Round 25's
half-cell stage or add a fallback. Existing neighborhood and fine mechanics are
in [broadcast_alignment.py:115-133](broadcast_alignment.py#L115-L133) and
[subgrid_alignment.py:123-135](subgrid_alignment.py#L123-L135).

Temporal margin .5 is a changed ambiguity statistic whose calibration is not
established by retaining its numeric value. Per-offset geometry limits preserve
those local cost bounds, not equivalence to the historical spatial margin.
The declared far comparison also intentionally omits near alternatives and all
off-grid global alternatives; it is not an exhaustive fine-grid uniqueness
certificate. This is a limitation to report, not a reason to run extra variants.
The production margin's excluded neighborhood and coarse grid are explicit in
[broadcast_alignment.py:96-110](broadcast_alignment.py#L96-L110).

## Safety, checkpoints and source bounds

Before any new formation-cost call, require the tactical coarse anchor's full
`t +/-25` inside one official half and actual prefix, disjoint from the sealed
intervals and held-frame intervals expanded inclusively by 250. Neighbor coarse
decisions must each pass this guard. That gives accepted endpoints safe +/-50
context and mapped rows safe conservative +/-75 context without reading new
geometry in excluded tactical samples. The existing safe-interval helper has
the required inclusive exclusion semantics.
[subgrid_alignment.py:17-21,86-95](subgrid_alignment.py#L17-L21),
[games.py:197-208](../games.py#L197-L208).

The inherited checkpoint guard of 50 is insufficient for the new mapped-row
dependency. Use one common checkpoint universe with **radius 75** for baseline,
candidate and extension, residues `f %25 in {6,19}`, and the exact mapped source
frame/reflection without searching. New objective observations occur at tactical
residue 0; the extra gates use residues 12 and 13, leaving 6 and 19 unused by
matching and acceptance. Retain eligible, unmapped, mapped, usable and
unavailable-mapped denominators; new-only is the extension's newly mapped stratum.
Require >=50 usable new-only rows, availability >=.98 and bad fraction no more
than the baseline's plus .02. Zero mapped or usable new-only rows and an empty
baseline usable denominator fail. These remain correlated consistency checks,
not synchronization labels. The inherited checkpoint and gate definitions are
in [subgrid_alignment.py:147-182](subgrid_alignment.py#L147-L182).

Tracking and usable image bounds must stay distinct. `games.py` retains the
164488-frame source constant, but the verified video metadata contract requires
164487 images, valid indexes 0..164486. Schedules must use that latter bound and
require all 199 source frames from `b-100` through `b+98` usable, including the
99 intervening frames that are not sampled by the 100-frame model clip. A source
index 164487 fails even if tracking metadata contains it.
[games.py:78-89](../games.py#L78-L89),
[broadcast_action_sampling.py:61-64](broadcast_action_sampling.py#L61-L64),
[sports_action_sampling.py:19-20](sports_action_sampling.py#L19-L20),
[alignment_coverage.py:29-34](alignment_coverage.py#L29-L34).

## Comparators, gates and verification before closure

Reproduce the historical strict map and inverse-125 schedule exactly: 3359 mapped
public rows and 2328 usable public queries. Separately report that same strict
map through inverse-100, plus candidate and extension through inverse-100. The
primary is the row-wise extension preserving every original valid frame,
orientation, cost and margin exactly; retain separate anchor masks and never
manufacture a composite allowed-anchor graph.
[SUBGRID_ALIGNMENT_REPORT.md:70-80](SUBGRID_ALIGNMENT_REPORT.md#L70-L80),
[subgrid_alignment.py:139-144](subgrid_alignment.py#L139-L144).

Freeze primary query gates against the historical 2328-query mask: >=3000 total,
>=750 net additional and >=98% historical query retention. The conjunction makes
**3078 the effective total target**. Report paired strict-100 differences for
interpretation without substituting that comparator into the historical gates.
Exact row preservation does not itself preserve query coverage: joins can create
cuts. Reapply the local component rules to the actual extension.
[subgrid_alignment.py:173-182](subgrid_alignment.py#L173-L182),
[alignment_coverage.py:39-54,130-143](alignment_coverage.py#L39-L54).

Inverse-100 is synthetically feasible for an identity/rate-one map: its scan
contains the source clip's -100/+98 footprint. It need not preserve inverse-125
coverage: a uniform rate-.8 map requires up to 125 tactical frames to cover the
100-frame source reach. This follows directly from the fixed clip and inverse
endpoint requirements; it predicts a structural limitation, not a realized
competition-data result.
[sports_action_sampling.py:19-20](sports_action_sampling.py#L19-L20),
[alignment_coverage.py:36-59](alignment_coverage.py#L36-L59).

Before observations, meaningful synthetic checks should establish the new
winner/runner and individual-cost semantics, reflection and first-tie behavior,
exact-250 exclusion, absent candidates, unusable triplets, disconnected fine
span, guard equality, no excluded geometry, 75-frame mapping dependency,
100-frame bounded inverse missingness, union cuts and source-index limits.
Existing tests demonstrate how these boundary conditions are represented without
competition inputs.
[test_subgrid_alignment.py:43-81,96-109,143-162](test_subgrid_alignment.py#L43-L81),
[test_alignment_coverage.py:28-60,74-83](test_alignment_coverage.py#L28-L60).

For execution, independently reconstruct the new method from actual tactical
prefixes 18935 and 19060; compare public anchor/trace evidence, mappings and query
metadata with sufficient summary context, every public row of all 121 synthetic
feature columns, checkpoint rows and public aggregates. Rebuild rather than
truncate one completed mapping. Prefix-dependent whole-grid rejection counts
need their own deterministic replay, not an assertion that totals at different
prefix lengths are equal. Then require a fresh-process byte replay and an
independent source/input/cache/preservation audit. The prior execution exposed
this specific aggregate-prefix pitfall.
[prepare_subgrid_alignment.py:79-108,121-153](prepare_subgrid_alignment.py#L121-L153),
[SUBGRID_ALIGNMENT_REPORT.md:145-181](SUBGRID_ALIGNMENT_REPORT.md#L145-L181).

Freeze once, run at most this method, preserve failures, and close on the frozen
conjunction without retuning. Preserve all selected/reference artifacts and the
entire ledger. A positive metadata result authorizes no automatic image decoding,
action-head work, possession fitting, scoring or promotion. The existing later
promotion requirements remain intact, and this review makes no competition-score
or full-match accuracy claim.
[SUBGRID_ALIGNMENT_PLAN.json:49-53](SUBGRID_ALIGNMENT_PLAN.json#L49-L53).
