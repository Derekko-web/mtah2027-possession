# Round 24: strict formation anchor failure diagnosis

Completed 2026-10-23. The diagnostic investigation is complete. Geometry filters
and their neighborhood requirements dominate rejection under the frozen reporting
order; unavailable tactical observations do not explain the public-window losses.
The original alignment and thresholds remain unchanged. No images, action
outputs, possession targets, fits, scores or candidate predictions were processed.
The selected eight-model candidate and all 1,468 ledger rows remain unchanged. This diagnoses executed rules, not synchronization truth.

## What removes coverage

The exact production replay recovers the same 3,359 mapped and 4,141 unmapped
public frames. All 300 public-window tactical anchor observations are usable,
and all 7,500 public tactical frame observations meet the original source-validity
rule. Missing public mappings are therefore not explained by unavailable tactical
players or homography in this time band.

| Final frame-level outcome | Public frames |
|---|---:|
| Mapped under the original strict algorithm | 3,359 |
| No incident interval with two allowed endpoints | 3,791 |
| Allowed, common-orientation endpoints fail the fine-time rate check | 350 |
| Other final rejection stages | 0 |
| Total | 7,500 |

The endpoint category accounts for 91.55% of the 4,141 missing frames, and the
fine-rate category for 8.45%. The frame classification considers every incident
inclusive interpolation interval, including both sides of shared knots. It does
not assign a missing frame to only its nearest anchor. Of the 301 intervals
intersecting the public time band, 153 fail endpoint acceptance, 14 fail fine-time
rate and 134 are accepted. Intervals and frames are dependent units, not separate
experimental observations.

## Why anchors are rejected

Of 300 anchors in the fixed public time band, 162 are allowed and 138 rejected.
The ordered categories, frozen before processing, partition all 300:

| First failed anchor stage | Anchors |
|---|---:|
| Coarse neighborhood geometry above 3.5 | 59 |
| Coarse neighborhood far margin below .25 | 15 |
| Coarse neighborhood time increment outside 0..50 | 5 |
| Fine geometry above strict 2.5 | 14 |
| Fine uncertainty span above strict 10 | 8 |
| Coarse neighborhood geometry above strict 2.5 after fine passes | 35 |
| Coarse neighborhood far margin below strict .5 after prior checks | 2 |
| Allowed | 162 |
| All other first-failure categories | 0 |

The coarse prefilter prevents 79 fine searches. All 221 attempted fine searches
produce finite results; no attempted fine search fails because its three source
comparisons are unavailable. The subsequent strict gates reject 59 of those 221.
Geometry categories account for 108/138 ordered rejections, far ambiguity for 17,
fine span for 8 and the coarse temporal increment for 5. These groupings describe
executed gate precedence, not unique physical causes or safely recoverable matches.

Many anchors fail more than one predicate. Among all 300 finite neighborhoods,
59 violate permissive geometry, 56 permissive far margin, 27 orientation agreement
and 48 the coarse increment check. Strict neighborhood geometry fails in 123/300
and strict neighborhood margin in 78/300. For example, zero *first* orientation
rejections does not mean all neighborhoods have stable orientation: the 27
orientation failures already encounter an earlier failed gate. Fine strict span
fails in 9/221 raw cases but is first in 8 because one also fails fine geometry.
The saved overlapping counters explicitly distinguish evaluated predicates from
undefined/unattempted later stages.

For unmapped frames, overlapping endpoint-reason counts include 1,683 touching
permissive coarse geometry rejection, 1,236 touching strict neighbor geometry
rejection, 584 touching fine geometry, 505 permissive far ambiguity, 327 fine span,
175 coarse increments and 76 strict margin. These unions are counted once per
reason per frame, but different reasons overlap and must not be added as unique
coverage losses or predicted recoveries.

The global broadcast coarse grid has 3,959 usable candidates among 6,578 sampled
positions. Its source-unavailability flags overlap: 2,248 fail supplied homography
validity, 2,524 lack four finite team-0 players, and 2,539 lack four finite team-1
players. Thus some broadcast observations are unavailable, but every public
coarse query still finds a finite winning candidate. This does not show that the
correct corresponding broadcast moment is present or usable.

## Additional observation consistency

The protocol inspects only the existing coarse winner and existing finite fine
winner. Fixed extra offsets -12/+12 were excluded from the original fine objective,
which uses -25/0/+25. Geometry diagnostics require the full +/-50 tactical context
to satisfy half/prefix guards and the 24 sealed/60 held-frame exclusions expanded
by 250. Three public anchors are skipped by that context guard. No diagnostic
changes a winner or mapping.

| Existing fine candidates | Count | Median directed residual | Median worst residual at extra offsets | Extra-offset residual >2.5 |
|---|---:|---:|---:|---:|
| Strict allowed | 162 | .427 | .540 | 1 / 162 |
| Strict rejected | 59 | 2.274 | 2.360 | 24 / 59 |

These correlated internal checks favor the accepted group descriptively. They
are not independent validation, correspondence labels or proof of possession
accuracy. The existing 2.5 threshold is used only to summarize a fixed diagnostic,
not as a new acceptance gate.

There are 18 fine-eligible anchors that pass their own fine geometry/span and own
coarse cost but fail a neighboring strict coarse-cost requirement. Some may fail
other predicates too; they are not 18 verified recoverable anchors. One additional
anchor passes the other strict checks and its own margin but is vetoed by a
neighbor's margin. These provide a concrete lead about coarse neighborhood and
temporal-grid sensitivity rather than a reason to relax thresholds.

Reciprocal geometry cannot inherit the directed threshold without calibration.
For accepted fine candidates, the median reverse residual is 4.119 versus .427
forward; 154/162 exceed 2.5 in reverse. The two cameras see different player sets,
so reverse nearest-neighbor costs have different visibility calibration. Neither
that disparity nor a balanced RMS is evidence that the accepted alignment is
wrong. The reciprocal diagnostic was not used to change acceptance.

## Frozen method and faithful production replay

[ANCHOR_FAILURE_PLAN.json](ANCHOR_FAILURE_PLAN.json) was frozen before diagnostic
counts, SHA256
`1c4e17700dd0cfd2aafa84bfdb9a886a3440a7be45c80d780369db109c0a3d24`.
The source-based design review is
[ANCHOR_FAILURE_DESIGN_RESEARCH.md](ANCHOR_FAILURE_DESIGN_RESEARCH.md).

The tracer preserves the original joint-profile anchor construction. Fine search
uses the original permissive neighborhood prefilter (3.5/.25), then strict
acceptance uses 2.5/.5/span10. Using the strict profile alone to recreate missing
fine rows would change the computation and misattribute rejection. Positive-infinite far margins remain valid, the runner-up exclusion remains strictly
>250 broadcast frames across orientations, and shared-knot overwrite order is
unchanged. The final canonical mapper checks broadcast usability; this diagnostic
does not silently introduce a tactical-interior validity gate. In this public
window, no mapped row happens to have an unusable tactical observation anyway.
Fine uncertainty is the full span of candidates within best+.05, not a confidence
interval or calibrated probability.

The first-failure order in the frozen plan is authoritative. Raw overlapping
predicates and denominators are retained separately, as are the OR over incident
intervals and each endpoint's reason union. No attribution is inferred from only
the interpolated mapping margin.

The implementation is [anchor_failures.py](anchor_failures.py),
[prepare_anchor_failures.py](prepare_anchor_failures.py),
[audit_anchor_failures.py](audit_anchor_failures.py) and
[test_anchor_failures.py](test_anchor_failures.py). The original production
`broadcast_alignment.py` is untouched. Only cached positions, team membership and
validity are consumed; ball/speed arrays, action heads, images and possession
labels are not loaded.

The original coarse and fine arrays and every field of the strict mapping replay
exactly. Actual tracer computations from 18,935 and 19,060 tactical observation
prefixes agree with the inherited independently recomputed mappings. Anchor
reasons, conditional predicates, interval/frame attribution and extra consistency
rows agree across both prefixes over the 7,500 public rows plus 50 context. This
is diagnostic metadata parity; it does not certify any new actual action features.
The underlying broadcast formation search remains global. The prior Round 23
qualification of image versus complete tactical dependency bounds remains in
force; no new feature pipeline is introduced here.

## Verification, preservation and reproduction

A separate process/output generation reproduces all 17 diagnostic artifacts,
including manifests, byte-for-byte. The ordinary audit reconciles 300 anchor and
7,500 frame denominators, all conditional overlap counters and 1,036 guarded
consistency rows across both prefixes. All 202 production tests pass, including
10 new synthetic tests for exact original replay, infinite margins, shared knots,
orientation/rate stages, broadcast-only validity, exclusions and prefix stability.

All 17 protected candidate/reference hashes and every byte of the 1,468-row ledger
remain exact. The selected CSV is still
`data/candidate_motion_dead_trim/candidate.csv`, SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
There are 0 images decoded, new neural calls, action outputs consumed, possession
targets read, model fits, possession-score rows, confirmation runs or full-match
inferences. All owned process handles are terminal at closeout. Source snapshots,
plans, per-anchor/frame attribution and provenance stay under ignored
`data/control_anchor_failures_v1/`; only aggregate results and source code are
included in this report and hosted context.

Ordinary audit, safe to rerun without overwriting stage outputs:

```sh
.venv/bin/python experiments/audit_anchor_failures.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -q
```

Original generation sequence, which intentionally requires fresh output folders:

```sh
.venv/bin/python experiments/prepare_anchor_failures.py
.venv/bin/python experiments/prepare_anchor_failures.py --output data/control_anchor_failures_v1/assessment_replay
.venv/bin/python experiments/audit_anchor_failures.py
```

Evidence includes `assessment/summary.json`, both assessment manifests,
`integrity.json`, `production_tests.log`, `execution_integrity.json` and the round
stage/closeout files. All original promotion requirements remain unchanged. No
alignment or model is promoted by a diagnostic result.

## Next question and limits

A useful next protocol would test whether sub-grid refinement of existing coarse
winners before neighborhood rejection improves observation consistency under
unchanged strict geometry thresholds and additional temporal/ambiguity checks.
This follows from the coarse/neighbor veto counts and is a hypothesis, not a
validated fix. It should be frozen separately and must retain locality, missingness,
exclusion, prefix and eventual possession-promotion requirements. It has not been
started; no rejected threshold or previous model was silently retuned.

No supplied correspondence ground truth exists, so these diagnostics cannot
separate valid pairs rejected by noisy tracking from incorrect pairs correctly
rejected. Public possession experiments remain correlated, extensively adaptive
single-match development on 17 labelled possessions. Private/full-match accuracy
and maximum score remain unverified, and the broader optimization objective is
unfinished. No manual labels, private/corrected ground truth, commits, uploads, submissions,
publication or external contact occurred.
