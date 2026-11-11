# Round 26: temporal global formation matching — completed, opportunity rejected

Completed 2026-10-23. One frozen metadata investigation is complete. Safe public
query coverage changes from **2,328 to 1,785**
(-543), with 76.68% historical
query retention. The frozen conjunctive gate is **rejected_metadata_opportunity**.
The selected model and all 1,468 ledger rows remain byte-exact. No images,
action outputs, possession targets, fits or scores were used; no alignment or
model is promoted.

The authoritative [plan](TEMPORAL_GLOBAL_PLAN.json) SHA256 is
`27af3500086bd1eb8c47b019ee045f570ba21cf3fb7b77b8968185ce1d1a8933`. The [design review](TEMPORAL_GLOBAL_DESIGN_RESEARCH.md) and preceding
[advisory review](TEMPORAL_GLOBAL_PLAN_REVIEW.md) distinguish this engineering
hypothesis from possession improvement. Detailed arrays, schedules and logs
remain local under ignored `data/control_temporal_global_v1/`.

## Frozen method and dependency preflight

Each global 25-frame broadcast-grid candidate is scored by RMS of the existing
directed, same-team capped-10-metre costs at paired tactical/broadcast offsets
-25/0/+25. All triplet observations must be usable. All four reflections are
searched; first reflection-major/earliest-source argmin wins. No geometry cutoff
filters the candidate universe before selection. The far runner is the minimum
of this same temporal objective strictly more than 250 broadcast frames from
the new temporal winner across all reflections. An absent far runner has a
valid infinite margin; an unavailable winner stays missing.

All three neighboring coarse winners must have temporal RMS <=2.5, every selected
individual residual <=2.5, temporal margin >=.5, equal reflection and increments
in 0..50. Own fine search uses integer source centers within +/-25, the same
triplet objective, first minimum, every individual residual <=2.5, and full
best+.05 span <=10 including disconnected minima. Own paired offsets -12/+12
must both be usable with residual <=2.5. These are acceptance observations,
not unused checkpoints. No half-cell refinement, second-choice fallback or
recursive neighbor fine/extra checks are used.

The mapper remains production adjacent-endpoint interpolation, reflection/rate
.8..1.2 guards, original rounding/knot overwrite and final source usability.
The primary extension copies all original valid frame/reflection/cost/margin
fields byte-for-byte and fills only original missing rows. Baseline and candidate
anchor masks remain separate; inserted rows cannot force bridges across cuts.

Temporal coarse context +/-25 makes endpoint acceptance depend on +/-50 and
mapped values or absence conservatively on +/-75. Inverse search is reduced to
**+/-100**, giving **100+75+25+50=250** for complete tactical dependencies,
including failed searches, missingness and future summaries. Potential image
content is bounded by 175; no images were processed. Schedule decisions use
c+/-175 half/prefix/exclusion guards. All 199 intervening source indices of each
100-frame stride-two clip must be usable and within the actual 164,487-image
video, independently of the 164,488-row tracking domain. Source components break
at missing rows, reflection changes, decreases or jumps >2. Global broadcast
formation search remains global; 250 is a tactical-axis bound.

All new coarse geometry is guarded before evaluation against the 24 reserved
windows and 60 held images with inclusive +/-250 exclusions. Checkpoints use a
common +/-75 guard and unused residues 6/19 modulo25, without source search.
Only cached positions, team membership and validity are loaded. Original anchors
are hash-verified inherited independent-prefix reconstructions; this round does
not recompute new geometry at excluded original anchor locations.

Preflight passed 23 meaningful synthetic tests, including a false spatial winner
replaced by the temporal objective, far duplicates/reflections/ties, selected
individual residuals, fine span, guards, missingness, dependency perturbations,
prefix reconstruction and all 121 future-looking synthetic feature columns.
The inverse100 choice is feasible at rate 1 but can reject slow-rate clips that
inverse125 supports. Historical strict125 and paired strict100 are both retained.
Before observations, independent source review prompted a stronger exhaustive
global winner/runner audit; its earlier source snapshot and first freeze remain
preserved. No protocol, matching method, gate or observed result changed.

## Coverage and consistency results

| Mapping / schedule | Mapped public rows | Safe clip associations | Safe public queries | Net vs historical | Historical retention |
|---|---:|---:|---:|---:|---:|
| Historical strict / inverse125 | 3,359 | 90 | 2,328 | +0 | 100.00% |
| Paired strict / inverse100 | 3,359 | 59 | 1,785 | -543 | 76.68% |
| Temporal candidate / inverse100 | 3,009 | 55 | 1,609 | -719 | 69.12% |
| Primary extension / inverse100 | 3,359 | 59 | 1,785 | -543 | 76.68% |

Associations are whole-prefix schedule totals, not independent examples.
The original strict mapping reproduces 3,359 rows and historical inverse125
reproduces all 2,328 queries exactly. Paired strict100 costs
543 public query frames relative to historical scheduling.
The primary adds +0 net queries over that paired
comparator; gates still use the historical mask. It adds
0 mapped rows while preserving every old field.
The temporal search changes 103 of
299 comparable public coarse winners.
The candidate adds no missing public mapping rows when the reported extension gain is zero.
Those winner changes are not verified corrections or proof that spatial aliasing caused
the original rejection. Candidate-only mapping differs from the old source frame
or reflection on 0 shared mapped public rows.

| Candidate anchor first failure | Public anchors |
|---|---:|
| allowed | 134 |
| fine_span | 6 |
| neighbor_individual_cost | 27 |
| neighbor_temporal_cost | 130 |
| unavailable_neighbor | 2 |
| unsafe_context | 1 |

These categories partition 300 anchors under the frozen algorithmic order;
they are not unique physical causes or estimates of recoverable frames.

| Checkpoint group | Eligible rows | Mapped | Usable | Cost >2.5 | Median directed cost |
|---|---:|---:|---:|---:|---:|
| baseline | 591 | 268 | 268 | 0 | 0.392 |
| candidate | 591 | 240 | 240 | 0 | 0.376 |
| extension | 591 | 268 | 268 | 0 | 0.392 |
| new_only | 0 | 0 | 0 | 0 | undefined |

New-only source availability is undefined;
its above-cutoff fraction is undefined, compared
with baseline 0.00%. These are correlated
same-source consistency diagnostics, not synchronization labels.

## Frozen gate and stopping decision

The primary requires >=3,000 total queries AND >=750 net over historical 2,328
(effective minimum 3,078), >=98% historical retention, >=50 usable new-only
checkpoints, >=98% new source availability, and new bad fraction no more than
baseline+.02, plus all preservation/safety/prefix/replay requirements.

| Condition | Result |
|---|---|
| total | Fail |
| net_added | Fail |
| retained | Fail |
| checkpoint_count | Fail |
| checkpoint_availability | Fail |
| checkpoint_geometry | Fail |
| preservation_safety_prefix_replay | Pass |

The conjunction is not satisfied. This round closes
as **rejected_metadata_opportunity**, with no fallback, threshold change, new variant or automatic
downstream work. A positive metadata result would still establish opportunity
only. All inherited model-promotion requirements remain unchanged.

## Replay, audit and preservation

- **242 production unittest tests pass** (23 new synthetic contracts).
- Both actual tactical prefixes 18,935/19,060 independently reconstruct the new
  global searches, fine decisions, mapping and extension. Anchor/trace/mapping
  parity includes 75 context; query parity includes 50; all 7,500 public rows and
  all 121 synthetic feature columns agree exactly for all four comparisons.
- Whole-prefix schedule/rejection totals are retained and independently recreated;
  only defined public metrics are equated across prefixes.
- Separate fresh-process generation reproduces **45 artifacts byte-for-byte**.
- The auditor independently reconstructs the full global objective matrix and
  far minimum for 598 safe coarse searches,
  then verifies fine predicates, guards, schedules, queries, checkpoints and features.
- All 303 frozen source files, input/cache identities, 17 protected artifacts,
  six underlying source files and nine inherited caches verify. Both source
  symlinks remain intact. All 1,468 ledger rows preserve every preceding byte.
- Selected CSV SHA256 remains
  `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
- All owned executions have terminal exit0. No execution failure was discarded.
  Zero new images, neural calls, actual action outputs, possession targets, fits,
  score rows, confirmation runs, full-match inference, commits or external actions.

Reproduce with a fresh output directory, without overwriting retained evidence:

```sh
.venv/bin/python experiments/prepare_temporal_global_alignment.py --output data/control_temporal_global_v1/new_replay
.venv/bin/python experiments/audit_temporal_global_alignment.py
```

The ordinary auditor compares retained assessment/assessment_replay. Underlying
source identity verification is documented by source_provenance_audit.py and
source_provenance.json inside the ignored round directory. The source snapshots,
freeze records, execution_integrity.json, integrity.json, round_stage.json and
round_closeout.json retain the reproduction chain.

The temporal margin is an uncalibrated changed statistic; per-offset gates do
not make it equivalent to the original spatial margin. Near alternatives inside
250 and off-grid global alternatives remain outside the declared ambiguity test.
Metadata consistency and reproducibility do not establish correct synchronization,
possession improvement, private/full-match accuracy or maximum score. The broader
optimization objective remains unfinished. No next round has started.
