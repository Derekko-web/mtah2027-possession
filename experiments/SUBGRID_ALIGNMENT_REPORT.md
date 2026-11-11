# Round 25: sub-grid formation refinement — completed, opportunity rejected

Prepared 2026-10-23. One frozen metadata experiment is complete. Refining the
original coarse winners adds 454 mapped public frames, but only **50 safe query
frames**, increasing source coverage from 2,328 to 2,378 of 7,500. This fails the
required 750-frame net gain and 3,000-frame total. The 36 usable newly mapped
checkpoints also fall below the required 50. No images, action outputs,
possession targets, fitting or scoring were used. No alignment or model is
promoted; the selected artifact and every ledger byte remain unchanged.

The authoritative protocol is [SUBGRID_ALIGNMENT_PLAN.json](SUBGRID_ALIGNMENT_PLAN.json),
SHA256 `8e1834ec0640f204d1085e2b19c2c5806d9be8281ff3a6f16f0594f8f9cdccf6`.
It was frozen before new observation geometry. Primary-source design research is
in [SUBGRID_ALIGNMENT_DESIGN_RESEARCH.md](SUBGRID_ALIGNMENT_DESIGN_RESEARCH.md).
Detailed arrays, source identities, checkpoints and failure evidence remain
local under ignored `data/control_subgrid_alignment_v1/`; only aggregates are
reported here.

## Hypothesis and fixed method

Round 24 found geometry and neighborhood rejection dominant, including 18
fine-eligible anchors passing their own coarse cost but receiving a neighbor
cost veto. Those were possible investigation targets, not verified recoveries.
The hypothesis was that a 25-frame lattice could give a poor individual
coarse cost even where a nearby sub-frame correspondence would pass the
existing strict predicates. See [ANCHOR_FAILURE_REPORT.md](ANCHOR_FAILURE_REPORT.md).

The new method independently searches integer source offsets -12..+12 around
each original global coarse winner, keeping its reflection and the original
directed, same-team capped-10-metre RMS geometry. It separately refines the
original far-coarse runner at that runner's original reflection. The effective
margin is the minimum of the original global margin and the refined-original-
runner margin. This is an additional check against one original alternative;
it is not an exhaustive search for every sub-grid alias.

Every three-anchor neighborhood must pass refined costs <=2.5, effective margins
>=.5, half-cell best+.05 spans <=10, common reflection and refined source
increments in 0..50. Fine search then uses refined center +/-25, paired tactical
and source offsets -25/0/+25, original RMS and cost/span limits 2.5/10. Both
additional paired offsets -12/+12 at the center's own fine winner must be usable
and have cost <=2.5. These offsets are acceptance evidence. Neighbor fine or
extra-offset checks are never recursively required. The production alignment
module remains unchanged. Implementation:
[subgrid_alignment.py](subgrid_alignment.py).

New coarse geometry is evaluated only where t+/-25 is inside one official half
and actual tactical prefix, disjoint from all sealed/held intervals expanded
by 250. Original baseline anchors are reused from hash-verified Round 24 actual
prefix reconstructions; production mapping is rebuilt and all fields compared
to inherited independently reconstructed strict maps. At every newly safe
coarse anchor, the original global four-reflection search is recomputed to
recover the runner identity and verify the original coarse row exactly.
No new geometry is evaluated at excluded tactical samples. This guarded replay
does not claim a new full-prefix reconstruction of original coarse arrays at
excluded locations.

The predeclared primary is a row-wise extension: copy every original valid
frame, reflection, cost and margin exactly; use the refined candidate only for
originally missing rows. Baseline and candidate anchor masks stay separate;
there is no composite anchor graph. The unchanged bounded inverse-25 sampler
still rejects discontinuous joins and requires all intervening source frames
usable. Complete tactical dependency remains <=250 (125 inverse scan +50
alignment context +25 query +50 summary); potential image content is <=200.
Broadcast formation search remains global. See
[alignment_coverage.py](alignment_coverage.py) and
[prepare_subgrid_alignment.py](prepare_subgrid_alignment.py).

## Results

| Mapping | Mapped public frames | Safe clip associations | Safe public query frames | Net query gain | Original query retention |
|---|---:|---:|---:|---:|---:|
| Unchanged strict baseline | 3,359 | 90 | 2,328 | 0 | 100% |
| Refined candidate alone | 3,737 | 92 | 2,378 | 50 | 100% |
| Primary extension | 3,813 | 92 | 2,378 | 50 | 100% |

The extension preserves all old per-frame fields bit-for-bit and adds 454 mapped
rows. The standalone candidate omits 76 old mapped rows; where both map, its
source frame and reflection do not change. Preservation therefore follows the
explicit extension rule, not a claim that recentering preserves the old
interpolation graph.

Of 300 public anchors, 299 have safe new coarse context. Ordered candidate
first-failure counts are:

| Stage | Anchors |
|---|---:|
| Allowed | 176 |
| Neighbor refined geometry | 86 |
| Neighbor half-cell span | 13 |
| Neighbor effective margin | 10 |
| Own extra-offset geometry | 5 |
| Refined coarse increment | 4 |
| Fine span | 2 |
| Unavailable neighbor | 2 |
| Fine geometry | 1 |
| Unsafe context | 1 |

These are ordered algorithmic categories, not unique physical causes. No public
half-cell span is censored by the source-center bounds. The original strict
baseline's 162 allowed anchors remain unchanged.

Most additional mapped rows still do not support a complete clip. Over the fixed
319-center schedule grid, the extension has 122 unmapped-center rejections
versus 144 originally, but 61 inverse-component rejections versus 44 and 12
source-unusable rejections versus 9. The resulting eligible count rises only
90 to 92. Both have 18 boundary/prefix and 14 sealed/held rejections. These are
aggregate first-failure counts, not a paired causal allocation of new rows.

## Unused checkpoint gate

Checkpoint tactical residues are fixed at f mod 25 in {6,19}, distinct from the
matching and extra-gate residues. Every comparison uses the exact rounded mapped
source frame and reflection, with no additional search. The full +/-50 tactical
dependency is guarded. These are correlated same-source consistency checks,
not synchronization labels or independent validation.

| Group | Eligible checkpoint rows | Mapped / usable | Cost >2.5 | Median directed cost |
|---|---:|---:|---:|---:|
| Baseline | 595 | 268 / 268 | 0 | 0.392 |
| Candidate | 595 | 298 / 298 | 0 | 0.420 |
| Extension | 595 | 304 / 304 | 0 | 0.421 |
| Newly filled extension rows | 36 | 36 / 36 | 0 | 2.156 |

The new-only group is a stratum of the common checkpoint universe, so its
eligible count denotes newly mapped checkpoint rows. All mapped comparisons
have usable source observations. Its above-cutoff fraction is 0%, satisfying
the baseline-plus-.02 rule, and source availability is 100%, satisfying .98.
However, only 36 usable new comparisons exist, below the required 50. A zero
above-cutoff count does not establish correct synchronization; the new group's
median residual is also substantially higher than the baseline's.

| Frozen primary condition | Requirement | Observed | Pass |
|---|---:|---:|---|
| Safe public queries | >=3,000 | 2,378 | No |
| Net gain over strict inverse-25 | >=750 | 50 | No |
| Old query retention | >=98% | 100% | Yes |
| Newly mapped usable checkpoints | >=50 | 36 | No |
| New source availability | >=98% | 100% | Yes |
| New checkpoint bad fraction | <=baseline +.02 | 0% <=2% | Yes |
| Old mapping fields, safety, prefix and replay | Exact / pass | Exact / pass | Yes |

The conjunction fails. There is no fallback variant, threshold adjustment,
image processing or downstream action-model trial in this round.

## Reproduction, execution repair and preservation

Both actual tactical source prefixes, 18,935 and 19,060, independently reconstruct
safe original coarse searches, runner identities, refinement, fine searches,
candidate mapping and extension. All anchor/trace and mapping/query comparisons
pass on the public interval plus 50 frames of context. The complete 121-column
synthetic feature views agree exactly on every public row for all three
mappings. No real action-feature parity is claimed.

The first assessment stopped at an overly broad aggregate-prefix assertion.
Five whole-grid candidates beyond the shorter source prefix change their
first-failure label from prefix boundary to sealed/held when the prefix grows:
boundary counts 23->18 and sealed counts 9->14. Every public statistic,
mapping/query comparison, checkpoint and synthetic feature already agreed.
The execution repair compares every public aggregate exactly, retains both
whole-grid rejection tables, and independently regenerates each table against
its own prefix. It changes no plan, geometry, guard, schedule, threshold,
mapping or opportunity gate. The plan hash and matching-module hash remain
unchanged. Original logs, source archives, minimal red/green reproduction and
repair records are preserved in `execution_failure_01/`; all 34 preexisting
non-dispatch artifacts remain byte-identical after repair.

Final validation:

- **219 production unittest tests pass**, including 17 new refinement and
  aggregate-prefix regressions.
- Separate fresh-process generation reproduces **all 37 artifacts byte-for-byte**.
- Source/input/cache hashes, every refined predicate, schedule bounds, exclusions,
  source usability, checkpoint denominators and actual-prefix parity pass
  [audit_subgrid_alignment.py](audit_subgrid_alignment.py).
- All **17 protected selected/reference artifacts** retain their hashes.
- The entire **1,468-row ledger** remains byte-identical: 778 inherited and 690
  continuation comparisons; this round adds zero.
- Selected CSV SHA256 remains
  `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
- No new images, neural calls, actual action outputs, possession targets, fits,
  scores, confirmation runs, full-match inference, commits or external actions.

Reproduce locally, with a fresh output directory:

```sh
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -q
.venv/bin/python experiments/prepare_subgrid_alignment.py --output data/control_subgrid_alignment_v1/new_replay
.venv/bin/python experiments/audit_subgrid_alignment.py
```

The auditor compares the retained `assessment` and `assessment_replay` directories;
the example `new_replay` is additional evidence and cannot overwrite either.
Authoritative ignored evidence is `integrity.json`, both assessment manifests,
`execution_integrity.json`, `round_stage.json` and `round_closeout.json` under the
new round directory. Successful assessment, fresh replay, tests and audit all
have verified terminal exit 0; the original failure is retained separately.

## Next hypothesis, not started

Refining a fixed original winner improves geometry but cannot correct selection
of the wrong global moment. A separately frozen future investigation could test
whether a three-frame temporal objective during the global coarse search
reduces spatial aliases and neighbor rejection before choosing the winner.
Current counts motivate that question without proving an alias is the cause.
Do not retune this rejected method or begin that investigation automatically.

The existing substantial-gain, short-possession, macro/false-D/D-recall,
protocol/seed, actual-feature replay and independent-review promotion gates
remain unchanged. This metadata experiment establishes no possession accuracy
gain, private/full-match accuracy or maximum score. The broader optimization
objective remains unfinished.
