# Round 43: fixed hard-proposal projection rejected

One exact monotone alignment projected Round37's final predicted labels onto
selected8's predicted run order/count, separately in every live component and
decode block. Its integer objective minimized Hamming distance to the proposal,
then to the reference, with a fixed stay preference on ties. True labels never
entered projection. All reference D/X remained exact. Prior constrained boundary
methods are acknowledged; this test changes the proposal/evidence interface.
See [research](PROPOSAL_PROJECTION_RESEARCH_20261027.md) and
[prospective plan](TRANSCRIPT_PROJECTION_PLAN.json).

| Protocol | Selected8 | R37 proposal | Projection |
|---|---:|---:|---:|
| Buffered minutes | .741176 | .754286 | .752941 |
| Purged spells | .685714 | .720000 | .697143 |
| Buffered 100 seconds | .670270 | .691892 | .691892 |

Projection improves all three aggregate scores over selected8 and removes the
proposal's extra buffered spurious possession. However, it does not exceed the
historical .754286 record; buffered and purged mean-fold primary also regress.
The frozen conjunction fails. No cost, tie, radius, emission or endpoint sweep,
confirmation fit, full candidate or promotion follows.

All 13 R37 checkpoints reload with exact probabilities and final proposal labels.
Zero new fits occur. Saved-model/shared/official score and gate replays are exact.
Eight synthetic tests include exhaustive small-sequence optimum checks.
Independent source/spec review finds no actionable issue. All 17 selected
artifact hashes and source links remain unchanged. Nine rows append while
preserving all previous 1,546 rows/bytes (1,555 total). Selected8 stays unchanged.

Evidence: ignored `data/control_transcript_projection_v1/`. Reproduce:

```sh
.venv/bin/python experiments/research_transcript_projection.py --replay --output data/control_transcript_projection_v1/review_replay
```

These are correlated adaptive public development checks on 17 possessions,
not independent/private accuracy or an established optimum. The requested
research continues through a separately designed hypothesis.
