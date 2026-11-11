# Round 47: fixed three-seed latent pool rejected

A fixed equal probability pool of three independently seeded versions of the
pretrained166 head was tested under the unchanged selected8 decoder and
live-only overlay. The screen used seeds42,142,242: 13 immutable R37 checkpoints
were reused and 26 new heads fitted on the same training folds. Individual
members were not scored or selected. See the
[frozen plan](LATENT_SEED_ENSEMBLE_PLAN.json).

| Protocol | Selected8 | Three-seed latent pool |
|---|---:|---:|
| Buffered minutes | .741176 | .754286 |
| Purged spells | .685714 | .720000 |
| Buffered 100 seconds | .670270 | .681081 |

The pool ties the historical buffered record rather than exceeding it.
Its frozen promotion conjunction fails, so there are no confirmation fits,
full-match candidate or selected-model change. No seed, member count, pooling
weight, feature, support or decoder search follows.

All 39 checkpoints, saved probabilities, training records, shared predictions,
official scores and gates replay exactly. Unsupported observations explicitly
retain original slot0 probabilities; D/X labels stay exact. Six score rows
append while preserving the previous 1,570 rows/bytes (1,576 total). The
17 selected artifact hashes and source links stay unchanged.

Evidence: ignored `data/control_latent_seed_ensemble_v1/`. This is adaptive,
correlated public development evidence on 17 possessions. It establishes no
private accuracy or global optimum. The current default remains selected8,
and the highest comparable buffered development score remains .754286.
