# Round 44: randomized trees with transferred features rejected

The fixed historical ExtraTrees settings were applied to ordered102 and
pretrained166 features, with train-only median imputation and missingness
indicators. Each replaced only selected8 slot0 before unchanged decoding and
the live-only overlay. Parallel tree fitting and serial probability reduction
retain exact checkpoint prediction replay. See the
[prospective plan](EXTRATREES_LATENT_PLAN.json).

The method follows the variance-reduction motivation in
[Geurts, Ernst and Wehenkel's original paper](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf).
ExtraTrees itself was already tested on early compact/geometry features;
this test concerns the later representation and selected8 interface.

| Protocol | Selected8 | Original102 ExtraTrees | Pretrained166 ExtraTrees |
|---|---:|---:|---:|
| Buffered minutes | .741176 | .688889 | .666667 |
| Purged spells | .685714 | .655556 | .658824 |
| Buffered 100 seconds | .670270 | .681081 | .472727 |

Neither arm passes the frozen conjunction or beats the .754286 record.
The pretrained arm loses short matches, including all three in the longer
protocol. Both are rejected without a parameter, feature or inference sweep.
No confirmation, full candidate or promotion follows.

All 26 models/imputers reload with exact probabilities, and saved-model/shared/
official scores and gates replay exactly. Three synthetic tests verify train-only
imputation and checkpoint replay. All 17 selected artifact hashes/source links
stay exact. Nine rows append, preserving all previous 1,555 rows/bytes (1,564
total). Selected8 remains unchanged. Evidence is under ignored
`data/control_extratrees_latent_v1/`.

These adaptively reused 17 public possessions establish no private accuracy or
maximum score. The broader research continues through independent technical
feasibility and a separately frozen nested ownership-learning hypothesis.
