# Trained contact-history ownership heads

Neither trained contact head improves selected8. No finalist advances and the
selected full-match candidate remains byte-identical. The frozen plan is
[CONTACT_HEADS_PLAN.json](CONTACT_HEADS_PLAN.json), SHA256 `3ee86c099128cb1145232303669760f9b7f08f042d35035f66b6ecf8ba73309d`.
This is a separate supervised hypothesis after the fixed color-contact proposal
experiment; its failures remain recorded rather than relabeled successful.

Both heads extend the same170-column ordered/broadcast joint model. The metadata
head adds27features; the color head adds81features, including metadata and both
tracked/detected-ball color scores. Fixed past/future25/75/125means and finite
fractions retain missingness. Only adjacent finite5-frame contact samples are
interpolated, within each half. There is no temporal player-slot interpolation,
new media extraction, detector inference, weak-color training or manual labeling.
The155-frame maximum observation span fits the existing250-frame target buffer.
Support is checked before labels; source preparation reads no possession targets.

The unchanged frame-objective XGBoost depth3/160trees, seed42 and team-symmetric
fitting/prediction produce26new checkpoints over13folds. Targets/weights and exact
training/validation indices are saved. The existing170-column broadcast joint
model supplies13verified matched-control checkpoints, without new control fits.
Three decoders use the new head in the joint slot, replace conditional live
ownership with referenceD exact, or trim referenceD only from the joint-slot
alternative. This avoids changing the raw-model slot for a broadcast-joint head.

| Variant / decoder | Buffered spellF1 | Purged spellF1 | Longer-block spellF1 |
|---|---:|---:|---:|
| Selected8 | .741176 | .685714 | .670270 |
| Matched control, live replacement | .588235 | .533333 | .482353 |
| Metadata, live replacement | .555556 | .564706 | .458824 |
| Color, live replacement | .606061 | .564706 | .452632 |
| All joint-slot / D-only variants | .741176 | .685714 | .670270 |

Every joint-slot and D-only label array is exactly the reference, despite changed
new-head probabilities. All direct live variants regress against selected8.
The new color live variant matches only2/2/1short IoU.5spells versus3/3/3for the
reference. The metadata live variant matches2/2/0. No proposed variant meets the
+.04buffered and extra-short-match requirements or all protocol protections.
There are24primary regressions among78candidate/clipped-block comparisons.
The known71-frame away possession has0correct raw argmax frames for both new
heads and the matched control in all3protocols. The failure therefore also exists
in the classifiers, before the conservative joint decoder. This is diagnostic
only and is not used to change a frame, threshold, training target or plan.

All30official score rows agree. All26new and13existing checkpoints reload to
exact saved probabilities; shared decoder predictions are exact. Training-only
targets, weights,250-frame buffers, source/input/cache hashes, and selected8
reference replay verify. Three new synthetic tests cover interpolation gaps and
half boundaries, exact prefix/team symmetry and bounded past/future support;
all157tests pass. Actual18935vs19060public feature parity is exact.

A separate workspace regenerates every feature cache and preparation manifest
byte-for-byte, then reruns the ordinary model-reload/shared/official/gate audit.
All metrics, diagnostics and checks reproduce exactly. It reuses cached contact
observations and saved classifiers; it does not repeat the original video/NN
processing or all26model fits. No current-round before archive or candidate
snapshot is needed for that replay.

All selected candidate artifacts and every preceding ledger byte are preserved.
The30dependent rows append once:1,372total =778inherited +594added in this chat.
No confirmation seeds, full-match candidate, promotion or review was warranted.
Adaptive same-match public validation does not establish private accuracy or a
maximum score. No research commit, upload, publication, submission or contact
was made.

Evidence under `data/control_contact_heads_v1/` includes before-source/candidate
snapshots, target-blind inputs, saved26heads and fitting arrays,30-row screen,
official comparison, `screen_checks.json`, `integrity.json`, `block_deltas.csv`,
`raw_head_short_diagnostics.csv`, `label_identity.json`, test logs,
`fresh_head_replay.py`, `fresh_feature_parity.json`, and `round_closeout.json`.
All tracking-derived arrays remain local under ignoreddata.
