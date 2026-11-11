# Round37: fixed player-set representation transfer

The pretrained representation improves all three aggregate spell scores, but
fails the frozen substantial-gain and secondary gates. Selected8 is unchanged.
The random encoder performs worse on two protocols. No source model or decoder
threshold is tuned after seeing this result.

The [prospective plan](PLAYER_REPRESENTATION_PLAN.json) fixes the R36 temporal
checkpoint and an identically initialized untrained encoder. Each supplies
64 oriented pre-classification features, averaged across four pitch reflections.
These join the original102 ordered features in the unchanged XGBoost head.
Only slot0 changes in the complete selected8 pipeline, and original D/X are
preserved. This is a different supervised interface from R36's failed scalar
override; that earlier route remains rejected.

| Protocol | Selected8 | Pretrained latent | Random latent |
|---|---:|---:|---:|
| Buffered minutes | .741176 | .754286 | .690909 |
| Purged spells | .685714 | .720000 | .654545 |
| Buffered100seconds | .670270 | .691892 | .681081 |

Pretrained macroF1 also improves (.870833→.876330, .871604→.879389,
.844383→.851309). It retains all13 matched possessions but still matches only
three of six shorts. Buffered spurious count grows4→5 and mean-fold primary
regresses. Its+.013109 buffered gain falls short of+.04; no extra short match
appears. Thus neither view proceeds to seed confirmation or candidate building.
These adaptively reused public protocols are development evidence, not an
independent or private score.

All13 original-head refits reproduce saved probabilities and complete selected8
labels before the26 new fits. All39 saved heads reload exactly. Source extraction
reads no targets; actual18935/19060/full199319 inputs produce identical1501
windows, including the final interpolation endpoint. Hidden states are identical
between prefix and full inputs; team swaps, reflection averaging and the R36
source probability identity pass. The observation footprint is79 frames, within
the250-frame validation exclusion. Both interpolation endpoints must be observed.

All327 tests pass. Input/encoding rebuilds, checkpoint inference, shared scoring
and official score/gate replays match byte-for-byte. Twelve prespecified score
rows append to the ledger, preserving its1519-row prefix (1531 total).
All17 protected artifact hashes and source links remain exact. No manual labels,
private targets, reserved media, installations, submissions or contacts are used.
An initial invocation missed the repository Python path; its failure log is
preserved, and setting PYTHONPATH fixes execution without changing frozen source.

Evidence: ignored `data/control_player_representation_v1/`. Selected CSV SHA256
remains `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
The broader user goal remains active; this representation/interface is closed.
