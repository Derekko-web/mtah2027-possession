# Round36: masked player-set temporal ownership

Completed 2026-10-25. **The new model transfers useful ownership information,
but direct high-confidence corrections fail. No possession-score improvement or
candidate is claimed.** Selected8 and all1,519 ledger rows remain unchanged.

The [research note](MASKED_PLAYER_OWNERSHIP_RESEARCH.md) motivates an original
missing-aware model after PathCRF could not cover any short possession. Runtime,
external fitting, native support/inference and raw signal were separately frozen:
[A](MASKED_PLAYER_RUNTIME_PLAN.json), [B](MASKED_PLAYER_EXTERNAL_PLAN.json),
[C](MASKED_PLAYER_NATIVE_PLAN.json), [D](MASKED_PLAYER_SIGNAL_PLAN.json).

## Model and external evidence

The24,321-parameter model encodes unordered player XY sets with an explicit
observation mask and null token. One32-wide/four-head attention layer pools team
means/counts before a31-sample5Hz temporal CNN. Shared scoring enforces H/A
equivariance. It requires no stable slots, keeper identity, complete roster, ball
coordinates or supplied velocity. The maximum observation radius is75 frames.

Synthetic temporal-only accuracy is1.0 versus.5 for the center-only control.
Masked NaNs, all-missing neutrality, node permutation, team swap, deterministic
refitting and strict checkpoint reloading pass. Projected two-fit training time
is1,648seconds and peak synthetic RSS.555GiB, within the frozen30minute/2GiB
limits. The existing CPU Torch2.5.0 runtime is reused; no installs occur.

The public DFL source supplies direct live H/A labels from four training matches.
Coordinates are normalized by pitch metadata. Clean/dropout/crop/burst views are
built and hashed before targets are joined. No status-dependent input masks or
true episode cuts are used. A saved equal-match/class schedule supplies4,000
steps×32 examples to each arm, with identical views/reflections/initialization.
The control repeats the corrupted center snapshot; both have equal capacity.
Training uses60,259 eligible source centers, and validation50,148 centers.
Both fits complete their entire budget; peak RSS is.726/.733GiB.

| Masked validation match | Temporal balanced accuracy | Center-only balanced accuracy |
|---|---:|---:|
| WPY | .795930 | .725373 |
| WQQ | .837778 | .763777 |
| WR9 | .828887 | .772198 |

Temporal passes balanced accuracy>=.75 and both class recalls>=.65 in all three
matches. The center model fails WPY. Temporal is fixed before native targets.
These historical4/1/2 partitions were already used in earlier research and are
not described as untouched validation. DFL/Bassek et al.2025, CC-BY4.0:
[primary source](https://www.nature.com/articles/s41597-025-04505-y).

## Complete native coverage, unsafe direct corrections

All1,500 uniform public centers meet the fixed four-player-per-team support
checks. Actual18,935/19,060-row inputs and the199,319-row source cache produce
exact public windows and model probabilities. All inputs use player observations
only. Unlike PathCRF, all short cases have observation support.

An initial native loader incorrectly rejected windows crossing kickoff. Public
frame11185 is the official first-half start. Execution revision1 supplies masked
padding outside the known half, as the original plan required. The first failed
source/run is preserved. Model weights, thresholds, targets and gates are
unchanged; synthetic boundary tests and actual-prefix replays pass.

The fixed p>=.65 /p<=.35 proposals can edit selected-live H/A only. D/X remain
exact. The diagnostic overlay requires three consecutive qualified5-frame cells
and resets at validation blocks. Outputs are immutable before the target join.

| Protocol | Native balanced accuracy | Corrected centers | Damaged centers | Correction precision | Diagnostic spurious |
|---|---:|---:|---:|---:|---:|
| Buffered minutes | .912808 |25|28|.471698|4→6|
| Purged spells | .914492 |26|26|.500000|5→7|
| Buffered100seconds | .911288 |33|27|.550000|7→6|

Every protocol loses a prior matched possession, and none adds a short match.
Net corrections are-3/0/6 rather than the required25. Direct scalar-teacher
integration is rejected without lowering confidence thresholds.

There is a bounded reason to test a different interface: the model gets9/14
sampled decisions right in the previously missed71-frame possession, but none
is confident enough for the direct proposal. It also makes confident errors in
already-correct regions. This supports investigating supervised conditional use
of pre-classification representations; it does not establish that such a new
experiment will improve the score or waive its original substantial-gain gates.

## Verification and unchanged selection

All325 tests pass, including the native half-padding repair. A fresh corpus
rebuild matches10 artifacts byte-for-byte. All four external probability arrays,
external arm selection,11 native-input,3 native-output and5 signal artifacts
replay exactly. Both external checkpoints reload strictly. All17 selected
artifact hashes, source/input guards and original source links remain intact.
There are no native fitting or new possession-score rows in this round.

No human annotations, private fixture targets, audit media, competition-image
calls, installs, commits, uploads, submissions or contacts occur. Selected CSV
SHA256 stays `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
Evidence remains under ignored `data/control_masked_player_v1/`.

The wider user goal is active. Any later representation-transfer trial must
freeze its separate supervised interface and controls before possession scoring,
and still meet+.04 buffered spellF1, one additional short match, all secondary
protections, all seeds/protocols, reproducibility and independent review.
