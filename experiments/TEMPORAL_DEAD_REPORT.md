# Automatic score work, rounds 30–32

Completed 2026-10-25. **No tested candidate improves the main possession score;
selected8 remains selected.** This work follows the user's request to use provided
data and public labels without further manual annotation. Manual annotations are
not read as training/evaluation data or used in inference. No private score is
available.

## Round 30: learned temporal live/dead evidence

The [frozen plan](TEMPORAL_DEAD_PLAN.json) compares a compact temporal network with
a parameter-identical center-only control. Each has16 channels, five residual
kernel3 convolutions with dilations1/2/4/8/16 and a63-frame observation window.
The center-only control repeats the current frame across this window. Thirteen
folds fit both heads from scratch:26 public-only research models, no pretrained
weights or external labels.

Both heads use114 existing team-invariant physical features, training-only
normalization, missing flags and the same balanced binary targets, seed42,
600-step optimizer schedule and batch size128. Their maximum source reach is181
frames, inside the250-frame target exclusion. Actual18,935/19,060-row feature
reconstruction and comparison with the full tracking cache pass exactly. The
existing isolated CPU Torch2.5.0 environment is reused without installation.

Each head has two fixed integrations into complete selected8 inference. Trim may
remove an existing D prediction while preserving the original conditional team
probability. Union may add D using the existing physical stoppage detector.
Shared live ownership and all X labels stay exact. Unchanged-pipeline and self-D
controls reproduce selected8 before fitting.

| Method, seed42 | Buffered minutes spell F1 | Purged spells | Buffered100seconds |
|---|---:|---:|---:|
| Selected8 reference | .741176 | .685714 | .670270 |
| Center-only trim | .717647 | .731429 | .688889 |
| Center-only union | .694118 | .670588 | .670270 |
| Temporal trim | .717647 | .708571 | .688889 |
| Temporal union | .715152 | .690909 | .670270 |

Every variant loses the main score. Buffered IoU.5 matches fall13→12. Trim reduces
false D predictions but also removes true stoppages: temporal D recall falls
.748128→.620150 on buffered minutes. Temporal union improves buffered frame macro
F1 .870833→.884731 and D recall .748128→.843431, while false D frames rise88→175
and the possession score declines. A frame-accuracy gain does not satisfy the
possession objective. Secondary-protocol gains cannot override the frozen
main-protocol and protection failures.

No variant meets the prospective+.01 main-score gain or all protection gates.
The historical+.04 gain and extra-short-match objectives are also unmet. No
seed43/44 continuation, full candidate or promotion follows. A gated full-builder
source exists but is unexecuted; no full-build validation is claimed.

The six targeted transform/exclusion/overlay tests pass. The synthetic order-only
task reaches100% temporal accuracy versus50% center-only; deterministic refitting
and checkpoint reloading are exact in the pinned CPU runtime. These establish
implementation behavior, not competition accuracy. All26 public checkpoints
reload. A fresh process reproduces128 common artifacts byte-for-byte, including
probabilities, fitting arrays, predictions, official/shared scores and gates.
Fifteen dependent score rows append after Round31's earlier completion, retaining
all preceding ledger bytes.

Plan SHA256:
`4e36d16ffe8c05dfce779c6a0691aa98e114735f6f4dc007de95e4210a64a291`.
Local evidence: `data/control_temporal_dead_v1/`.

## Round 31: automatic short-stoppage bridge

The separately frozen no-fit bridge accepts0/4,0/6,0/4 D runs across the three
protocols. Scores are exactly unchanged. The independently frozen rule is not
relaxed after seeing this result. See [DEAD_BRIDGE_REPORT.md](DEAD_BRIDGE_REPORT.md).
Its34 common replay artifacts agree byte-for-byte and six dependent rows append.

## Round 32: conservative agreement follow-up

After observing Round30's false-D increase, a [new frozen plan](TEMPORAL_CONSENSUS_PLAN.json)
tests exactly one dependent follow-up. It uses the minimum of the two saved D
probabilities, requiring both heads to support adding a stoppage, through the
unchanged union integration. No new model, threshold sweep or supervision is
introduced. The two heads are correlated; this is adaptive development using
the same17 public possessions.

| Protocol | Reference spell F1 | Agreement spell F1 | Reference→agreement false D |
|---|---:|---:|---:|
| Buffered minutes | .741176 | .694118 | 88→173 |
| Purged spells | .685714 | .670588 | 142→214 |
| Buffered100seconds | .670270 | .670270 | 16→16 |

This fails the unchanged Round30 quality conjunction. No confirmation, full
candidate, further threshold or composition follows. Fresh shared/official replay
reproduces33 common artifacts byte-for-byte. Six dependent score rows append.

Plan SHA256:
`e59c51addad0272b0595212828a75c59b0444a13bbdafea468bedb6f85fa32fd`.
Local evidence: `data/control_temporal_consensus_v1/`.

## Final verification and limits

All307 tests pass. All17 protected selected artifacts, both source symlinks,
frozen source/input hashes and reserved-window metadata verify. Every byte of the
original1,483-row ledger remains a prefix. Twenty-seven dependent rows append:
1,510 total,778 inherited plus732 continuation; append idempotency is checked.
Detailed targets, traces, features and probabilities remain in ignored local data.
The audit stays sealed. No candidate is changed or submitted.

Selected CSV SHA256 remains
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
Its public spell scores remain .741176/.685714/.670270. The tested changes do not
improve those scores reliably across the frozen requirements. This does not
establish optimality, private accuracy, or the absence of useful automatic
information; it rejects these particular models and integration rules. No manual
annotation is required to reproduce or continue the automatic pipeline.
