# Round35: bounded ball-free PathCRF opportunity audit

Completed 2026-10-25. **Native signal is accurate where supported, but adds no
short-possession recovery. The checkpoint avenue is rejected; selected8 remains
unchanged.** No possession F1 screen, fit, candidate or ledger row follows.

The research skill's [primary-source note](POST_DFL_DEAD_RESEARCH.md) motivates a
new player-motion channel after failed ball-dependent external transfer. Three
prospective stages freeze [runtime feasibility](PATHCRF_FEASIBILITY_PLAN.json),
[observation support](PATHCRF_SUPPORT_PLAN.json), and
[paired native signal](PATHCRF_SIGNAL_PLAN.json) before their respective outputs.

## Public source and runtime

PathCRF infers directed player-possession edges from player trajectories. Source:
Kim et al., KDD2026, [primary paper](https://arxiv.org/abs/2602.12080),
[official repository](https://github.com/hyunsungkim-ds/pathcrf), MPL-2.0.
Pinned commit: `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`.
Only `saved/140/model/state_dict_best_acc.pt` is downloaded:19,325,114bytes,
publisher Git blob `f8e48952d01d6d70db986948962822478884a336`.
All43 downloaded code/metadata/license/model files match publisher identities.
No upstream observations, event labels or other checkpoints are downloaded.

The original set-attention and CRF source is retained. Its pure-Torch edge-map
helper is extracted verbatim, with AST equality checked; unused PyG batching
utilities are omitted and MPL notices retained. No model math/state keys change.
The existing Torch2.5.0 CPU runtime strictly loads all3,431,031 parameters with
`weights_only=True`, without installing dependencies. Synthetic input/CRF checks
pass: repeatability, fixed source-node relabeling, finite interfaces and fresh-
process checkpoint reload. Six synthetic prediction artifacts replay byte-exact.
Measured .0592seconds/window projects177.5seconds for the planned1,500-center
two-arm upper bound; peak RSS .901GiB, below2GiB and20minute gates.

## Target-blind native inputs

Stock inference is bypassed: it can use true episode fields and decode over a
whole episode. This adapter calls the network and CRF independently for each
51-sample5Hz window, all target channels `None`, no terminal-node constraint,
and retains only its center. Original selected8 D/X are preserved.

Inputs use player XY only. A301-frame bounded neighborhood constructs anonymous
same-team correspondences with unique linear assignments, physical speed and
ambiguity checks, bounded interpolation and minimum observed support. Local
orientation and keeper guesses need no player identities. Source-matching
kinematic smoothing feeds offsets-125..125 in5-frame steps. Total source radius
is150, inside250; no period-wide interpolation or hidden unbounded context.
The complete model core must be predicted live and source context must stay
inside its validation block/half. Missing required trajectories reject a center.

Both actual18,935/19,060 tracking prefixes match the full-cache public source
neighborhood. Eighteen valid feature-prefix/team-renaming witnesses pass, as do
synthetic slot permutation, identity-switch, missingness and outside-footprint
tests. Support is measured before any model output or target join:

| Protocol | Selected-live grid centers | Supported centers | Coverage |
|---|---:|---:|---:|
| Buffered minutes | 1,262 | 372 |29.48%|
| Purged spells | 1,241 | 371 |29.90%|
| Buffered100seconds | 1,306 | 461 |35.30%|

All pass the prospective25% coverage minimum. There are461 union windows.

## Native signal and failed continuation

The real temporal arm and center-repeated control use the same checkpoint,
roles, eligibility and CRF. The control repeats the center's complete snapshot;
it is an inference ablation, not a separately trained matched-capacity model.
Both immutable output sets are saved before joining the public targets.

Ownership is the center edge's sender, retaining control in flight. Outside
sender states are unsupported. Confidence uses raw emission sender-team sums,
outside mass<=.10 and chosen-team conditional mass>=.65; these are not calibrated
CRF posterior probabilities. One fixed diagnostic overlay requires three
consecutive qualified5-frame cells and preserves D/X.

| Arm/protocol | Supported-live balanced accuracy | Corrected centers | Damaged centers | Diagnostic spurious change |
|---|---:|---:|---:|---:|
| Temporal/buffered |.990909|5|3|4→6|
| Temporal/purged |.990909|7|3|5→7|
| Temporal/100seconds |.991968|6|3|7→9|
| Repeated/buffered |.955494|2|14|4→6|
| Repeated/purged |.955494|4|14|5→7|
| Repeated/100seconds |.950294|3|16|7→9|

Neither arm recovers an additional short match. Crucially, **all six short
possessions and all four previously unmatched possessions have zero supported
centers** in every protocol. The high balanced accuracy describes selected,
mostly already-correct regions. It does not establish useful evidence on the
remaining errors. The temporal arm's5/7/6 corrections fall short of25, net gains
are2/4/3 rather than25, and both diagnostic overlays add spurious predictions.
All prior matches survive, but the complete continuation conjunction fails.
No checkpoint/window/roster/threshold relaxation follows within this avenue.

## Verification and reporting correction

The first case report accidentally used union-supported output centers as its
sampling denominator. Original source/results/replay are preserved under ignored
local data. Execution revision1 counts the full fixed1,500-center grid, with a
complete-live-denominator assertion. It changes only case reporting; raw outputs,
accuracy/correction statistics, matching, cutoffs and rejection are unchanged.
Corrected summaries and diagnostics replay exactly in a fresh process.

All320 tests pass. Fresh processes reproduce8 observation-support,4 model-output
and8 corrected-signal artifacts byte-for-byte. All17 protected selected hashes,
source/input guards, original source links and every byte of the1,519-row ledger
remain exact. No human annotations, private fixture targets, audit media,
competition-image calls, installs, fits, new possession-score rows, commits,
uploads, submissions or contacts occur. Selected8 CSV SHA256 remains
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.

Local evidence: `data/control_pathcrf_v1/`. The wider user goal remains active.
A new learner that accepts partially observed anonymous player sets is a
different hypothesis; this result neither validates it nor permits treating
missing players as observed in the rejected pretrained checkpoint.
