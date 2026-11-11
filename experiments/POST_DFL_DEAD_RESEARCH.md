# After DFL live/dead transfer: ball-free learned possession evidence

Prepared 2026-10-25. Advisory primary-source research, not an executed or frozen
experiment. Only this new note was written. No `data/` path, competition media,
tracking, frame labels or probabilities was opened; no weights were downloaded,
packages installed, fits or scores run, or selected artifacts changed. Repository
source, aggregate reports, public papers and official repository text/metadata
were reviewed. The already-dispatched background researcher applies the research
skill; this does not create a separate chat.

## Recommendation

The strongest bounded next step is **one no-fit feasibility and native-signal
audit of the released PathCRF dynamic masked checkpoint**, using player motion
alone, a genuinely bounded local adapter and a center-repeated mechanistic
control. Do this before training another external teacher. This tests a new
observation channel with existing learned weights; it does not presume useful
native signal or authorize a possession-score screen, full candidate or promotion.

The parent reports that Round 33's paired D-head addition leaves all three spell
scores unchanged. Its separately frozen Round 34 raw native D audit has balanced
accuracy .6035, D recall .2192, zero visible-ball D recall on 949 public D frames,
and high-confidence correction precision .1167/.1655/.0295. These are aggregates
supplied by the executing parent, whose report is being prepared, not results
independently replayed here. They make another ball-dependent D teacher especially
unattractive; they do not establish that ball-free motion transfers.

The historical ownership teacher already fails on the selected system's errors:
native balanced accuracy .854971 coexists with correction precision only
.0875/.1412/.2000, and no additional missed-possession recovery. Its shared inputs
contain ball proximity and ball kinematics. Public-only temporal D learning and
handcrafted full-team shape also failed. PathCRF's learned, player-specific
temporal representation remains different from those tested paths.
[DFL_SIGNAL_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/DFL_SIGNAL_REPORT.md),
[dfl_transfer_features.py](/Users/derekko/Projects/more-than-a-hack-2027/experiments/dfl_transfer_features.py),
[TEMPORAL_DEAD_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/TEMPORAL_DEAD_REPORT.md),
[SPATIAL_SHAPE_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/SPATIAL_SHAPE_REPORT.md).

## What the primary work actually supplies

PathCRF predicts player-to-player possession edges from player trajectories. A
self-loop represents control, while a directed edge represents an airborne kick.
Its paper trains on the seven public DFL matches, using synchronized events and
ball-derived extra touches as external supervision. It trains/evaluates on true
in-play episodes, at 5 Hz in 10-second windows. Reported event F1 .7569 uses
player/event matching within one second, not this challenge's team-spell IoU
score. This supports investigating the representation, not transferring that
headline score. [PathCRF paper, §§2–3](https://arxiv.org/html/2602.12080v1).

The official repository advertises five trained variants. Use only the dynamic
masked model in `saved/140`, prospectively, without comparing checkpoint choices.
The public repository is MPL-2.0; retain its source notices. This review fetched
repository metadata and source text only, not the binary checkpoint.
[Official repository](https://github.com/hyunsungkim-ds/pathcrf),
[license](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/LICENSE).

Pinned source commit: **`1b7c6f85a6b78c92e4543fde72ca2b86a933f997`**, committed
2026-06-08, returned by the official GitHub commit API. One predeclared artifact:

- Path: `saved/140/model/state_dict_best_acc.pt`.
- Publisher metadata: **19,325,114 bytes**, Git blob SHA
  `f8e48952d01d6d70db986948962822478884a336`.
- [Commit-pinned checkpoint URL](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/saved/140/model/state_dict_best_acc.pt).
- [Commit metadata](https://api.github.com/repos/hyunsungkim-ds/pathcrf/commits/1b7c6f85a6b78c92e4543fde72ca2b86a933f997),
  [checkpoint-directory metadata](https://api.github.com/repos/hyunsungkim-ds/pathcrf/contents/saved/140/model?ref=1b7c6f85a6b78c92e4543fde72ca2b86a933f997).

The other published file is `state_dict_best.pt`; do not try it after a failure.
A Git blob SHA is not a file SHA256. Later acquisition must verify publisher
size/blob identity, retain a local SHA256 and load tensors with
`torch.load(..., map_location='cpu', weights_only=True)` plus strict state keys.
Actual binary reachability, serialization compatibility and loading remain
unverified by this review.

### Interface and dependencies

The model's published arguments specify 3,431,031 parameters, 11 players per team,
eight node features, attention sequence modules and a dynamic sparse CRF. The
input node order is left team, right team, then four outside nodes. The eight
features are team indicator (-1/+1; outside 0), keeper indicator, x, y, vx, vy,
speed and acceleration. [Published arguments](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/saved/140/args.json).

Coordinates use a standardized 105×68 metre pitch, not z-score or min-max inputs.
The dataset code passes physical features as float32 without a fitted scaler;
its reflection augmentation also flips appropriate velocity components and
reorders team blocks. Source pitch dimensions must be transformed consistently,
including derived motion. [Pitch constants](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/datatools/config.py),
[dataset and augmentation](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/dataset.py).

Call `model(input)` with shape `[B,T,26,8]`, leaving `macro_target`,
`micro_target` and `random_mask` all `None`. The forward then creates zero
target channels; no manual event, DFL owner/status or true possession initializer
is required. Outputs are macro sender/receiver logits `[B,T,26,2]`, edge logits
`[B,T,676]` and edge embeddings `[B,T,676,16]`. Select `valid_orig_edge_ids` before
CRF decoding: its compressed state space has 576 edges. The temporal layers run
per node, so a consistent player correspondence inside each window is required.
There is no player-observation mask argument in this forward path.
[Model source](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/set_lstm.py),
[sparse CRF](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/dynamic_sparse_crf.py).

Set-attention and CRF calculations use ordinary Torch operations. Stock
`models/utils.py` nevertheless imports `torch_geometric.data.Batch/Data`, used
by graph batching, while the set model needs its pure-Torch edge-compression
helper. A narrowly documented extraction/lazy import of that helper can avoid
installing PyG; preserve names, state keys and mathematics and verify parity.
Do not replace attention with a merely similar Torch module. The published
requirements omit Torch and PyG and are not a complete inference lockfile.
[Utility source](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/utils.py),
[attention source](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/set_transformer/attention.py),
[requirements](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/requirements.txt).

The minimum source import closure for a standalone synthetic worker is:

| Source to preserve from the pinned commit | Imports needed by this path |
|---|---|
| `models/set_lstm.py` | Torch, math/typing, constants, set modules, four CRF classes, edge helper |
| `models/dynamic_sparse_crf.py` | Torch and typing; selected CRF implementation |
| `models/dynamic_dense_crf.py`, `models/static_dense_crf.py`, `models/static_sparse_crf.py` | Torch and typing; imported by `set_lstm.py`, even though not selected |
| `models/utils.py`: extract **only** `build_edge_compression_maps` verbatim into the worker's corresponding module | Torch and `typing.Tuple`; no NumPy/PyG needed by this helper |
| `set_transformer/model.py`, `set_transformer/blocks.py`, `set_transformer/attention.py` | Torch and imports among these three files |
| `datatools/config.py` | Only `PITCH_X=105.0`, `PITCH_Y=68.0` are required by the model |

Empty package markers may be local packaging conveniences. Preserve full source
files where practical; record the helper-only extraction as a declared import
adaptation with its original source hash. No preprocessing, dataset, stock
inference, training, kloppy, RDP or event module is required for tensor-only
forward/decode. Synthetic load, reload, fixed-node-permutation and bounded-window
tests can therefore run without PyG installation. This is a source dependency
conclusion, not a successfully executed load test.
[Pinned model source tree](https://github.com/hyunsungkim-ds/pathcrf/tree/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models),
[pinned set-attention tree](https://github.com/hyunsungkim-ds/pathcrf/tree/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/set_transformer).

Instantiate `SetLSTM` with `macro_type='poss_edge'`, `micro_type='poss_edge'`,
`seq_model_type='sab'`, `crf_model_type='dynamic_sparse_crf'`, `feat_dim=8`,
`team_size=11`, `macro_ppe_dim=16`, `macro_fpe_dim=16`, `micro_enc_dim=64`,
`macro_seq_dim=128`, `micro_seq_dim=128`, `crf_edge_dim=16`, `dropout=.1`, then
strictly load the raw state dict and call `.eval()`. The constructor creates
the CRF and all compression buffers. Published training saves
`module.state_dict()` directly, and `best_acc` means validation edge accuracy.
The reviewed args use DFL LOMOCV fold 6; this identifies source provenance,
not a fresh validation split for this project.
[Checkpoint-saving source](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/train.py).

Macro `[...,0]` and `[...,1]` are separate sender and receiver logits: normalize
each over its 26 nodes, not over the two channels. They are auxiliary outputs,
not a 26-class joint possession distribution. For structured inference, select
the 576 valid indices along edge dimension 2 of micro logits and embeddings,
call `model.crf.decode(emissions, embeds, last_node_idx=None)`, and map returned
compressed IDs through `model.comp_src`/`model.comp_dst`. Original edge IDs are
`src*26+dst`. A player sender's physical-team mapping supplies H/A; outside
sender states supply uncertainty. The code exposes Viterbi paths, not normalized
CRF per-frame marginals. Softmax of edge emissions is consequently **not** a
CRF marginal; adding forward-backward posterior inference would be a separate
declared implementation, unnecessary for this first audit.

CPU Torch 2.5.0 inference is plausible, not certified: weights alone require
about 13.7 MB at float32, but activations, buffers and independent overlapping
window calls add cost. The existing runtime is documented by
[temporal_dead_network.py](/Users/derekko/Projects/more-than-a-hack-2027/experiments/temporal_dead_network.py)
and [TEMPORAL_DEAD_PLAN.json](/Users/derekko/Projects/more-than-a-hack-2027/experiments/TEMPORAL_DEAD_PLAN.json).
This review did not inspect/run that runtime because it resides under `data/`.
Require a fresh-process, two-thread, deterministic synthetic load/forward/decode
benchmark, including `[8,51,26,8]`, before any native inference. Freeze a practical
budget, e.g. projected complete public two-arm audit under one CPU hour and peak
memory under 2 GB; stop if it fails rather than changing windows after outcomes.

## A legitimate bounded adapter is possible; stock inference is unsuitable

Stock `inference.py` can infer episodes from `ball_state`, infers keepers across
whole phases, and defaults `evaluate=True`. Its nominal windowed mode blends
emissions before whole-episode CRF decoding; the final window can also exceed its
nominal length. Thus `window_seconds=10` alone does not establish a bounded
prediction. Disable evaluation and episode-end correction; bypass its whole-
episode assembly. Call network and CRF independently on each local window and
retain only its center. Use `last_node_idx=None`, not a true/predicted next-event
constraint. [Inference source](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/inference.py).

Selected8 **predicted** live episodes can supply resets, provided each protocol
uses its own frozen OOF predictions and no true episode/status labels. A reset
must not seed H/A ownership. Clip within the fixed local observation window and
half; no search to a distant episode endpoint. True-live segmentation, true
receiver/control events or an event-derived terminal node rejects the audit.
Freeze reference outputs before producing new predictions and preserve their
D/X exactly. They remain upstream model decisions and can be wrong.

A 51-sample window at 5 Hz covers ten seconds between endpoints: offsets
`-125,-120,...,125` at 25 Hz. The training argument nominally uses 50 samples;
51 is a small but explicit inference adaptation, not exact training parity.
Alternatively an exactly 50-sample asymmetric window is possible, but choose one
before acquisition/native processing. The 125-frame radius excludes derivative,
stitching and any downstream smoothing reach; audit their sum against the
existing 250-frame exclusion. Global Viterbi or unlimited identity stitching
would invalidate that argument.

The official running-feature helper derives motion from XY, removes >12 m/s
player outliers, interpolates over removed points, smooths velocity with a
15-frame Savitzky–Golay filter and acceleration with nine frames. Its period-wide
interpolation is potentially unbounded. Use one prospective bounded substitute;
do not import provider speeds unavailable with matching semantics in both
domains. A local geometry-based keeper guess is observation-only, but its
correctness and short-window stability are unverified.
[Running-feature and keeper source](https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/datatools/utils.py).

Native slots must not be differenced as identities. Existing code explicitly
warns of identity-changing slots; earlier mutual-nearest co-motion is only an
approximate correspondence.
[inspect_player_motion.py](/Users/derekko/Projects/more-than-a-hack-2027/experiments/inspect_player_motion.py),
[PAIR_MOTION_PLAN.json](/Users/derekko/Projects/more-than-a-hack-2027/experiments/PAIR_MOTION_PLAN.json).
Freeze same-team anonymous stitching using observations only, with a speed cap,
unique/margin checks, bounded interpolation and deterministic tie rejection.
Missing/ambiguous trajectories must yield unsupported centers. Do not turn zero
padding into evidence of an observed player: stock training drops NaN episodes,
and its computed padding mask is not consumed by the set forward.

## One prospective experiment and stopping rules

The executing parent must freeze the implementation, artifacts, all thresholds
and acceptance conjunction before downloads or native adapter construction.
The following is a recommended bounded design, not a replacement freeze.

1. **Technical/support gate, before target reads.** Pin the single checkpoint and
   source above. Verify strict loading, finite deterministic outputs, CRF/edge
   indexing, source-slot permutation and team-renaming parity, true-field
   removal/randomization invariance, actual short/full prefix parity and
   perturbation outside the declared observation reach. Require finite complete
   model inputs backed by the declared stitching/interpolation rule. Reject
   incomplete windows; no target-conditioned choice of supported players.
   A useful prospective minimum is supported centers covering at least 25% of
   selected-live public frames in every protocol. If this is infeasible, stop
   this off-the-shelf path without relaxing support to obtain a score.

2. **Matched inference control.** Candidate: the actual local temporal window.
   Control: repeat its center's complete eight-feature snapshot through all 51
   steps, with the same checkpoint, team/keeper assignment, validity eligibility,
   node order, CRF and center extraction. Also retain an unchanged selected8
   identity reference. The repeated arm retains instantaneous derived motion,
   isolating the additional trajectory history. It is an inference ablation,
   not an independently trained matched-capacity model, and may itself be out of
   distribution. Do not claim its loss proves temporal causality.

3. **One raw ownership statistic.** Obtain the center decoded edge's sender team
   as last controller, matching the challenge's retention of ownership while
   the ball travels. Keep outside sender/self-loop predictions as unsupported
   uncertainty, not D. For confidence use center raw emission softmax summed by
   sender team over valid edges, retaining outside mass separately; conditional
   pH divides H mass by H+A mass. Freeze e.g. outside mass <=.10 and chosen-team
   conditional support >=.65. These are emission-support numbers, not calibrated
   CRF posterior probabilities. Do not silently use receiver team, which can
   anticipate a turnover before control occurs.

4. **Native-signal gate, no fitting or possession-ledger rows.** Join the existing
   public targets only after immutable outputs are saved. Cover all 17 public
   possessions, all six short cases, all four currently unmatched cases and all
   spurious predictions; unsupported cases stay in denominators. Require in
   every protocol: supported-live balanced accuracy >=.70; at least 25 confident
   candidate-correct/reference-wrong frames; correction precision C/(C+H) >=.55;
   and candidate C-H at least 25 greater than the repeated-center arm on common
   eligibility. Report coverage, both class recalls, all disagreements and
   uncertainty explicitly. The stronger requirement is one common originally
   unmatched short case recovered by one predeclared conservative diagnostic
   overlay, with no prior GT match lost and no extra spurious prediction. Freeze
   overlay/matching before the join; preserve selected D/X and retain the
   reference on unsupported centers. This is an opportunity test, not the final
   possession score or a guarantee of achievable integration.

Any failed/undefined conjunction stops the checkpoint avenue. No second
checkpoint, stitching relaxation, keeper rule, confidence sweep, external refit
or alternate integration follows within this experiment. A pass justifies one
separately frozen integration; it does not waive the inherited +.04 buffered
spell-F1, one extra short match, all secondary protections and seeds 43/44
confirmation requirements. Selected8 remains .741176/.685714/.670270 until a
candidate passes that complete process.
[DFL_SIGNAL_PLAN.json](/Users/derekko/Projects/more-than-a-hack-2027/experiments/DFL_SIGNAL_PLAN.json),
[PAIR_MOTION_PLAN.json](/Users/derekko/Projects/more-than-a-hack-2027/experiments/PAIR_MOTION_PLAN.json).

## Why this precedes training or trajectory repair

| Path | Genuine advantage | Why not the immediate next experiment |
|---|---|---|
| Small team-set temporal DFL H/A learner | Masked per-frame set attention, pooling by team **before** temporal encoding avoids player IDs/keeper guesses; direct existing DFL H/A labels need no new manual annotation. | A sensible later original model, not a released/validated solution. Pooling first discards individual temporal correspondence, and a new CPU training budget plus center-only trained control is needed. Static shape and public-only temporal failures temper expectations. |
| Ball Radar | Official set-transformer/Bi-LSTM possession-plus-ball inference code; public Metrica reproduction route. | Its README's principal dataset includes 15 private Fitogether games and three public Metrica games; the root listing/instructions do not establish a downloadable pretrained checkpoint. Predicting a plausible ball and then assigning ownership adds a regression/heuristic error layer. [Official source](https://github.com/hyunsungkim-ds/ballradar). |
| TranSPORTmer | Masked social/temporal attention and XY-only ball inference; four state classes include out-of-play. | Paper soccer training uses 283 LaLiga matches, not the seven-match public DFL source. Four state labels do not identify team ownership. A public soccer checkpoint/download was not located in reviewed primary sources; do not call it available. [Paper](https://arxiv.org/html/2410.17785v1). |
| MIDAS/DBHP-style self-supervised repair | Released derivative-consistent imputation code; observation masking can train without team-possession annotations. | The demonstrated imputation targets are player trajectories, not a validated soccer-ball possession repair. Native observed ball points are not verified targets; withholding them can reward reproducing tracking errors. Smooth low-error XY does not certify short-control recovery. [Official MIDAS code](https://github.com/gkswns95/MIDAS), [primary paper](https://arxiv.org/abs/2408.10878). |

The existing local contact experiment already tested bounded ballistic/median
ball repair and failed to fix brief controls. Repeating that under a
self-supervised name is not new evidence.
[CONTACT_DYNAMICS_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/CONTACT_DYNAMICS_REPORT.md).

PathCRF's shipped DFL training/validation supervision is legitimate external
training, not competition-label leakage. However, its DFL splits overlap the
seven already-inspected matches; they are not fresh holdouts for selecting this
checkpoint or making another external generalization claim. No competition raw
observations, labels, probabilities or clips should leave local processing.
The tiny reused public possession universe remains adaptively overused, and no
method here is guaranteed to produce substantial improvement.
