# Mask-aware anonymous player sets for external ownership supervision

Prepared 2026-10-25. Advisory research applying the already-read research skill.
Only this new note was written. No existing file, data, checkpoint, selection or
score changed; no data/media/target arrays were opened, fits or scores run,
packages installed or weights downloaded. Sources are public primary papers,
official Torch source, local implementation/protocol text and aggregate results
reported by the executing parent. Numerical proposals below are unverified
recommendations until the parent's prospective protocol fixes them.

## One recommended experiment

Train **one compact mask-aware, set-first temporal H/A learner on the existing
public DFL direct ownership labels**, alongside a separately trained center-only
control of identical capacity. Both learn from scratch. The experiment addresses
missing observations and arbitrary slot fragmentation by design, with no player
correspondence step. External validation and a raw native opportunity audit must
pass before any competition-label fitting or integration.

This is a different hypothesis from relaxing PathCRF checkpoint 140. The parent
reports that its frozen Phase 35 technical/support gates passed: support covered
29.5/29.9/35.3% of selected-live centers. Temporal balanced accuracy on the
359/359/448 supported true-live centers was .9909/.9909/.9920, versus
.9555/.9555/.9503 for center repetition. However, temporal corrections were only
5/7/6 with three damaging changes in each protocol, no extra short case and two
additional diagnostic spurious possessions. The frozen continuation failed and
that checkpoint avenue is closed. These aggregates were supplied by the parent,
not independently replayed here. High accuracy on supported centers does not
establish coverage or reliable corrections on the selected system's errors.
[Frozen PathCRF support protocol](/Users/derekko/Projects/more-than-a-hack-2027/experiments/PATHCRF_SUPPORT_PLAN.json),
[frozen signal protocol](/Users/derekko/Projects/more-than-a-hack-2027/experiments/PATHCRF_SIGNAL_PLAN.json).

The new learner pools independently within each frame **before** any temporal
layer. It therefore cannot mistake a changing native slot for a persistent
person. It accepts variable observed rosters and can be trained on missing
players, unlike the complete-window pretrained interface. The tradeoff is real:
pooling removes individual motion paths, so the temporal module must learn from
changes in learned team geometry and observation patterns. Neither set theory
nor the Phase 35 accuracy proves those pooled changes identify the missing short
possessions.

## Primary-source basis and what is original here

Deep Sets supplies the shared-element-encoder plus invariant aggregation pattern.
Set Transformer supplies attention over set elements to learn their interactions
without assigning meaning to their order. These justify a permutation-invariant
spatial representation; they do not establish soccer transfer, robustness to
incorrect team assignments or accurate team ownership.
[Deep Sets, NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html),
[Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html).

DFL's primary descriptor explicitly documents ball possession 1=home, 2=away,
and status 0=inactive, 1=active. Positions are in metres and the release contains
seven matches under CC BY 4.0. Use these existing external labels with proper
attribution; no additional user annotation is required. Its control convention
is not independently established as identical to this challenge during every
loose-ball, duel and restart phase.
[DFL data descriptor, Position data](https://www.nature.com/articles/s41597-025-04505-y).

The proposed network, corruption mixture and numeric gates are an original
bounded experiment. There is no claimed released checkpoint or published result
for this exact architecture. Implement directly with existing Torch 2.5.0 and
NumPy 2.1.2; importing PathCRF, PyG, an external set-transformer package or another
runtime is unnecessary. No downloaded source is needed for the model.

## Freeze this one representation and architecture

The parent's current proposed concrete recipe is coherent:

| Component | One fixed design |
|---|---|
| Observations | Finite player XY sets with home/away membership and explicit validity masks; normalize by documented field dimensions into a common centered coordinate range |
| Spatial embedding | Shared linear projection of `(normalized x, normalized y, null flag)` to width 32; no player/team positional index, identity, speed or keeper feature |
| Spatial interaction | One width-32 self-attention block, four heads, residual connections and layer normalization; feed-forward width 64, ReLU, dropout .1 |
| Empty-input handling | One explicit always-valid null token per frame; exclude it from physical-team pooling |
| Team pooling | Masked mean of 32-dimensional observed-player embeddings per team, plus count/11 and a nonempty indicator: 34 dimensions each |
| Temporal input | Concatenate H/A pooled vectors; 68→32 pointwise projection |
| Temporal encoder | Four residual width-32 kernel-3 convolutions, dilations 1/2/4/8, symmetric padding, ReLU and dropout .1 |
| Ownership logit | Shared score difference `g(H,A)=f(H,A)-f(A,H)`; sigmoid gives conditional pH |
| Context | 31 observations at offsets `-75,-70,...,75` in the native 25 Hz grid; output its center only |
| Fitting | Two models, seed 42, 4,000 optimizer steps each, batch 32, AdamW learning rate .001; no early stopping, checkpoint selection or hyperparameter sweep |

Freeze the remaining mundane optimizer settings too: e.g. weight decay .01,
gradient clipping at norm 1, float32 and final-step weights. The score difference
has the exact team-swap sign property at evaluation because the same spatial
encoder and temporal scorer serve both team orders. Spatial encoding need not
receive a home/away bit: membership is supplied to pooling, which prevents an
implicit team-specific spatial branch. No attacking-direction or goal-side
inference is required. With both teams entirely unobserved throughout a window,
the symmetric construction returns pH=.5.

This is about 24,000 parameters, an estimate until construction counts them.
The receptive field is `1+2*(1+2+4+8)=31` samples, or a 75-native-frame radius.
There is no kinematic or stitching halo. Any later overlay/smoothing reach must
be counted separately; no whole-episode Viterbi is part of this model. Preserve
half and validation-block boundaries. Use observations from dead portions of
the context when present: true live/dead segmentation must not select, reset or
pad windows. The model is trained to predict H/A only at known live centers.

Torch 2.5.0 already supplies the required attention and dilated convolution.
Its attention key-padding mask uses True for ignored keys; `need_weights=False`
avoids returning unnecessary attention arrays. Sanitize invalid XY to finite
zeros **before** projection/attention, then mask them, since a padding mask is
not a substitute for removing NaNs. Explicitly exclude padded query outputs
from means and divide by `max(count,1)`. The null token prevents an all-masked
key set. Padded zeros are never physical player observations.
[Official Torch 2.5.0 attention source](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.0/torch/nn/modules/activation.py),
[official convolution source](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.0/torch/nn/modules/conv.py).

Do not infer velocities, interpolate player tracks, identify a keeper, stitch
identities or require exactly 11 observations per team. Retain all valid
observed points, with only a predeclared resource/shape sanity ceiling; reject
an exceeded ceiling rather than silently choosing a label-dependent subset.
Keep native homography/geometry validity rules target-blind and share the same
finite/pitch-bound convention externally. Coordinate normalization is fixed by
field metadata, not fitted on public validation frames.

## One target-blind corruption mixture, not a search

Freeze four equally weighted training views: clean, dropout, crop and burst.
A defensible explicit initial recipe, for the parent to adopt or replace before
processing, is:

- **Clean:** observed XY only, with fixed random x/y reflection augmentation.
- **Dropout:** independently retain each available external player with
  probability .5 per 50-native-frame block; add .5 m standard-deviation XY
  noise to retained observations before the shared validity rule.
- **Crop:** retain points inside one 60×45 m rectangle whose center is drawn
  uniformly on the standardized pitch for each 100-native-frame block, plus
  the same noise. The rectangle is team-blind and never follows the ball.
- **Burst:** the dropout view plus one complete-observation blackout of 15–45
  native frames in each 250-frame block, at a uniformly drawn valid start.

Masks/noise operate on observations before pooling/window extraction. Generate
the per-half views once using frozen seeds and counter keys, independently of
owner/status, event boundaries and model outputs; save schedules/hashes. This
keeps neighboring validation windows consistent with one corrupted observation
sequence. Training may choose the four views uniformly for each sampled center;
use the same choices in both arms. Apply identical reflections to a window's
coordinates, preserving team targets, and randomly permute slots independently
at every time step after corruption. External identities may address which
observation is removed within a corruption block; no identity enters the model
or is required in native inference.

These masks approximate some dropout, fragmentation and limited-view effects.
They do not reproduce native tracking errors exactly: ball-following cameras,
systematic homography errors, duplicate detections and incorrect team tags remain
domain gaps. Crop centers or dropout probabilities conditioned on external owner,
true transitions, native labels or the ball would invalidate this ball-free
design. Do not forward-fill hidden players or reject examples because a true
controller became hidden. Actual source slot permutations and fragmentation
must leave the retained point sets/model outputs unchanged.

## Targets, whole-match split and a genuinely matched control

Reuse the frozen DFL identities: train WMX/WN1/WOH/WOY, development WPY, tests
WQQ/WR9, all prefixed `DFL-MAT-J03`. Keep both halves within their match. These
matches and previous results have already been inspected; call them fixed
development/test partitions, not untouched holdouts.
[Existing external split and target contract](/Users/derekko/Projects/more-than-a-hack-2027/experiments/DFL_TRANSFER_FEASIBILITY_PLAN.json).

Supervise center H only when `BallStatus==1` and `BallPossession==1`, A when
active and owner==2. Other centers are unsupervised. Do not smooth, interpolate
or propagate targets. Status determines loss eligibility only; it never gates
coordinates or context. Build observation tensors before joining targets and
verify that removing/randomizing all provider target/event fields leaves them
unchanged. No event synchronization or PathCRF pseudo-label is needed.

Sample known live centers at a fixed five-frame stride, with uniform sampling
over the eight train-match×class cells, then uniformly within the selected cell.
Use the same saved center/view/reflection schedule, initialization seed, optimizer
steps, parameter count and all target weights in both models. Candidate sees
the actual corrupted 31-frame window. Control repeats the corrupted center
snapshot at every in-half context offset; retain identical half padding outside
the half. Repeat **after corruption**, so newly sampled noise at other times
cannot become a hidden temporal channel. Train this control independently;
repeating frames only at inference would reproduce Phase 35's weaker ablation.

## Staged go/no-go gates before public fitting

The parent's final frozen protocol is authoritative; these are prospective
recommendations, not achieved results or permission to adjust failed gates.

**A. Synthetic/runtime and source contract first.** In the existing CPU runtime,
set two threads and deterministic algorithms. Test finite forward/backward for
empty/partial/full sets; padding-value irrelevance; per-frame arbitrary slot
permutation; team-swap complementarity; missing-frame handling; half resets;
and perturbations beyond radius 75. Use a synthetic order-only task with identical
center snapshots to verify that the temporal arm can learn a sequence signal
while the center arm stays at chance. This is capability evidence, not an
ownership benchmark. Verify a fresh-process deterministic tiny refit and strict
checkpoint reload. Use numerical tolerances for mathematically equivalent
permutations, and byte-exact replay for identical canonical inputs/batching.

Benchmark the exact frozen batch/window/max-slot shapes, including backward
passes and data construction. Require projected **two complete fits within 30
minutes** and peak RSS under a frozen ceiling, e.g. 2 GB, before DFL processing.
No speed/load certification is made here. Do not materialize every overlapping
window; gather minibatches. Inference can cache per-frame pooled embeddings
within each frozen model, because the spatial encoder has no temporal input.
Predict native centers at 25 Hz using stride-five context offsets, avoiding
five-frame output quantization of very short spells.
[Existing deterministic Torch worker pattern](/Users/derekko/Projects/more-than-a-hack-2027/experiments/temporal_dead_network.py).

**B. External paired gate, frozen before actual DFL processing/fitting.** Fit
exactly the two models on the four training matches. No development-based early
stopping or model choice. Evaluate all three fixed non-training matches with
clean and the three frozen corrupt views. One useful conjunction is: clean
balanced accuracy >=.75; average corrupt-view balanced accuracy >=.70; no corrupt
view below .60; and temporal-minus-control average corrupt-view balanced accuracy
>=.02, separately in each match. Report per-team recall, calibration, support
and conditional spell matching as well; these thresholds are suggestions with
no empirical guarantee. A failure stops, without retraining, changing corruption
or selecting the control. No external gate establishes native transfer.

**C. Target-blind native support and immutable outputs.** The parent's proposed
eligibility is at least four valid players per team at the center and mean count
at least four per team across its 31-sample context, with the full observation
radius inside the official half/evaluation block. Require >=80% of selected-live
centers supported in every protocol. No complete roster, identity continuity or
all-live surrounding episode is required by this newly trained masked model.
Compute this support before reading targets. If it fails, stop without relaxing
the condition. Freeze selected8 references and save both raw native pH streams
before any target join. Source-prefix parity must actually use the distinct
18,935- and 19,060-row inputs, not merely claim two lengths.

**D. Raw native opportunity gate, before competition-label fitting.** Cover all
17 possessions, six short cases, four currently unmatched cases, and every
spurious reference prediction. Report supported/unsupported counts per case,
including the old PathCRF-supported/unsupported strata. Never drop unsupported
cases from the opportunity denominators or select a favorable single miss.
Using fixed confidence >=.65, let C be confident teacher-correct/reference-wrong
true-live frames and H be confident teacher-wrong/reference-correct frames.
Require in every protocol: supported-live balanced accuracy >=.70, C>=25,
C/(C+H)>=.55, C-H>=25 and temporal net corrections at least 25 greater than the
trained center control on common eligibility. Undefined terms fail.

For one frozen diagnostic overlay, change H/A only on supported selected-live
frames where the temporal arm has confidence >=.65; retain reference elsewhere
and preserve D/X exactly. Require one **common** originally unmatched short
case newly matched at official IoU .5 across protocols, every prior GT match
retained and no additional spurious prediction. Freeze matching/overlay before
the join, with no alternate smoothing or proposal variant. This raw opportunity
test can reject an accurate but useless teacher before expensive public adapters.
Its matching counts are not a claimed possession-F1 improvement.

Any failed stage closes this specific pair/recipe. A pass justifies one later
separately frozen integration; it does not select a candidate. The inherited
requirements remain: >=.04 buffered spell-F1 gain, an extra short match, all
secondary protections, seeds 43/44 confirmation, official replay, full-build
reproducibility and independent review.
[Historical promotion requirements](/Users/derekko/Projects/more-than-a-hack-2027/experiments/PAIR_MOTION_PLAN.json).

This experiment changes representation, missingness training and supervision
scale together relative to earlier failed public-only neural work; its paired
control specifically isolates the extra value of temporal observations within
that new family. It does not explain every historical failure or guarantee
substantial improvement. Reusing the same small public possession universe is
adaptive development, not private or full-match accuracy evidence.
