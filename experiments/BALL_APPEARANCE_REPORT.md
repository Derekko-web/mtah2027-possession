# Local ball-appearance feasibility

All twelve weak ball-appearance models fail the frozen feasibility gates. No
public possession head was fitted, no public possession score row was added,
and no new full candidate was built. The selected eight-model candidate remains
unchanged, SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
The ledger remains exactly 1,291 dependent possession evaluations: 778 inherited,
513 added in this chat. The twelve weak proxy evaluations are recorded separately.
They are not possession-score improvements or independent ball ground truth.

## Starting point and hypothesis

The prior directional/weighting rounds did not fix the short possessions.
The next hypothesis tested an independent local visual observation: appearance
support at a tracked ball location, then translation-tolerant shape statistics,
then pitch-compensated temporal change. The earlier local Ollama live/stopped
probes were different questions and had already failed; no local LLM service was
started or reused. NumPy, Pillow, SciPy, sklearn and ffmpeg were already installed.
No new dependency or model download was needed.

Before this round, `data/control_ball_appearance_v1/` received the entire source
archive, selected entry, 1,291-row ledger, selected8 candidate and confirmation,
with protected hashes. The usual short feedback was rerun and still failed:
3 short IoU≥.5 matches, required4. It was not used as an inference input.

The three immutable plans were frozen sequentially before their corresponding
feature extraction/fitting:

1. [BALL_APPEARANCE_PLAN.json](BALL_APPEARANCE_PLAN.json): centered patch models.
2. [BALL_POOLED_PLAN.json](BALL_POOLED_PLAN.json): frozen after the first four
   failures; spatial pooling, with unchanged gates.
3. [BALL_MOTION_APPEARANCE_PLAN.json](BALL_MOTION_APPEARANCE_PLAN.json): frozen
   after the next four failures; new three-frame evidence, unchanged gates.

These are adaptive weak-label follow-ups. The same60 weak validation frames are
reused openly. Each later plan requires a fresh separated weak confirmation if
any method passes the old sample. None does, so that new sample was not consumed.
No detection threshold or score criterion was relaxed to pass a failed model.

## Weak sample and privacy boundary

Only tactical video is decoded, locally at full1920×1080 resolution. Its25fps,
zero-start-time and199319-frame contract matches ffprobe. No image, crop, pixel
array, tracking extract or visual scene description was displayed or sent to a
hosted model. Only aggregate diagnostics were inspected. Numeric caches,
checkpoints and predictions remain under ignored `data/`.

Candidate centers lie on a25-frame grid inside official half bounds. The public
11185..18684 window plus250 frames and every reserved annotation window plus250
are excluded. A weak positive needs a finite ball and±5-frame ball observations,
both directional speeds1..35m/s, at least3m from every observed player, at least5m
from goal lines and3m from touchlines, and valid homography with reprojection
error≤1. These are automatic moving-track proxies; they do not establish that
a real ball is visible at that pixel.

There are750 eligible grid centers. Seed42 selects60 validation centers, then
removes training candidates within250 frames of any validation center. Of450
remaining candidates,160 are selected for fitting. Training uses the center and
four±1-pixel jitters as positive crops; validation uses only the center. Negative
crops are projected±4m ground offsets and the two closest eligible off-ball
player feet at least4m away, providing hard footwear negatives. Invalid/outside
crops stay missing. These augmentations are not extra independent frames.

The temporal phase uses center±5 source frames. All660 distinct sampled frames
also lie outside public and reserved250-frame buffers. Every sampled temporal
homography is finite, valid and invertible. No private/corrected possession labels,
manual annotations or final-frame patches are used. All24 reserved windows remain
unlabelled and unused; metadata SHA stays
`5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c`.

## Three representations

The homography maps a normalized pitch point to an image center. Expected ball
diameter is .22m times the largest local Jacobian singular value, clipped to2..12
pixels. Two fixed profiles use halfwidth one or two diameters. Crops are resampled
to16×16. Location, identity, time and scale are not classifier features.

Centered features have778 dimensions: RGB residual relative to the outer-ring
median, plus ring background, center color mean/std and nongrass fraction.
Raw-model fitting uses1749/1748 crops for the two profiles. There are409 valid
validation crops in each:60 positive,349 negative,100% positive-crop validity.

Spatial pooling reduces the same cached patches to103 features: robust channel
statistics, Gaussian differences and Hessian compact-blob responses at three
fixed scales, peak summaries and centered spatial covariance. It uses no new
images. A pooled synthetic test distinguishes an isotropic spot from a line.
A diagnostic over the cached crops showed stronger mean local
contrast in positive patches, motivating this follow-up without establishing
ball detection accuracy.

Temporal features use locally decoded previous/current/future frames. Each current
16×16 pixel grid is mapped through the inverse current homography and neighboring
homographies, so all three patches represent the same ground-plane location.
After subtracting their separate ring backgrounds, current minus mean-neighbors
measures change. Concatenating103 appearance and103 temporal pooled inputs gives
206 features. Synthetic tests cancel stationary field markings and camera
translation while retaining a moving spot. There are1747 fitting crops per
profile; validation still has all409 crops. This does not independently verify
the real ball or solve airborne geometry.

Each representation/profile fits the same two models: training-only scaling plus
balanced logistic regression C1, and a200-tree random forest, depth10, minimum
leaf5, balanced bootstrap classes. Seed42 and one forest thread make predictions
repeatable. All four models in a family are retained; selection is the first
passing model in declared profile/model order.

## Results and decision

Every model must have at least40 positive and120 negative validation crops,
positive validity≥.5, balanced weak accuracy≥.80, and confidence≥.8 coverage≥.50
with agreement≥.90. Saved checkpoints must reload exactly. Count/validity gates
pass for every model; none passes the accuracy gate.

| Features | Crop halfwidth | Model | Balanced proxy accuracy | Confident coverage | Confident proxy agreement |
|---|---:|---|---:|---:|---:|
|Centered|1 diameter|Logistic|.54565|.35452|.68276|
|Centered|1 diameter|Forest|.58097|.09046|.94595|
|Centered|2 diameters|Logistic|.57951|.55012|.68444|
|Centered|2 diameters|Forest|.61313|.14425|.96610|
|Pooled|1 diameter|Logistic|.63173|.27139|.90991|
|Pooled|1 diameter|Forest|.63723|.11980|.97959|
|Pooled|2 diameters|Logistic|.69971|.42787|.89714|
|Pooled|2 diameters|Forest|.73711|.32274|.97727|
|Temporal|1 diameter|Logistic|.61156|.38631|.86076|
|Temporal|1 diameter|Forest|.65155|.16870|.94203|
|Temporal|2 diameters|Logistic|.72498|.52812|.91204|
|Temporal|2 diameters|Forest|.70781|.33985|.97122|

Pooling helps this weak task, but its best model remains below.80 and has low
confident coverage. Temporal logistic passes the coverage/agreement gates but
not balanced accuracy. High agreement among a small confident subset therefore
does not pass the complete feasibility rule. No fresh confirmation, public
feature extraction, possession training or promotion follows these failures.

This does not show that video is unhelpful. It shows that these small weakly
supervised patch models do not supply a validated detector under the frozen
checks. Weak positive localization and model limitations remain confounded.
An independently trained local detector would be a distinct hypothesis; choosing
another blob threshold against these same60 weak frames would add little evidence.
No private full-match accuracy or maximum score has been established.

## Integrity and reproduction

The audit regenerates all twelve weak metric rows and exact checkpoint reloads.
It verifies fitting-only means/variances for all six scalers and regenerates every
cached pooled feature exactly. Three centered and three temporal sample feature
sets are recomputed from freshly decoded local images and match exactly. Source,
actual video/tracking/homography, plan and cache hashes pass. All129 tests pass.

Fresh-workspace replay succeeds without source archives, baseline candidates or
session-created protection files. Ordinary replay does not claim historical
checks; `--audit-history` separately verifies the preserved ledger/entry archive,
selected source files, candidate and model/probability artifacts. The1291-row
ledger is byte-identical to the protected start. The selected entry only gains
a link to this research; its trained artifacts do not change. No commits,
competition-data uploads, submissions, publication or messages to others occurred.

The added historical source audit initially assumed every selected source lived
under `experiments/`. That failed on the tracked `games.py`; the failed audit and
source are preserved. The repair resolves actual source paths, using the archive
for untracked experiment sources and fixed Git commit `3de6b60` for tracked ones.
It changes no feature, model, plan or metric. Ordinary fresh replay remains
independent of this optional history branch.

Use the existing Python3.12 `.venv` and pinned requirements, with the local assets
and unchanged reserved-window metadata available. In fresh stage directories:

```sh
.venv/bin/python experiments/prepare_ball_appearance.py
.venv/bin/python experiments/research_ball_pooled.py
.venv/bin/python experiments/prepare_ball_motion_appearance.py
.venv/bin/python experiments/finalize_ball_appearance.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
```

The two extraction scripts support `--resume` for a confirmed stopped process;
do not restart a live extraction. With completed caches present, the finalizer
replays evidence without fitting. Add `--audit-history` only with the original
local source archive/protected manifest. Evidence under
`data/control_ball_appearance_v1/` includes all three family folders,12 models,
numeric feature caches, `proxy_inventory.csv`, plans, extraction/test/audit logs,
`temporal_context_check.json`, `fresh_reference_audit.json` and `integrity.json`.
Weak proxy results are deliberately outside the possession experiment ledger.

## Next local detector route

The installed Swift entry point cannot run because the Xcode licence is not
accepted; no licence agreement or global toolchain setting was changed. A separate
JavaScript for Automation probe successfully imports native CoreML and Vision.
This checks framework availability only, not model inference. The probe outputs
are retained under this round's data directory.

Apple's [official model gallery](https://developer.apple.com/machine-learning/models/)
lists downloadable object detectors, providing an inspectable, independently
trained model source for a later local experiment. No model was downloaded or
competition frame passed to that route in this round. Loading/inference and
football-specific utility remain untested; imported frameworks alone do not
establish a usable ball detector.
