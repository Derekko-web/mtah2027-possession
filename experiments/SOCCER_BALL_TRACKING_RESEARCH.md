# Football-specific pretrained ball localization as a conditional fallback

Prepared 2026-10-25 while the separately frozen Round 36 masked-player models
run. This note does not change or prejudge that experiment. Only this new note
was written. No existing file/data/model/score/selection changed; no competition
media, tracking or target arrays were opened, weights/videos downloaded,
packages installed, fits/scores run or external contacts made. Public source,
papers and metadata were read. A Google Drive HTML view and a HEAD request were
used to verify an advertised artifact; its binary body was not fetched.

## One credible fallback, with explicit unresolved gates

If Round 36 fails, the strongest source-backed visual fallback found is **one
bounded no-fit feasibility audit of the official WASB soccer checkpoint**. It
has football-specific training and a multi-frame small-ball heatmap interface.
It is genuinely different from another generic YOLO model/crop size. This is a
recommendation to verify an accessible research artifact and its local runtime,
not a claim of better native ball localization or possession accuracy.

Official source and an openly reachable soccer weight artifact are verified.
The code license is verified MIT. A separate explicit license statement for the
Drive-hosted weight binary was not found in the reviewed material; do not simply
label that binary or its source videos unrestricted MIT. The original training
data expressly permit non-commercial research only. Preserve these distinctions
in the prospective source/usage manifest before any acquisition or use.

## Why the previous visual experiments do not answer this hypothesis

The old native YOLOv3/Tiny and modern COCO YOLO11s/m crop comparisons failed weak
location gates; the best modern balanced weak accuracy was .653773. At larger
crops the generic detectors often found some ball elsewhere, with frequent
positive detections on negatives. Those observations did not establish corrected
ball coordinates. Adding their presence/geometry features to possession heads
also failed, and detected metadata contacts regressed while color-fused proposals
made no changes. Median/robust XY repair likewise did not supply missed control
information.
[NATIVE_BALL_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/NATIVE_BALL_REPORT.md),
[MODERN_BALL_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/MODERN_BALL_REPORT.md),
[DETECTOR_FEATURES_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/DETECTOR_FEATURES_REPORT.md),
[COLOR_CONTACT_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/COLOR_CONTACT_REPORT.md),
[CONTACT_DYNAMICS_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/CONTACT_DYNAMICS_REPORT.md).

WASB instead predicts dense ball-position heatmaps from three consecutive full
frames, using a high-resolution network trained with precise ball centers.
Its paper adds position-aware hard-sample training and temporal inference.
The soccer weights were trained on four of six synchronized fixed-camera views
of one short game segment, with revised manual ball annotations. The remaining
two cameras form its test split: this is cross-view evaluation of synchronized
play, not unseen-match broadcast validation. Ball-location benchmark F1 is not
team-possession spell F1. The paper also reports no soccer benefit from its
online tracking addition, so a first audit can focus on the bounded multi-frame
detector rather than assume recursive tracking is needed.
[Primary WASB paper, §§3–5](https://arxiv.org/html/2311.05237v1).

## Verified source, artifact and provenance

- Official project: [NTT Communications WASB-SBDT](https://github.com/nttcom/WASB-SBDT).
- Pinned commit: **`923462cacdeb3353b84ddebdedb3f4b7a8553b0f`**, committed
  2023-11-23; obtained from the official GitHub API.
  [Commit metadata](https://api.github.com/repos/nttcom/WASB-SBDT/commits/923462cacdeb3353b84ddebdedb3f4b7a8553b0f).
- One artifact, prospectively: **`wasb_soccer_best.pth.tar`**, Google Drive ID
  **`1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l`**. The pinned official model zoo links
  this artifact in its Soccer/WASB cell.
  [Pinned model zoo](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/MODEL_ZOO.md),
  [publisher-linked weight page](https://drive.google.com/file/d/1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l/view).
- A metadata-only HEAD to the Drive download endpoint returned HTTP 200,
  `application/octet-stream`, attachment filename above and **6,102,633 bytes**.
  [Artifact endpoint](https://drive.usercontent.google.com/download?id=1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l&export=download).
  No publisher SHA256 was located; actual bytes, local SHA256 and tensor content
  remain unverified here. Mutable Drive content must be bound to a new local
  source manifest if acquired later.
- Code: [MIT license, NTT 2023](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/LICENSE.md).
  The HRNet source separately retains Microsoft MIT notices; preserve those.
- Training video provenance: the ISSIA-CNR/FIGC-CNR dataset author provides six
  synchronized Full-HD 25 Hz camera views, with ball/player/referee annotations.
  The author page explicitly restricts dataset use to non-commercial research
  in video signal processing and requests the 2009 source citation.
  [Original dataset and terms](https://pspagnolo.jimdofree.com/download/).
- WASB's setup script links the original video archive and **its reannotated**
  `soccer_annos.zip` on Drive. Do not use the original annotations as though
  they were the revised ball labels used for this checkpoint.
  [Soccer setup/provenance script](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/setup_scripts/setup_soccer.sh).

The model zoo also has soccer-trained reimplementations of TrackNetV2,
ResTrackNetV2, DeepBall and BallSeg. They are not additional candidates in this
proposal. In particular, original TrackNetV2 badminton weights and newer
table-tennis BlurBall weights are not football-specific alternatives merely
because they share architecture names. No checkpoint/model sweep is proposed.

## Exact tensor interface and local feasibility

The released WASB config selects HRNet, `frames_in=frames_out=3`, 512×288 input
and output, RGB differences disabled and output scale `[0]`. Stem strides are
`[1,1]`; the model retains high-resolution feature maps. The final channel count
is three ball heatmaps. Use the complete pinned config; do not substitute a
generic ImageNet HRNet checkpoint or change input resolution after a failed
runtime gate.
[Pinned model config](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/configs/model/wasb.yaml).

`HRNet(cfg_model)` accepts `[B,9,288,512]`: three chronological RGB tensors
concatenated in channel order. Forward returns a dictionary with `out[0]` shaped
`[B,3,288,512]`, before sigmoid. The network file itself imports only Torch and
standard-library logging/os. A tiny local attribute-access dictionary can supply
the config's mixed `cfg['...']` and `cfg.MODEL.EXTRA` access; no Hydra/OmegaConf
installation is necessary. The stock detector loads
`checkpoint['model_state_dict']`; later acquisition must inspect this with
`weights_only=True`, verify tensor shapes/keys and load strictly on CPU. Actual
serialization compatibility remains untested.
[HRNet source](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/models/hrnet.py),
[detector/loading source](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/detectors/detector.py).

The unmodified detector wrapper explicitly rejects CPU and asserts CUDA. That
is a wrapper restriction, not evidence of a CUDA-only operator in HRNet. Bypass
that wrapper with a documented minimal adapter, preserving all network math and
weights. Do not import the package-wide model factory unnecessarily: it imports
other models and dependencies unrelated to this path. The official tested
environment is Python 3.8/CUDA 11.3 with V100s; no official Apple CPU/MPS result
was found. Torch 2.5 CPU execution is plausible but must be measured; small
parameter count does not make full-resolution convolution cheap.
[Official setup/environment](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/README.md).

Preprocessing uses one shared affine image transform across the three frames,
preserving scale with the image center mapped to the network center. Pixel
values become RGB float [0,1], then ImageNet normalization with means
(.485,.456,.406), standard deviations (.229,.224,.225). The source uses OpenCV
bilinear affine warping, not a stretched crop or arbitrary PIL resize. Preserve
that convention and its inverse when returning pixel centers. If the existing
runtime lacks OpenCV, a Torch/Pillow/NumPy adapter is possible but its sampling
and coordinate parity need explicit synthetic checks before any real images;
do not silently install another dependency or claim exact source preprocessing.
[Data preprocessing](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/dataloaders/dataset_loader.py),
[normalization](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/dataloaders/__init__.py),
[affine geometry](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/utils/image.py).

The shipped heatmap postprocessor applies sigmoid, threshold .5, eight-connected
components and heatmap-weighted centers; component confidence is summed heatmap
mass, not a calibrated probability. It maps centers through the inverse affine.
The .5 threshold concerns heatmap pixels, not the old YOLO .25 detection rule.
Synthetic blob/mapping controls can verify a local NumPy/available-SciPy version
without needing a new package.
[Detector defaults](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/configs/detector/tracknetv2.yaml),
[postprocessor](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/detectors/postprocessor.py).

Avoid stock recursive tracking in the first bounded audit. The released online
tracker filters by distance from the previous accepted location, while its
`xy_pred` in candidate scoring remains None; it does not actually invoke the
advertised predictor there. Accepted locations can therefore depend recursively
on old history. Both tracker implementations also use `np.Inf`, incompatible
with NumPy 2's removed alias. A documented `np.inf` compatibility substitution is
minor; changing tracking logic to match the paper would be a new method.
[Online tracker](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/trackers/online.py),
[intra-frame selection](https://raw.githubusercontent.com/nttcom/WASB-SBDT/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/src/trackers/intra_frame_peak.py).

## One bounded feasibility audit, only after a separate freeze

1. **Source/runtime gate first, no competition processing.** Pin the one commit,
   checkpoint ID, config, license/terms evidence and adapter before acquisition.
   Retain binary identity and all failed evidence. Require safe strict tensor
   loading, finite `[1,9,288,512]` forward, identical CPU reload/replay, affine
   forward/inverse controls and exact blob-selection semantics. Benchmark the
   real shape, preprocessing and postprocessing on synthetic tensors; project
   both audit arms' complete workload and memory against a predeclared budget
   (e.g. <=60 minutes and <=2 GB). CPU is the reference. MPS may be tested locally
   with fallback disabled, but use it only if frozen numerical/detection parity
   and resource checks pass. No MPS feasibility is claimed by this research.

2. **One temporal detector pair.** Candidate receives adjacent native video
   frames `[t-1,t,t+1]`; control repeats `[t,t,t]`, with the same frozen checkpoint
   and postprocessor. Keep only output channel 1 for center t, so visual reach
   is one frame each side. This is a mechanistic inference ablation, not a
   separately trained matched model. Use full images at the published input
   size, independent of the existing tracked ball point. No crop sweep,
   target-centered frame selection, recursive track, trajectory repair or
   confidence tuning is included. Respect cuts, missing images, half boundaries
   and existing sealed/held-image exclusions; unsupported samples stay missing.

3. **External localization sanity if eligible source data are available.** Use
   revised WASB ball centers and visibility on a fixed subset of cameras 5/6,
   with source preprocessing and a frozen point-distance rule. A reasonable
   prospective continuation threshold is F1>=.80 at four original-image pixels
   and positive visible recall>=.75. Report candidate/control and all invisible
   false positives. These are suggested gates, not expected measured results.
   That same-play camera split is a checkpoint/adapter sanity test, not fresh
   unseen-game validation. Stop if source labels/usage/access cannot be verified;
   no training on unavailable or unlicensed data is a fallback inside this audit.

4. **Raw native opportunity before any public fitting.** Only after the earlier
   gates pass, freeze uniform public sampling and one existing image-contact
   ownership readout. Produce immutable detections before joining possession
   targets. Compare both detector arms and the unchanged original observation
   readout across all 17 possessions, all six shorts, all four misses and every
   spurious case. Require the inherited correction precision/net-correction
   conjunction, one common extra short IoU .5 case, no previous match lost and
   no extra spurious prediction before considering one separately frozen
   integration. D/X remain exact. The earlier weak point-agreement failures stay
   failures; agreement with a fallible original ball track is a diagnostic, not
   corrected ball truth. Do not repeat the failed broad detector-feature screen
   merely because the new network returns nonempty detections.

A detected airborne ball is not automatically a ground-plane point or physical
control. Existing homographies do not resolve height; nearest-player geometry
and team tags can still be wrong. Stronger localization needs to supply useful
incremental ownership evidence, not just a prettier path. No additional manual
annotation is required by this proposed audit, but that limits independent
native ball-localization verification.

## Public-label alternatives and benchmark distinctions

ISSIA-CNR plus WASB's revised centers is a real public soccer-ball supervision
route, subject to its research terms, but only a short synchronized segment.
Retraining from it is not the proposed fallback: pretrained evaluation is much
cheaper and avoids another unvalidated CPU training recipe.

**SoccerNet-Tracking contains real ball annotations**, alongside players and
referees. Its official MOT-style files/evaluation combine object classes, so a
high aggregate HOTA or released person-tracking detector is not verified
ball-specific accuracy. Confirm associated class/identity metadata before
claiming it is a ready ball-detector training table.
[Official tracking task/data format](https://github.com/SoccerNet/sn-tracking#task-description).

By contrast, the established **SoccerNet Game State Reconstruction** task lists
player/goalkeeper/referee/other individuals and their roles, teams and jerseys;
that description does not supply ball XY/bounding-box supervision. Ball Action
Spotting supplies temporal actions, not necessarily framewise ball locations.
Do not substitute either task's popularity for verified ball labels.
[Official GSR definition](https://www.soccer-net.org/tasks/game-state-reconstruction).
SoccerNet's current official data page also distinguishes public downloads from
gated protected video requiring account/NDA acceptance. No new acceptance,
authentication or data acquisition was performed here.
[Official data access](https://www.soccer-net.org/data).

The fallback therefore exists as an author-released, reachable **soccer-trained
model**, with clear code provenance and limited training-domain evidence. Its
separate binary terms, actual safe loading, CPU/MPS budget and native incremental
signal still need to pass. It is not a claim that fully unrestricted pretrained
football tracking, corrected native ball truth or substantial possession gain
has already been obtained. The +.04 buffered spell-F1, extra short match,
secondary protections, confirmation seeds, replays and independent review remain
unchanged for any eventual candidate.
