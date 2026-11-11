# Detector observations as possession features: rejected screen

All six new head/decoder variants fail the frozen promotion gates. No candidate
is promoted. Selected8 remains SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
This round adds30 dependent possession evaluations, including3 selected references
and9 reused no-vision controls, bringing the ledger to1,321 rows:778 inherited
and543 added during this chat. These are not independent experiments.

## Frozen hypothesis and local observations

[DETECTOR_FEATURES_PLAN.json](DETECTOR_FEATURES_PLAN.json), SHA256
`45af653d80ea079f65811da743458392e6489e97d016837515d45cf0871fb84b`,
was frozen before public-window media extraction or fitting. It treats pretrained
detector outputs as fallible predictors. The earlier point-localized weak gates
remain failed. Cross-model agreement is not corrected ball truth.

The same two pinned public YOLO11 models from the preceding modern detector round
run locally on CPU. There are1,321 unique256-pixel crops and2,642 first-generation
model/image inferences. Each crop is resized to384square with Lanczos and padded
to384×640 with RGB114. Original tracking arrays remain unchanged. Centers use
original ballxy or bounded same-half interpolation with both endpoints within25
frames. No one-sided extrapolation, target-driven sampling or error-frame patch
is used. All24 reserved windows and60 held weak-confirmation centers have250-frame
image exclusions. No images or private detections are displayed or uploaded.

Within the public interval there are1,313 valid crop samples and1,146 consensus
samples, passing the predeclared250/100 support minima. These are observability
checks, not recognition-accuracy gates. Candidates require confidence≥.25 and
mutual-nearest model centers within8source pixels. Every no-detection state stays
recorded. Nominal ground geometry uses the original homography without replacing
the supplied ball coordinates.

## Feature and fitting comparison

The presence view has40 columns; geometry/context has93. They include candidate
counts/scores, detector agreement, image offsets, nominal nearest-team geometry,
near-person-foot evidence and fixed past/future means. Dense interpolation is
limited to consecutive5-frame crop samples with finite endpoints. Summaries stay
within official halves. The maximum observation footprint is155frames, inside the
250-frame target exclusion. Actual18935/19060 public feature and crop-center
constructions agree exactly. No target labels are read during preparation.

Both new heads append their view to the existing102 ordered+68 broadcast inputs.
Training uses unchanged frame-balanced, team-symmetric depth3/160-tree fitting,
seed42, eligible labels only, no calibration or early stopping. There are26 new
checkpoints over13 folds. The no-vision control reuses the verified broadcast joint
head with the same original170-column view and fitting hyperparameters. It is
scored in the same three decoders, without new control fitting.

| Head / decoder | Buffered minutes | Purged spells | Buffered100s |
|---|---:|---:|---:|
|Selected8|.741176|.685714|.670270|
|No vision / raw replacement|.556098|.590000|.543590|
|No vision / live replacement|.588235|.533333|.482353|
|No vision / D trim|.731429|.688889|.652632|
|Presence / raw replacement|.621053|.631579|.588889|
|Presence / live replacement|.606061|.564706|.494118|
|Presence / D trim|.700000|.688889|.652632|
|Geometry / raw replacement|.594595|.616216|.588889|
|Geometry / live replacement|.548571|.511111|.529412|
|Geometry / D trim|.694118|.685714|.652632|

Every cell is seed42 spellF1. All30 official scores agree with the shared scorer.
No variant meets the original+.04 buffered gain or extra short IoU.5-match
criterion. All six fail at least one protocol protection; there are43 primary
regressions among78 clipped-block comparisons. No seeds43/44, full-match fitting
or full candidate generation follow this failed screen.

The presence raw head improves over the reused raw-slot control, but remains well
below selected8. This is not a gain over the current solution. Its D-only trim
reduces buffered falseD88→12, purged142→12 and longer16→0, while D recall falls
.748128→.646698, .784888→.668482 and .649421→.640572. Buffered MacroF1 falls
.870833→.850340. Preserving some frames from each stoppage is insufficient: the
real dead-ball losses violate the recall/accuracy protections. Geometry trim has
similar losses. The selected8 D predictions are retained.

Raw short diagnostics retain the71-frame away miss and19-frame home problem:
both new heads have0 argmax-correct frames in those intervals. Presence correctly
classifies23/33 of the33-frame home interval at the raw-head level; this does not
produce an extra decoded short match. Diagnostic probabilities are saved locally
as aggregates, not corrected labels or training-set validation.

## Verification and retained artifacts

The fifteenth before snapshot preserves source, selected8 and reference OOF files.
Every preceding ledger byte is preserved; the30 verified rows append once, with
an idempotent duplicate check. Selected models, probabilities and candidate files
remain exact. No earlier failed plan or artifact is overwritten.

All raw native replies reparse exactly into canonical predictions. The complete
crop schedule is reconstructed from source observations and exclusions. Three
freshly decoded crops match PNG bytes; both models reproduce sampled predictions
exactly after reload. All26 new and13 existing control checkpoints reproduce their
complete saved OOF probabilities exactly. Eligible-only targets/weights,250-frame
exclusions, feature schemas and shared decoder outputs verify.

All150 tests pass, including bounded brackets, mutual matching, missing-crop
handling, team symmetry, exact context parity and append-only ledger protection.
A separate workspace regenerates every feature/cache file byte-for-byte from
saved detector observations, reloads all39 head/control checkpoints and reproduces
official metrics and acceptance checks. It has no current-round baseline copies
or archives. This reuses saved upstream observations/fitted artifacts; it does
not claim a second complete2,642-image detector run or new model refitting.

The first isolated head audit omitted declared upstream manifest dependencies.
Its failure log and original workspace script remain. The fixture now links the
required logical input identities from the verified manifests, including only
public GT. Feature/model/scientific parameters are unchanged.

Evidence under `data/control_detector_features_v1/` includes observations,
private crops/native results, prepared views,26 checkpoints,30 score rows,
training indices/weights, official reports, replay checks, raw short diagnostics,
block deltas, source snapshots, fresh parity and tests. No private/corrected GT,
manual annotations, commits, uploads, submissions, publication or external contact
occurred. Maximum score and private full-match accuracy remain unverified.

## Reproduction

Use the pinned Python3.12 environment, native macOS CoreML/Vision/JXA and the
verified upstream selected8/broadcast observation artifacts. In fresh stage
output directories:

```sh
.venv/bin/python experiments/prepare_detector_observations.py
.venv/bin/python experiments/finalize_detector_observations.py
.venv/bin/python experiments/prepare_detector_features.py
.venv/bin/python experiments/research_detector_features.py
.venv/bin/python experiments/finalize_detector_features.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
```

Resume observation preparation only after its previous process is confirmed
stopped. Optional finalizer `--record-ledger` appends verified rows once;
`--audit-history` separately verifies protected references. The current selected
production entry remains [SELECTED_MODEL.md](SELECTED_MODEL.md).
