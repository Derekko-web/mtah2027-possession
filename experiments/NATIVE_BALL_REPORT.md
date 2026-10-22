# Native pretrained ball-detector feasibility

Native Core ML inference now works without new Python dependencies or Swift
compilation. However, all six frozen pretrained-detector/crop combinations fail
the weak-data gates. No confirmation images, public possession heads or full
candidate were produced. The selected eight-model CSV remains SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
The possession ledger remains byte-identical at 1,291 rows: 778 inherited and 513
added in this chat. Six native proxy trials are recorded separately, alongside
the preceding twelve weak appearance trials. Reproduction is not extra evidence.

## A different source of supervision

The preceding weak image classifiers did not pass feasibility. This round tests
two externally pretrained 80-class detectors from
[Apple's official Core ML model gallery](https://developer.apple.com/machine-learning/models/),
YOLOv3Tiny and YOLOv3. Their source/author and licence metadata point to
[Darknet](https://github.com/pjreddie/darknet). No competition fitting,
fine-tuning, calibration or new annotations are used.

The exact public artifacts are retained under
`data/control_native_ball_v1/model_assets/`:

| Model | Downloaded bytes | Raw model SHA256 |
|---|---:|---|
|YOLOv3Tiny|35,527,626|`ed41d7f0c1652994944b4e3578b907f8634a98c2f3feaf33eab8f1c67df44e3f`|
|YOLOv3|248,381,690|`7fea1f2b83b303e7fc6acf08c05a062793229de474587da336d8724e5bf16e38`|

Canonical download/compilation is `prepare_native_models.py`; the official
URLs, metadata and checksums remain recorded. External weights, compiler caches,
all competition crops and output arrays stay under ignored `data/`. No
competition image, pixel array, tracking extract or model scene description was
displayed or uploaded to a hosted model. Only aggregate metrics were inspected.
No repository licence, global toolchain setting or Xcode licence agreement changed.

## Native runtime and synthetic failures

The Swift entry point is blocked by an unaccepted Xcode licence. The existing
JavaScript for Automation bridge can import CoreML/Vision and compile/load these
models through the installed OS frameworks. `native_detector_bootstrap.js` and
`native_detector_predict.js` use local files and CPU-only configuration.

The initial synthetic inference failed with a bridge TypeError. An empty-dictionary
change did not fix it. A minimal request reproduced the error with a nullable
completion-handler argument, while the callback-free initializer succeeded.
The production bridge therefore uses `initWithModel`; both failed sources/logs
and the minimal diagnostic remain in `synthetic/`. This repair preceded all
competition-image inference.

Blank controls return no objects. Repeated synthetic requests produce exact
objects/confidences. Tiny recognizes the drawn ball and correctly localizes an
off-center control, supporting the bottom-left Vision box conversion. Full YOLO
calls the drawing a clock and misses the off-center drawing. The initial
class-specific assertion for both models was overconstrained; its failure is
retained. Runtime correctness and football accuracy are separate questions, so
Full proceeded to weak evaluation without a claimed synthetic ball-positive result.

## Frozen weak protocol

[NATIVE_BALL_PLAN.json](NATIVE_BALL_PLAN.json) freezes both models and square
crop sides 16/32/64 pixels, before any competition input entered this route.
Each crop is centered on a projected candidate point, remains unclipped and is
upsampled to 416×416 with Lanczos. Native NMS uses confidence floor .01 and IoU .5.
The detection decision uses confidence .25, following the
[author's documented default](https://pjreddie.com/darknet/yolo/).

The target score is the maximum `sports ball` confidence whose center is within
`max(4,min(12,2*expected_diameter))` source pixels of the projected point.
Expected diameter comes from the existing .22m/Jacobian calculation clipped
to 2..12 pixels. The unlocalized maximum is a diagnostic only. No threshold or
localization tolerance changes after observing scores.

The screen reuses the preceding 60 weak validation centers, selected outside the
public and all 24 reserved windows with 250-frame buffers. Each has one tracked
ball positive and off-ball ground/footwear negatives from the existing algorithm.
There are 60 positive and 349 negative valid crops per size, 1,227 unique crops
and 2,454 first-generation model/image inferences. No model is fitted to these
images. This remains adaptive same-match agreement with fallible tracking
proxies, not independently corrected ball truth or possession accuracy.

The first passing variant in the declared Tiny16/32/64, Full16/32/64 order would
be frozen for a separate 60-frame confirmation. Its metadata was selected from 77
eligible candidates with seed 43 before screen decoding, excluding every prior 220
sample center plus 250 frames and all public/reserved buffers. The schedule
SHA256 is
`a24b3e73b99cd508b22ae71476b4cb37b42246e688f91f1066f108858069ca83`.
The schedule regenerates exactly from code; none of those images was decoded.

Gates require at least 40 positive/120 negative valid crops, positive validity≥.5,
balanced weak accuracy≥.80, and confidence≥.8 coverage≥.50/agreement≥.90.
Predicted positives use their score as confidence; negatives use 1-score. Thus
confidently calling every crop negative cannot pass. Every variant meets the
count/validity requirements and fails accuracy.

## Results

| Model | Crop side | Balanced accuracy | Positive recall | Negative false-positive rate | Confident coverage | Confident agreement |
|---|---:|---:|---:|---:|---:|---:|
|Tiny|16|.50000|.00000|.00000|1.00000|.85330|
|Tiny|32|.51523|.03333|.00287|.99022|.85679|
|Tiny|64|.60377|.23333|.02579|.95844|.88010|
|Full|16|.50000|.00000|.00000|1.00000|.85330|
|Full|32|.52500|.05000|.00000|.99022|.86173|
|Full|64|.66927|.35000|.01146|.97311|.89196|

The strongest variant matches 21 of 60 weak positives at the prescribed location.
It reports a ball somewhere in the crop on 32 of 60 positives, but those additional
detections are not verified correct balls or corrected locations. Tiny64's
unlocalized positive recall is .40. Low-resolution appearance, context and weak
localization errors remain confounded. Neither high negative agreement nor a
working synthetic interface establishes a useful field detector.

No variant passes, so no fresh confirmation, public video feature extraction,
possession fitting or promotion follows. The original substantial +.04 buffered
and additional short-IoU .5-match criteria remain untouched. Maximum score and
private full-match accuracy remain unverified. A newer or football-specific
external detector would be a distinct next hypothesis; these results do not
rule out video methods generally.

## Compiler-cache repair and complete reproduction

Independent compilation reproduced Tiny's cache exactly. Full changed 142 bytes
in the 1,377-byte `coremldata.bin`; all weights/network files remained identical,
and synthetic predictions were exact. A globally frozen checksum for generated
cache metadata would prevent an otherwise valid fresh compilation.

Before repair, `provenance_before/` preserved the entire first source tree,
frozen plan, crops, native outputs, arrays and scores, plus the original Full
metadata bytes and asset manifests. The plan's explicit provenance revision keeps
its original scientific configuration. Raw public model checksums remain pinned;
the local asset manifest now binds each compiler generation to those raw models
and compiler-adapter source. Historical cache hashes are retained as evidence.
No crop, model parameter, detection threshold or acceptance gate changes.

A complete fresh workspace then used the independently compiled caches and
rebuilt all six comparisons. Every one of the 1,227 PNG crops, six canonical
prediction files, all score-array files and the score CSV matches byte-for-byte.
The canonical generation uses those reproduced outputs; all first-generation
artifacts remain. Timings are retained in run logs, excluded from deterministic
prediction identities. `provenance_parity.json` records the complete comparison.

The final audit reparses all raw native outputs and regenerates every proxy-score
array/metric. It verifies sampled model reloads, crop geometry and all hashes,
and recomputes all crops at three freshly decoded centers exactly. It regenerates
the untouched confirmation schedule. All 135 tests pass, including coordinate
conversion, near/far filtering, malformed result rejection, majority-class gates,
compiler-cache corruption and raw-source mismatch checks. Ordinary replay also
passes without historical protection files, regenerating the schedule rather
than copying session metadata.

The possession ledger remains byte-identical at 1,291 rows. Selected 8 models,
probabilities, CSV and manifest remain unchanged. No commits, competition-data
uploads, submissions, publication or messages to others occurred. All24 reserved
windows remain unlabelled and unused; their metadata SHA remains
`5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c`.

## Reproduction

This experiment requires macOS with native CoreML/Vision/JXA, the existing
Python 3.12 environment and local competition assets. First reproduce the original
centered weak-cache stage in
[BALL_APPEARANCE_REPORT.md](BALL_APPEARANCE_REPORT.md); pooled/temporal models are
not prerequisites. Preserve the original frozen reserved-window metadata.
With fresh native screen directories:

```sh
.venv/bin/python experiments/prepare_native_models.py
.venv/bin/python experiments/prepare_native_schedule.py
.venv/bin/python experiments/prepare_native_ball.py
.venv/bin/python experiments/finalize_native_ball.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
```

Completed extraction/inference can resume with `--resume` only after its previous
process is confirmed stopped. Ordinary finalization needs no source archive or
baseline candidate. Optional `--audit-history` verifies protected starting files,
the original scientific plan and unchanged first-generation results. The
`confirmation` stage correctly rejects this screen because no variant passed.

Local evidence under `data/control_native_ball_v1/` includes model sources and
metadata, synthetic successes/failures, all six score rows, request/output logs,
1,227 private crop files, per-variant probability arrays, both compiler
generations, preservation/parity evidence, tests and audits. These six weak
trials are not added to the possession-score ledger.
