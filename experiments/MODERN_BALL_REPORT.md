# Modern local detector follow-up

All six frozen YOLO11/crop comparisons fail the weak location gates. The strongest
balanced accuracy is .653773, below .80. No confirmation images, public possession
models or new full candidate were produced. Selected8 remains SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
The possession ledger remains exactly 1,291 dependent rows: 778 inherited and 513
added during this chat. These six weak proxy trials are separate from that ledger,
as are the preceding twelve appearance and six older native-detector trials.

## Hypothesis and public assets

The older native detectors failed. This round prospectively tests newer externally
pretrained models and larger image context, preserving image proportions. Both
packages are verified assets from the
[official Ultralytics iOS repository](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0).
The [official Core ML documentation](https://docs.ultralytics.com/integrations/coreml)
describes local deployment. The repository and model provenance identify AGPL-3.0;
no model or project distribution occurred.

| Model archive | Bytes | SHA256 |
|---|---:|---|
|`yolo11s.mlpackage.zip`|8,771,041|`ab85797270a693c6bad9f0eef6dc553190a882922e578557947bf0f7c3b3e712`|
|`yolo11m.mlpackage.zip`|18,442,294|`54187b2efb0a5e3a17af6ec9c16eac78e4112692e581cdfe39d5b95436cd9441`|

Release metadata, original archives, package members and compiled caches are saved
under `data/control_modern_ball_v1/model_assets/`. Archive checksums pin the actual
weights because release assets may change. Package-member hashes must match the
pinned ZIP exactly. Compiled caches have per-generation manifests bound to the
raw packages and preparation/bridge sources. Two independent compilations were
performed. Native CPU-only inference uses the existing JXA bridge; no new Python
dependencies, Swift compilation or global toolchain changes were needed.

The first preparation assumed the ZIP contained its package directory. It actually
contains bare `Manifest.json` and `Data/`, causing a compilation failure before
competition processing. The original log remains. Canonical preparation extracts
each archive inside a named `.mlpackage` and checks safe paths and all members.

## Prospective protocol

[MODERN_BALL_PLAN.json](MODERN_BALL_PLAN.json) was frozen before competition crops
entered this route, SHA256
`c433fc81a9ef186ea6aa93bf54417d4d3286e67374101c2326277df0eddc63d0`.
It tests both models on square source crops of side 64/128/256 pixels, in that
selection order with the smaller model first. The actual native input is
**384 pixels wide and 640 high**. Crops resize to 384×384 with Lanczos, then receive
symmetric 128-pixel top/bottom padding with RGB114. Detections are mapped back
through that padding before evaluating their distance to the projected point.
Images are never stretched. Unclipped crops only are eligible.

The native confidence floor .01, NMS IoU .5, decision .25 and original location
radius `max(4,min(12,2*expected_ball_diameter))` remain fixed. There is no fitting,
fine-tuning, threshold search, calibration or new annotation. Synthetic blank,
drawn-center and drawn-offset requests repeat exactly for both models; blank
outputs are empty. Unit controls verify padding, clipping rejection and inverse
box geometry. Synthetic detections do not establish field accuracy.

The same 60 weak validation centers are reused adaptively from the original
appearance round. They exclude public and all24 reserved windows with250-frame
buffers. Positive points come from the automatic ball track; negatives are the
original ground offsets and off-ball tracked feet. There are60 positive/349 negative
crops at side64,60/348 at128, and58/343 at256: **1,218 unique crops and2,436
first-generation model/image inferences**. Every condition passes sample-count
and positive-validity requirements.

The gates require balanced weak accuracy≥.80, confidence≥.8 coverage≥.50 and
agreement≥.90, with≥40 positive/120 negative valid crops and positive validity≥.50.
Only the first passing variant would proceed to the previously frozen60-center
confirmation. None passes, so that sample remains unused. Its metadata SHA stays
`a24b3e73b99cd508b22ae71476b4cb37b42246e688f91f1066f108858069ca83`.

## Results

| Model | Side | Balanced accuracy | Positive recall | Negative FPR | Confident coverage | Confident agreement |
|---|---:|---:|---:|---:|---:|---:|
|11s|64|.653773|.333333|.025788|.975550|.884712|
|11s|128|.653448|.350000|.043103|.980392|.880000|
|11s|256|.653463|.344828|.037901|.975062|.892583|
|11m|64|.632283|.333333|.068768|.985330|.848635|
|11m|128|.630460|.350000|.089080|.987745|.836228|
|11m|256|.644717|.344828|.055394|.977556|.877551|

No variant clears either accuracy or confident-agreement gates. This is no
possession improvement, and the original +.04 buffered/extra-short-match promotion
requirements remain unchanged. No model, decoder or final frame was patched.

## Location diagnostics and limits

A separate, read-only diagnostic runs after completion. It changes no thresholds,
scores or selection. At side256 both models report a ball somewhere in56/58 positive
crops (.9655 recall), but also in85.13%/88.34% of negative crops. Unlocalized balanced
accuracy is only .5571/.5411. Finding a ball somewhere in a large crop is therefore
insufficient evidence for the candidate point.

For the highest-confidence ball per positive crop, off-center detections have a
median vertical displacement of about−18 pixels in the128/256 conditions;70–78%
are above the projected point. These are aggregate detector disagreements, not
verified corrected ball coordinates. This statistic chooses the strongest ball
anywhere, whereas the primary score chooses the strongest qualifying nearby ball;
their counts can differ.

As a separate alignment control, the nearest detected person-box bottom center
on negatives coincident with tracked players is typically about2.5 pixels from
the projected tracked foot. At128/256 the median vertical displacement is within
about.32 pixels. This gives less support to a large vertical error shared by every
object, but person boxes and automatic tracking are also fallible. It does not
identify the cause of ball disagreement. Possible tracking errors, image ambiguity
and what the ball coordinate represents require further local investigation.
The failed score cannot by itself establish poor ball appearance recognition.

The standalone diagnostic initially omitted the repository import path and failed
before reading observations. Its first source and failure record remain. The
repair changes only the diagnostic import path. A first minimal audit workspace
also omitted the `evaluation` Python package; its import failure and workspace
remain. The corrected workspace explicitly copies that required source package.

## Preservation and verification

Before this round, the entire experiments tree was archived, selected8 was copied,
and source/ledger/candidate hashes were recorded. The eighteen protected candidate
and ledger files remain exact; earlier frozen plans, outputs and failures remain.
No possession-score rows were added. The latest full suite passes **137 tests**.

Finalization reparses every raw native response and exactly regenerates all score
arrays and reported metrics. It verifies source/input/crop hashes, exclusions and
crop schedules. Crops at three centers are freshly decoded and match byte-for-byte.
Samples from every variant reproduce exactly with independently compiled models.
Ordinary finalization also passes in a separate workspace containing the required
source and explicit immutable input links, with no source archives or baseline
copies. This is an audit with sampled fresh decoding/inference, **not a complete
second generation of all2,436 inferences**. No such full-replay claim is made.

Evidence includes `screen/manifest.json`, `screen/scores.csv`, raw request/output
files, probability arrays, private crops, independent model generations,
`integrity_independent_compilation.json`, `fresh_reference_audit.json`,
`offset_diagnostic.json`, synthetic controls and test logs. All derived inputs and
outputs stay under ignored `data/`. No competition images, pixel arrays, raw
tracking, private model detections or scene descriptions entered hosted context.
Only code, hashes and aggregate diagnostics were inspected. No commits, uploads,
submissions, publication or external messages occurred. Maximum score and private
full-match accuracy remain unverified.

## Reproduction

Use the pinned Python3.12 environment on macOS and the original local assets. First
reproduce the original centered weak cache and native confirmation metadata using
the preceding appearance/native reports. With fresh output directories:

```sh
.venv/bin/python experiments/prepare_modern_models.py
.venv/bin/python experiments/prepare_modern_ball.py
.venv/bin/python experiments/finalize_modern_ball.py
.venv/bin/python experiments/diagnose_modern_offsets.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
```

Use `--resume` only after a prior process is confirmed stopped. Optional finalizer
`--audit-history` checks the saved before copy; `--compiled-assets` selects an
independent asset generation. Failed weak gates leave confirmation unavailable.
The selected full-match reproduction continues to use [SELECTED_MODEL.md](SELECTED_MODEL.md).
